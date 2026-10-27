from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import AllocStatus, BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.finished: list[Sequence] = []  # left the queues in a FINISHED_* state; blocks not yet freed
        self.preempted: list[int] = []
        self.prefill_attempt_interval = config.prefill_attempt_interval
        self.prefill_watermark = config.prefill_watermark
        self.decode_step = 0  # decode steps since the last prefill step (scheduler.hpp:677)

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if seq.num_prompt_tokens > self.max_model_len - 1:
            seq.status = SequenceStatus.FINISHED_IGNORED
        self.waiting.append(seq)

    def abort(self, seq_ids: list[int]):
        """Only marks. Queues and blocks are cleaned up by the next schedule() / step() (scheduler.hpp:144)."""
        ids = set(seq_ids)
        for seq in [*self.waiting, *self.running]:
            if seq.seq_id in ids and not seq.is_finished:
                seq.status = SequenceStatus.FINISHED_ABORTED

    def _want_prefill(self) -> bool:
        """Whether this step should try prefill (scheduler.hpp:308-313). 5.3 wraps a DP vote around it."""
        if not self.waiting:
            return False
        if not self.running:
            return True
        if self.decode_step % self.prefill_attempt_interval:
            return False
        bm = self.block_manager
        return 1 - len(bm.free_block_ids) / len(bm.blocks) <= self.prefill_watermark

    def _prefill(self) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                status, num_cached_blocks = self.block_manager.can_allocate(seq)
                if status == AllocStatus.NEVER:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                    self.finished.append(self.waiting.popleft())
                    continue
                if status == AllocStatus.LATER:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
            if seq.num_scheduled_tokens < num_tokens:
                break
        return scheduled_seqs
    
    def _decode(self) -> list[Sequence]:
        scheduled_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def schedule(self) -> tuple[list[Sequence], bool]:
        for queue in (self.running, self.waiting):
            self.finished += [seq for seq in queue if seq.is_finished]
            live = [seq for seq in queue if not seq.is_finished]
            queue.clear()
            queue.extend(live)

        if self._want_prefill():
            seqs = self._prefill()
            if seqs:
                self.decode_step = 0
                return seqs, True

        if not self.running:
            return [], True
        seqs = self._decode()
        self.decode_step += 1
        return seqs, False

    def preempt(self, seq: Sequence):
        seq.num_preemptions += 1
        self.preempted.append(seq.seq_id)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            if seq.is_finished:
                continue
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if not seq.ignore_eos and token_id == self.eos:
                seq.status = SequenceStatus.FINISHED_STOPPED
            elif seq.num_completion_tokens == seq.max_tokens or seq.num_tokens >= self.max_model_len:
                seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
