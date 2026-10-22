# DeepInfer 架构地图

对应 `reference/DeepInfer` @ `9bd9cf2`。路径省略前缀：`csrc/engine/`、`deepinfer/worker/`、`deepinfer/engine/`、`deepinfer/model_executor/`（其下 `layers/`、`models/v4_modules/`、`parallel_utils/`、`eplb/`）。这份地图是读路线图之前的底图，也是 agent 现场备课时找代码的索引。

## 0. 规模

| 层 | 主要文件 | 行数 |
|---|---|---|
| C++ 引擎 | `scheduler.hpp` 1329、`sequence.hpp` 419、`zos.cpp` 307、`mtp_varlen.hpp` 275、`block_manager.hpp` 242、`outputs.hpp` 308、`prepare_sample.cpp` 350、`flops_estimator.hpp` 330、`input_metadata/` 1073、pybind 460 | ~5.2k |
| Python 引擎 | `llm_engine.py` 1029、`stats_logger.py` 477、`profile_server.py` 320 | ~2.0k |
| worker | `cache_engine.py` 978、`model_runner.py` 644、`base.py` 357、`worker.py` 283、graph runner 268、`decode_perf_cache.py` 166 | ~2.8k |
| V4 模型 | `attn.py` 1008、`mlp.py` 616、`index_attn.py` 575、`hc.py` 397、`cross_layer.py` 383、`mla_attn.py` 299、`engram*` 399、`mhc_*` 290、`compress.py` 167、`yoco.py` 123；`models/deepseek.py` 525、`decoder_layer.py` 239、`mtp/dflash.py` 392 | ~6.2k |
| 通用层 | `sampler.py` 683、`sampler_utils.py` 196、`attention_backend.py` 146、`mtp_info_pool.py` 78 | ~1.7k |
| TileLang kernel | `tilelang_ops/` 接口 + `cuda/` + `asc/` | ~23k |
| 分布式 | `parallel_state.py` 230、`userbuffer.py` 240、`nccl_memory.py` 100、`cp.py` 54、`eplb/` 792、`distributed/shm_broadcast.py` 461 | ~2.0k |
| 其他 | `structure_output/` 595、`ve/` 979、`deep_cluster/` 656、`hugging_face/` 1375 | ~3.6k |

热路径上的每请求 CPU 工作全部在 C++（调度、序列状态、输出构造、采样输入准备、ZOS 准备），Python 只按步编排。

## 1. 进程与分层

- `LLM`（`entrypoints/llm.py:23`）→ `LLMEngine`（`llm_engine.py:130`）→ driver `Worker` **在同一进程内**（`llm_engine.py:402`）。CP > 1 时才 spawn `cp_size-1` 个 follower 进程（`:377-395`），它们跑 `start_slave_worker_loop`（`base.py:129`）：阻塞在 `cp_broadcast_object`，收到 `(method, kwargs)` 就调用。
- **一卡一进程，world_size = dp × cp。** 每个 DP rank 是一个独立引擎，有自己的 C++ `Scheduler`；调度器跑在该 DP rank 的 cp0 进程。DP rank 之间靠投票保持步型一致（§5）。
- **没有 TP。** `model_parallel_size = context_parallel_size`（`config.py:171`）。生产上 TP 和 CP 都不用，attention 纯 DP，#DP = #EP。
- 默认通信组是 **gloo**（`base.py:334`），NCCL 只在子组里懒初始化，并用一次 dummy allreduce 强制建立，包括单 rank 组（`parallel_state.py:173-187`）。原因：控制面的集合通信（投票、`has_unfinished_requests`）不能碰 CUDA stream，否则 ZOS 的重叠被破坏。
- CP 内的 RPC 走 `MessageQueue`（`shm_broadcast.py:162`）：POSIX shm 无锁环 + ZMQ 溢出通道，读端 `sched_yield` 自旋。

## 2. 请求状态机（`sequence.hpp`、`scheduler.hpp`）

- `SequenceStatus`（`sequence.hpp:30`）：`WAITING, RUNNING, SWAPPED(未用), FINISHED_STOPPED, FINISHED_LENGTH_CAPPED, FINISHED_ABORTED, FINISHED_IGNORED, FINISHED_PREEMPTED, FINISHED_SO_FAILED, FINISHED_VE_FAILED`。`is_finished = status >= FINISHED_STOPPED`。
- `Sequence` 是结构体套结构体（`:265`）：`request / input / generation / cache(KvCacheState) / mtp(MtpState) / structured_output / schedule(SequenceScheduleState)`。关键字段：`cache.block_table`、`cache.state_id`、`cache.num_tokens`、`schedule.total_inflight_tokens`、`schedule.chunked_rest_len`、`mtp.draft_tokens`。
- 三个队列（`scheduler.hpp:1124`）：`waiting`（deque）、`prompted`（deque，PD 分离和 fork 的入口）、`running`（vector）。
- 迁移：
  - `add_sequence`（`:135`）→ `waiting`。SO 请求先异步编译 grammar，`pump()` 发现完成才入队。
  - `_schedule_prefill`（`:351-554`）：队头依次检查死请求 / `schedulable` / `prompt_limit` / `can_allocate`（`NEVER` 拒绝、`LATER` 停）；chunk = `min(剩余, 预算剩余)` 向下对齐 `cp_align`；提交时 `allocate_seq` 分配**整个 prompt** 的 block；整段做完 → `RUNNING` + MTP 槽位，被切 → 保持 `WAITING`，一步最多切一个且排最后。
  - `process_outputs`（`:872-1033`）：prefill 做完 → `running.push_back`（`:952`）；chunk 没完 → `waiting.push_front`（`:1019`）；decode 中 `_check_stop`（`:1035`）只标记状态。
  - `_schedule_decode`（`:559-660`）：`free_finished_seqs` 移出 running（`:752`，不释放 block）；清理 waiting 里的死请求；`prompted` 队头进 running（fork 的做 COW）；按 `output_len` 降序排，从尾部（最短）开始调度，放不下时踢头部（**最长**）；`_append_slot`（`:817`）按 `get_num_blocks()` 增减 block。
  - `_preempt`（`:766`）：`stop_preempted_request` 或 SO 请求 → `FINISHED_PREEMPTED`；否则 `reset_for_recompute` 回 waiting 队头。
  - block 真正释放：`pre_sampler_sync` → `finalize_seq` → `free_seq`（`llm_engine.py:830-846`、`scheduler.hpp:738`）。
- fork / COW（`:1061-1122`）：子序列共享父的 `block_table` 和 `state_id` 并 pin；`branch_id` 以 `_` 开头表示待 COW；prefill 时只克隆尾部半块（`:398-438`），GPU 上 `clone_and_zero_states_and_blocks` 执行，源的 unpin 延迟到 `clone_done_evt`。

## 3. 每一步的执行（ZOS）

`LLMEngine.step`（`llm_engine.py:860`）：`scheduler.schedule` → 克隆 → `post_schedule` 钩子 → SO 簿记 → `execute_model_step`。

`Worker.execute_model_step`（`worker.py:88-189`）是所有步型的统一入口：
1. `_correct_seqlens_qk`（ZOS）→ `execute_main_model`：C++ 准备 pinned 输入或 ZOS 的 GPU 路径，`non_blocking` H2D，在 `forbid_cuda_sync()` 下发射 forward（decode 是 CUDA graph replay）。
2. `_post_process_step(prev_step_state)`：`sample_end_evt.synchronize()`（**每步唯一的同步**，`:263`）→ `.tolist()` 已经 D2H 的 pinned tensor → `LLMEngine.post_process` → `process_outputs` → `build_request_outputs`。
3. `execute_sample`；`pre_sampler_sync_fn`（释放上一步的池对象、`pump()`、`finalize_seq`）；`sample_end_evt.record()`。
4. prompt 步或 ZOS 关 → 当场 post-process；ZOS decode → `prev_step_state = curr`。

ZOS 的机制：
- `schedule(N+1)` 在 `process_outputs(N)` 之前跑，`Sequence` 滞后一步。用 `total_inflight_tokens` 做上界：`start_inflight_step(nextn+1)` 在调度时加（`scheduler.hpp:726`），`finish_inflight_step` 在处理时减；`get_num_blocks`、`seqlen_k` 都含 inflight（`sequence.hpp:291`）。
- `prepare_zos_inputs`（`zos.cpp:142`）算 `remap[i]` = 序列 i 在上一步 batch 里的下标；GPU 上 `compute_seqlens_q_by_confidence_rates`（varlen）和 `compute_seqlens_k_zos` 修正长度；`gather_input_ids_zos` 直接从上一步 GPU 上的采样结果拼 input_ids；`prepare_v4_decode_metadata` 在 GPU 上展开所有 decode 元数据。
- 采样前 `update_metadata_for_sampling_zos`（`zos.cpp:279`）把修正后的 seqlen 写回 `SequenceMetadata`，`refresh_for_sampling`（`sequence.hpp:408`）更新 last token、MTP 读槽位、SO 状态。这就是为什么元数据必须是带指针的池对象。
- 结束检测晚一步（EOS 在 N 步产生，N+1 步才被 `process_outputs(N)` 看到，GPU 多跑一步被丢弃）；block 在 N+2 步才释放，流序保证复用落在那次多余 forward 之后。
- `flush_inflight_reqs` 在队列空时排空挂起的一步。
- `forbid_cuda_sync`（`utils.py:119`）：forward 和采样里任何 torch 可见的 CPU-GPU 同步都抛异常，`Event.synchronize` 除外。

ZOS 关（PROMPT 引擎默认）时 GPU 在 post-process(N) + schedule(N+1) 期间空转。

## 4. 对象池与延迟释放

- `SequenceMetadataPool`：复用 Python wrapper + C++ 存储，每步按 `max_num_seqs` 分配（`scheduler.hpp:706`）。
- `RequestOutputPool`（`outputs.hpp:45`）：预分配 10000 个，`build_request_outputs` 释放 GIL 后返回 uint64 裸指针数组；Python 用 `RequestOutput.clone(ptr)` 才能保留。
- `DeferredRelease`（`llm_engine.py:67`）：收集第 N 步的元数据、输出、克隆 unpin，下一步 `pre_sampler_sync` 时归还。释放工作放在 GPU 忙的时候。
- 多个 C++ 函数返回静态缓冲区的 numpy 视图，约定"下一次调用前消费完"。
- GC：worker 初始化时 `gc.disable()`，每 `DEEPINFER_GC_INTERVAL_STEP`（1000）步在 `pre_sampler_sync` 里手动 collect。

## 5. DP 步型一致（`scheduler.hpp:296-341`）

NORMAL 引擎每步必须全 DP 组同型（prefill 或 decode），因为 EP 的 all-to-all 是组内集合通信。
- `decode_step` 各 rank 一致。ALL 点（每 70 步）：`allreduce MIN(want)`，全员有活才 prefill。ANY 点（每 100 步）：任一 rank 有活就一起做，兜底防饿死。`want = KV 用量 ≤ 0.9 && waiting 队头可调度`。
- 第二轮投票确认真有 rank 排出了 prefill，否则全组转 decode，避免空 prompt 步清零 `decode_step` 后死循环。
- 投了 yes 但没排出的 rank 跑空 prompt 步；没有任何请求的 rank 跑 placeholder 步（1 个 dummy 序列，block 0、slot 0），让所有集合通信照常发生。
- `has_unfinished_requests` 在 ZOS 下用 CPU gloo 组做 `MAX` allreduce。
- `profile_num_available_blocks` 跨 rank 取平均后向下取偶，所有 rank 的 block 数一致。

## 6. 显存与 KV（`cache_engine.py`、`block_manager.hpp`）

- **两种分配器**，同一个 `BlockManager` 类的两个实例：`block_manager`（paged block，128 token/块）和 `state_manager`（每序列一个 state 槽位，`num_state_slots = 1.2 × max_num_seqs + 1`）。`Blocks` 是侵入式双向链表 free list，block 0 保留为空指针。
- cache 种类：
  1. `ca_cache`：主压缩 KV，paged uint8 `[num_blocks, block_size/ratio, 1, 288 或 528]`（FP4 288B = 256 E2M1 + 32 scale；MXFP8 528B）。
  2. `ica_cache`：indexer K，paged uint8 68B/token（128 维 FP4 + 4 UE8M0）。
  3. `swa_cache`：SWA 环，`[num_state_slots, 134, 1, 528]`，slot = `state_id*134 + pos%134`，134 = 128 窗口 + 6 回滚。
  4. `ca_state`：4.2 压缩的 fp32 累加环。
  5. DFlash draft KV：MTP 层的 `swa_cache` 槽位。
  6. 共享层：别名源层的 `ca_cache / ica_cache`，零额外显存。
  7. engram 表（host / GPU）、`MTPInfoPool`（模型级 buffer）。
- `StateBlock`（`:221`）布局：`[stored: producer 层 SWA][dropped: consumer 层 SWA | 压缩状态]`，`state_size_to_disk` 只取 stored 段。这是为 PD 分离和 KV 落盘设计的。
- `KVCacheManager.process`（`:377`）把每层的 index + main 连续放进一个 uint8 block，`kv_offset` 按层索引。原始 block 用 `cuda_utils.alloc_tensor` 在 torch 分配器之外分配。
- 显存切分（`base.py:212`）：按 `max_num_seqs × max_num_batched_tokens` 跑一次 profile，`available = total × utilization − peak − 2 × state_size`，再除以每块字节数。

## 7. V4.1-Flash 模型（`models/`、`v4_modules/`）

- 配置（`hugging_face/config.py:237`）：`model_type == deepseek_v41`，`num_key_value_heads == 1`，FP8 权重 + FP4 专家。每层注意力类型由 `compress_ratios` 决定：0 = 纯 SWA 层；1/2 = MLA latent 压缩 + 稀疏 indexer + SWA。`n_prefill_layer` = 第一个 ratio==1 的层，之前是 YoCo producer，之后是 consumer（40 层，20/20）。
- 骨架：`[MultimodalEmbedding, n_layer × MoeDecoderBlockHC, MainPredictHeadHC, nextn × DFlashNextPredictHeadHC]`（`v4_modules/builder.py:93`）。`DeepseekModel.forward`（`deepseek.py:154`）：embed → CP 镜像重排 → engram 预取 → 层循环（`i == n_prefill_layer` 处切 YoCo）→ DFlash 主 cache 写入 → HC 折叠 → norm → CP all-gather。
- **attention**（`attn.py:228-880`）：`q_a_kv_swa_proj` 融合 q LoRA 和 SWA KV；consumer 层从 `shared_hidden` 投影 KV；prefill 把 `[swa_cached | swa_cur | compressed]` 聚成一个 buffer 跑 `flash_mla_sparse_fwd`；decode 跑 `flash_mla_with_kvcache` 带 `extra_k_cache` 和两套索引；per-head sink。
- **压缩**（`compress.py:82`）：压缩比 1 恒等；压缩比 2 `[score|latent]` 门控累加进 `ca_state` 环，跨 chunk 和回滚都能续；`forward_store_kv_cache` norm+RoPE+cast+store 一个 kernel。
- **稀疏 indexer**（`index_attn.py`）：index-K FP4；prefill 分块算 `fp8_fp4_mqa_logits` + DeepSelect topk；decode 用 paged 版本 + `convert_kv_index`；candidate block 两级稀疏。
- **YoCo**（`yoco.py`、`prepare_tensor_input_v4.cpp:203`）：consumer 层 KV 从第一个 consumer 层的 hidden 投影；`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact` 时每序列只保留最后 `keep_tokens`（140）行，边界处 gather hidden 和 HC bundle；mask 模式等价但不压缩，用于测试。
- **跨层共享**（`cross_layer.py:22`）：三张映射 KV / topk / index-K，约束 `topk ⊆ index-K ⊆ KV`，不能跨 YoCo 边界。
- **mHC**（`hc.py`、`mhc_ops.py`）：4 路残差 `[n, 4, H]`；每个子层前一次 TF32 GEMM 算出 pre / post / comb 三组混合系数，comb 做 Sinkhorn；`hc_lazy` 把上一个子层的 post 和本子层的 pre 融合进一次 `mega_mhc`，顺便输出 FP8 cast 后的 norm 结果。
- **engram**（`engram*.py`）：压缩 tokenizer 把 token 归一化（大小写、空白）；n-gram 哈希到每头素数大小的表；表按 engram 并行组分片，默认放 host，DeepEP RDMA 异步 fetch，第 i 层的 fetch 和前 i 层重叠。
- **MoE**（`mlp.py`）：gate `bf16_mega_gate`（sqrtsoftplus + noaux_tc + EPLB 物理映射在 kernel 内）；专家 MXFP4 打包；默认 Mega MoE：`fp8_fp4_mega_moe` 一个 kernel 做 dispatch/GEMM1/SwiGLU/GEMM2/combine，走 `SymmBuffer`；非融合路径用 DeepEP dispatch → grouped GEMM → combine，shared expert 在 `limit_num_sms` 下和 combine 并行、输出作为 combine 的 bias。**没有 two-batch overlap**。
- **DFlash**（`mtp/dflash.py`）：draft 层吃 target 层特征，Markov 头，置信度头（`:316`），每步采样 `nextn` 个 draft 写进 `MTPInfoPool`；独立 CUDA graph。
- **采样器**（`layers/sampler.py:115-299`）：fp32 logits GEMM → 黑名单 / EOS 抑制 → 强制 top-k（DeepSelect）→ `fused_sample`（温度/top-p/top-k/log-softmax/采样 + counter-based 哈希 RNG 按 `(seq_seed, position, tag)`）+ 融合拒绝采样 → `finalize_verification`。SO 位掩码在 CPU 按 draft 步填、侧 stream 拷贝、kernel 内应用。
- **batch-invariant**：`deep_gemm / tile_kernels.use_deterministic_algorithms`，FlashMLA `enable_batch_invariant`，关闭融合 norm-rope 路径，EPLB 开启时关闭 DeepEP 多重归约。每 rank 内的性质，不跨 DP。

## 8. 投机解码（`mtp_varlen.hpp`、`prepare_sample.cpp`、`sampler.py`）

- 槽位：调度器持有 `free_mtp_info_slots`，RUNNING 时分配、结束/抢占时释放；slot 0 是 padding。`MTPInfoPool` 在 GPU 上存 token / logprob / confidence。
- 输入 `[last committed, draft_0..draft_{q-2}]`；GPU 上 `process_sampling` 对 pool 里的 draft 做拒绝采样，`get_accept_result` 应用 golden 覆盖，`finalize_verification` 输出 `sampled / verified(-1=拒绝) / last_valid_offset`。
- CPU `process_MTPInfo`（`scheduler.hpp:1196`）拷 draft、算累积接受率；`process_outputs` 追加接受的 token，每个都过 `_check_stop`。
- 变长：`MTPVarlen::schedule`（`:150`）用 profile 的 `(graph_size, forward_ms)` 曲线（`DecodePerfCache`，磁盘缓存 1 小时）+ 运行时中位数估计器，按接受率降序分配 `seqlens_q = 1 + wins`，最大化 `期望 token / 步耗时`；ZOS 下预算用 N-2 步的接受率定，分配用 N-1 步的，GPU 再按置信度重分。
- CUDA graph 按 `total_q` 分桶：1,2,4,…8，再每 8 一桶；padding 行用 placeholder 元数据；ZOS 下 `input_on_device`，H2D 在 graph 之外。

## 9. 分布式（`parallel_utils/`、`eplb/`、`cp.py`）

- 分组（`parallel_state.py:51-126`）：`dp / cp / ep / ep_in_dp / enp / world / local` + gloo 镜像，用 `einops.rearrange` 从一个 rank 向量切出；CP 组是连续 rank 块；engram 并行可以跨多个 DP 副本。`_split_group_cached` 避免重复建 NCCL comm。
- **CP = zigzag 镜像**（`prepare_cp_mirror.py:126`）：序列切成 `2×cp` 块，rank r 拿块 r 和 `2cp-1-r`，平衡因果注意力的计算量。通信是 DeepEP `BucketBuffer` 的拷贝引擎 all-gather（`cp.py:42`），零 SM；`all_gather` 返回可调用，延迟 `wait()`。decode 不用 CP。
- **DeepEP**（`mlp.py:460-510`、`userbuffer.py:63-95`）：一个 `EPBuffer`，大小取 prefill/decode 的最大值；`DEEPINFER_USERBUFFER_COMM_SMS` 自动解出通信用的 SM 数；`limit_num_sms` 把计算限在剩余 SM 上。`userbuffer` 是进程级单例，`close()` 夹在两个 barrier 之间有序销毁。
- **EPLB**（`eplb/__init__.py`、`policy.py`、`shuffle.py`）：`expert_count` 由 dispatch kernel 直接写；窗口统计；policy 两阶段贪心（冗余分配 + 放置），目标是每步峰值而不是均值，全向量化；shuffle 优先用 symm_mem 单层镜像 + 两个 barrier 做本地 `copy_()`，否则 batched NCCL P2P。首次重排提前到间隔的 1/4。
- `nccl_memory.py`：按步采样所有 NCCL comm 的显存，只在变化时打印。

## 10. 引擎类型与 PD 接缝

- `EngineType` NORMAL / PROMPT / COMPLETION。PROMPT 只 prefill、无 CUDA graph、无 MTP、ZOS 关；COMPLETION 只 decode、入口是 `prompted` 队列、必须 `stop_preempted_request`；NORMAL 投票。
- 已实现：`do_copy_state`（把序列的 state 拷进 `copy_state` 副本并记录事件）、`StateBlock` 的 stored/dropped 布局、`try_allocate_seq` + `hit_tokens` + `swa_valid_start` 注入命中。
- 空钩子（留给内部版本）：`init_pd_split, do_save_kvcache, pe_prepare_save_kvcache, pe_do_save_kvcache, should_copy_state, send_processed_prompts, process_output, post_schedule, post_process_model_outputs`（`llm_engine.py:801-857`）。
- 只在文档里的：3FS、kvcache-store、GDR swap-in、负载均衡器、自动扩缩容。

## 11. 值得记住的设计决策

1. 每步只同步一次，而且放在下一步 forward 发射之后。
2. CPU 乐观调度 + GPU 侧修正，而不是让 CPU 等 GPU。
3. 热路径全在 C++ 并释放 GIL；对象和指针都用池。
4. gloo 做默认通信组，NCCL 只在子组，控制面集合通信不碰 CUDA stream。
5. DP 投票 + 第二轮确认 + 空步/placeholder 步，保证 Wide-EP 下步型一致。
6. 两种分配器：paged block 管容量可变的 KV，固定槽位管每序列有界的状态。
7. CP 用 zigzag 镜像 + 拷贝引擎 all-gather，不用 TP 的每层 allreduce。
8. EPLB 统计由 dispatch kernel 免费写，目标是每步峰值。
9. 延迟释放到处都是：正确性靠流序和事件，不靠 CPU 等待。
10. `forbid_cuda_sync` 把"forward 里不许同步"从约定变成可执行的不变量。
