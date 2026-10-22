# 路线图：从 nano-vllm 到 DeepInfer

起点 nano-vllm `bb823b3`（tag `baseline`，1.45k 行 Python）。终点 DeepInfer `9bd9cf2`（`reference/DeepInfer`，约 46k 行 Python + 6k 行 C++ + 11k 行 TileLang kernel）。DeepInfer 的架构地图见 [deepinfer-map.md](deepinfer-map.md)，读路线前先看一遍。

## 1. 差距是什么：七条轴

把两个代码库逐文件对照之后，差距可以归成七条轴。路线上的每一步都落在其中一条轴上。

| 轴 | nano-vllm | DeepInfer | 背后的瓶颈 |
|---|---|---|---|
| 1 调度与请求状态机 | 3 个状态、2 个队列；只要 waiting 非空就先做 prefill；victim 取 running 队尾 | 10 种结束原因；`waiting / prompted / running` 三个队列；`NEVER / LATER`；decode 每 70 步才投票插一次 prefill；fork + copy-on-write；abort；inflight 计数 | 生产流量的请求形态 |
| 2 每步执行模型 | `schedule → run → .tolist() → postprocess` 全同步 | 每步只同步一次（`sample_end_evt`），post-process 延后一步；CPU 乐观调度 + GPU 侧修正；C++ 调度器、对象池、`forbid_cuda_sync` 守卫 | GPU 每步只剩几毫秒，CPU 成了瓶颈 |
| 3 解码加速 | 每步 1 token | DFlash 5 个 draft token、融合拒绝采样、置信度变长验证、按 total_q 分桶的 CUDA graph | 访存瓶颈下算力有富余 |
| 4 KV 与显存 | 一个 bf16 大张量；refcount block；hash 链前缀缓存 | 两种分配器（paged block + 每序列 state 槽位）；7 种 cache 共存；FP4/FP8 字节布局；跨层别名零开销 | 长上下文下的 HBM 容量 |
| 5 模型 | Qwen3 dense、GQA、flash-attn | V4.1：MLA 单 KV 头、压缩比 1/2 的压缩 + fp32 状态环、FP4 稀疏 indexer + candidate block、SWA 环、YoCo producer/consumer、mHC 四路残差、engram、Mega MoE | 每个组件回应一个具体瓶颈 |
| 6 分布式 | TP + shm pickle RPC | 没有 TP；DP attention + EP MoE；CP 用 zigzag 镜像 + 拷贝引擎 all-gather；gloo 为默认通信组、NCCL 只在子组；DP 投票保持步型一致；EPLB；SM 在计算和通信之间分区 | Wide-EP 要求所有 rank 每步同型 |
| 7 采样与确定性 | temperature + gumbel argmax | fp32 logits GEMM、强制 top-k、融合采样、按 `(seq_seed, position)` 的哈希 RNG、SO 位掩码、batch-invariant | 训推一致、可回放 |

## 2. 路线设计的原则

1. **引擎线和模型线分开。** DeepInfer 自己就是这样分层的：vLLM 血统的通用外壳（worker、cache engine、graph runner、sampler 管线）和 V4 专属模块。引擎线在单卡 Qwen3 上就能做到和 DeepInfer 对齐，模型线再逐层换成 V4。
2. **每一步以现象开头。** 先用实验把问题复现出来，再分析、再解决，用同一个实验验收。前一步的验收暴露下一步的问题。新手需要方向，所以每一步都会告诉你该关注什么，但结论要你自己得出。理论在分析阶段出场，资料优先引用 `reference/gpu-perf-engineering-resources/`。
3. **硬件成本递增。** CPU → 1×H100 → 8×H100 → 1×B200 → 8×B200。DeepEP、DeepGEMM（FP8）、FlashMLA 都支持 Hopper，所以整个分布式一章可以在 H100 上做；只有 V4 的 FP4 路径和最终对拍需要 B200。
4. **提前埋下架构种子，避免返工。** 有几处设计是后面好几步共同依赖的（见第 6 节），它们在第一次出现的那一步就按最终形态来做，哪怕当时看起来过度设计。
5. **解法以 DeepInfer 为准。** 其他方案只用来做对比。路线固定，不开支线。

## 3. 三个对齐检查点

| 检查点 | 时机 | 怎么验 | 需要的硬件 |
|---|---|---|---|
| **P1 引擎对齐** | 第 2 章 结束 | 调度语义测试集（从 DeepInfer 的 scheduler 行为推导：NEVER/LATER、70 步间隔、按输出长度抢占、inflight 计数、延迟释放）全部通过；nsys 时间线上每步只有一次 CPU-GPU 同步；每步 CPU 耗时拆分和 DeepInfer 同量级 | 1×H100 |
| **P2 模型对齐** | 第 4 章 结束 | V4.1-Flash 前 N 层逐层输出和 DeepInfer 在容差内一致；开 batch-invariant 时 logprob 按位一致 | 1×B200 |
| **P3 系统对齐** | 第 5 章 结束 | 8 卡跑 V4.1-Flash，逐样本 decode PPL 和 DeepInfer 自带的黄金基线 `reference/DeepInfer/tests/models/offline_decode_ppl.json` 对上；吞吐达到 DeepInfer 的目标比例 | 8×B200 |

P3 是整条路线的期末考试：DeepInfer 自己的回归测试就是我们的验收标准。

---

## 4. 路线（按章节）

每一节的格式：**现象**（怎么发现）→ 本质 → DeepInfer 的解法（`reference/DeepInfer/` 下的位置，`csrc/engine/`、`deepinfer/worker/`、`deepinfer/model_executor/` 等前缀省略）→ 验收。

### 第 1 章：地基（任意 GPU）

| 步 | 现象 | 本质 | 做什么 | 验收 |
|---|---|---|---|---|
| **0** | nano-vllm 只打印一个总吞吐；改了模型代码也不知道对不对 | 看不见就发现不了问题；判不了对错就不敢改 | 观测：固定负载场景（稳态 decode、长短混合、KV 高压、高并发小请求）、TTFT / TBT 分布、每步耗时拆分、调度决策 trace 导出、nsys 脚本、MFU 估算（对照 `flops_estimator.hpp`）。正确性：按 seed 固定采样，PPL / KL 对 HF 参考实现回归（对照 `tests/models/test_kl.py` 和 `offline_decode_ppl.json` 的做法） | 基线可复现；改一行模型代码能被 KL 测试抓住 |

### 第 2 章：引擎控制面（1×H100，Qwen3-8B；CPU 开销实验用 Qwen3-0.6B 放大）

轴 1、2、3、7。做完达到 **P1 引擎对齐**。

| 步 | 现象 | 本质 | DeepInfer 的解法 | 验收 |
|---|---|---|---|---|
| **2.1** 请求状态机 | 发一个比显存总量还大的 prompt、中途 abort 一个请求：引擎卡死或断言 | 资源判断只有"放得下/放不下"，没有"永远放不下"；判定结束、出队、释放 block 混在一处 | `NEVER/LATER`（`block_manager.hpp:128`）；10 种 `SequenceStatus`（`sequence.hpp:30`）；abort 只标记、下一次调度再清理（`scheduler.hpp:144`）；`_check_stop` → `free_finished_seqs` → `free_seq` 分三处（`scheduler.hpp:1035/:752/:738`） | 边界请求测试全过；状态迁移和 DeepInfer 一致 |
| **2.2** prefill/decode 交替 | 长短混合负载，TBT p99 周期性尖刺 | waiting 非空就先 prefill，decode 被卡住（generation stall） | `decode_step` 每 70 步一个尝试点、KV 用量 90% 水位线、prefill 开始后连着做（`scheduler.hpp:275-345`）；被切的请求排最后、放回队头（`:542`、`:1019`）；空 prompt 步的概念（单卡时退化） | TBT p99 下降；TTFT 的代价能量化 |
| **2.3** 抢占 | KV 高压负载，抢占次数和重算 token 数很高 | victim 选择和抢占方式没考虑重算代价 | 按输出长度排序，踢最长的（`scheduler.hpp:622-646`，注意注释和代码不一致）；`reset_for_recompute`（`sequence.hpp:307`）；`stop_preempted_request` 开关 | 重算 token 数下降；吞吐抖动变小 |
| **2.4** fork 与 COW | `n=4` 采样时同一个 prompt 被 prefill 了 4 次 | 子序列没有共享父序列的 KV | `fork_seq` 共享 block_table + state 并 pin（`scheduler.hpp:1061`）；`branch_id` 前缀 `_` 标记待 COW；只克隆尾部半块（`:398-438`）；`prompted` 队列 | n=4 的 prefill 成本接近 n=1 |
| **2.5** CPU 开销与对象池 | 0.6B 模型高并发，nsys 上 GPU 有大片空隙 | 每步 Python 建 list、建 tensor、建对象的开销和 GPU 时间同量级 | 输入准备直接写进预分配的 pinned buffer（`csrc/engine/input_metadata/`）；`SequenceMetadataPool`、`RequestOutputPool` 返回裸指针数组（`outputs.hpp:45`）；`DeferredRelease` 把释放放到 GPU 忙的时候（`llm_engine.py:67`）；关 GC、定期手动 collect（`base.py:102`、`worker.py:165`） | 每步 CPU 耗时拆分；对象分配次数接近 0 |
| **2.6** 调度器下沉 C++ | profile 显示调度本身是 O(n) 的大头 | 解释器开销随 batch 线性涨 | `scheduler.hpp` + pybind，持锁时释放 GIL（`:276`）；block 用侵入式双向链表 free list（`block_manager.hpp:27`） | 调度耗时下降一个量级 |
| **2.7** ZOS | 调度变快了，GPU 在 post-process 和 schedule 期间仍空着 | CPU 和 GPU 串行，中间一次 `.tolist()` 同步 | 每步只同步 `sample_end_evt`，且放在下一步 forward 发射之后（`worker.py:144/:263`）；post-process 延后一步；`total_inflight_tokens` 做上界估计（`sequence.hpp:244`）；`remap` + GPU 侧 gather 输入（`zos.cpp:142`、`gather_input_ids_zos.py`）；结束判定晚一步、block 晚两步释放；`flush_inflight_reqs`；`forbid_cuda_sync` 守卫（`utils.py:119`）；CUDA graph 改 `input_on_device` | nsys 上空隙消失；吞吐提升和 2.5 测的 CPU 占比对得上 |
| **2.8** 投机解码（定长） | 低并发 decode，吞吐上不去，算力利用率极低 | decode 卡在访存，算力有富余 | draft 来源用 Qwen3-8B 的 EAGLE-3 头（`RedHatAI/Qwen3-8B-speculator.eagle3`），结构上和 DFlash 一样是 draft 层吃 target 特征。机制对齐 DeepInfer：MTP 槽位随 RUNNING 分配（`scheduler.hpp:125`）、`MTPInfoPool`（`layers/mtp_info_pool.py`）、输入 `[last, draft...]`（`prepare_tensor_input_base.cpp:104`）、融合拒绝采样（`layers/sampler_utils.py:115`）、ZOS 下用 `compute_seqlens_k_zos` 修正 K 长度、`MAX_BACKTRACKING` 回滚、CUDA graph 按 total_q 分桶（`cuda_graph_runner.py:41`） | 低并发吞吐提升；接受率曲线 |
| **2.9** 变长验证 | 开了投机解码，高并发下吞吐反而比不开低 | 高并发时算力不再富余，定长验证浪费 | `MTPVarlen::schedule`：profile 出 `(graph_size, forward_ms)` 曲线（`DecodePerfCache`），运行时估计器，按接受率给每个请求分配 draft 长度（`mtp_varlen.hpp:150`）；ZOS 下 CPU 只定总预算、GPU 按置信度重分（`schedule_mtp_varlen.py`；EAGLE-3 没有置信度头，先用逐 token 的 draft 概率代替） | 吞吐随并发的曲线全程不低于关闭投机 |
| **2.10** 采样器 | 同一 seed 两次结果不同；top-p 很慢 | RNG 状态依赖调用顺序；采样是多个 kernel 串起来的 | fp32 logits GEMM、`DEEPINFER_SAMPLER_FORCE_TOPK` 把词表压到 top-K 空间、`fused_sample`（温度/top-p/top-k/log-softmax/采样一个 kernel）、counter-based 哈希 RNG 按 `(seq_seed, position, tag)` 取数（`random_sampling.py`）、`sum_smaller_probs` | 固定 seed 可回放；采样耗时下降 |
| **2.11** 结构化输出 | 混入带 JSON schema 的请求，所有请求的延迟都被拖高 | grammar 编译阻塞了主循环 | `SOMatcherBuilder` 线程池异步编译、`pump()` 轮询（`structure_output/so_matcher_builder.py:97`）；位掩码按 draft 步填、侧 stream 拷贝（`layers/sampler.py:302`）；`rollback` 配合投机解码；超时 → `FINISHED_SO_FAILED` | SO 请求不影响其他请求的 TBT |
| **2.12** batch-invariant | 同一 prompt 放进不同 batch，logprob 不一样 | reduction 顺序依赖 batch 组成 | 固定 GEMM / attention 的 split 和归约顺序（`runtime_config.py:67`、`layers/attention_backend.py:141`）；确定性采样要求 varlen 关闭 + 固定 `seq_seed`；测试：单发 vs 64 并发 token 和 logprob 完全一致（`tests/models/test_batch_invariant.py`） | 测试通过 |

第 2 章结束时，引擎外壳的结构已经和 DeepInfer 一致：C++ 调度器、worker 的 `execute_model_step` 三段式、对象池、守卫、采样管线。后面换模型不动这一层。

### 第 3 章：从 dense 到 MLA + MoE（1×H100，DeepSeek-V2-Lite）

轴 4、5 的前半。V2-Lite 是 V4 和 Qwen3 之间最近的中间站（MLA + MoE，16B 单卡放得下）。

| 步 | 现象 | 本质 | DeepInfer 的解法 | 验收 |
|---|---|---|---|---|
| **3.1** MLA | 第 2 章的 KV 容量分析换算到 70B 级 dense 模型：每 token KV 太大，batch 开不起来 | KV 容量 | 单 KV 头的 latent cache；decode 时权重吸收（对照 `models/v4_modules/mla_attn.py`、FlashMLA 的 `flash_mla_with_kvcache` 接口） | 同样显存下 max batch 提升；KL 回归通过 |
| **3.2** MoE | dense 每步读全部权重，访存不够 | 每 token 只读部分专家 | gate（sqrtsoftplus / noaux_tc，`v4_modules/mlp.py:359`）、routed + shared、grouped GEMM、权重打包（`v4_modules/containers.py:62`） | 单卡 MoE 跑通；KL 通过 |
| **3.3** FP8 | 权重和激活的访存仍是大头 | 低精度 | DeepGEMM Hopper FP8 GEMM、per-token cast 融合进 norm（`tilelang_ops/norm.py`）、UE8M0 scale 布局 | 吞吐提升；精度损失可量化 |

### 第 4 章：V4.1-Flash 逐层（1×B200）

轴 4、5 的后半。只加载 V4.1-Flash 前几层的真实权重，每实现一层就和 DeepInfer 的同一层对拍。做完达到 **P2 模型对齐**。V4 的 kernel 直接用 vendor 库（FlashMLA、DeepGEMM、DeepSelect、TileKernels），学习者写的是胶水、cache 布局、metadata，以及几个自己动手的 TileLang kernel。

| 步 | 现象 | 本质 | DeepInfer 的解法 | 验收 |
|---|---|---|---|---|
| **4.1** 多种 cache 共存 | V4 每层的 cache 类型不一样，nano 的一个大张量放不下 | 需要按层异构的字节布局和两种分配器 | `KVCacheManager` 每层 index + main 连续放进一个 uint8 block（`cache_engine.py:377`）；`StateBlock` = `[stored SWA | dropped: consumer SWA / 压缩状态]`（`:221`）；`state_manager` 是第二个 `BlockManager` 实例（`llm_engine.py:265`）；`profile_num_available_blocks` 跨 rank 取平均（`base.py:212`） | 布局和 DeepInfer 字节级一致 |
| **4.2** SWA 环 + sink | 纯 SWA 层的 KV 不需要 paged | 窗口固定，环形缓冲最省 | `swa_cache [num_state_slots, 134, 1, 528]`，slot = `state_id*134 + pos%134`（`prepare_tensor_input_v4.cpp:98`）；134 = 128 窗口 + 6 回滚；per-head sink logits | 对拍通过 |
| **4.3** MLA 压缩 KV（压缩比 1/2） | 压缩层的 KV 是按窗口聚合的，chunk 边界会断 | 需要跨步保留的 fp32 累加状态 | `Compressor`：压缩比 1 恒等，压缩比 2 `[score|latent]` 门控累加进 `ca_state` 环（`v4_modules/compress.py:82`）；`forward_store_kv_cache` norm+RoPE+cast+paged store 一个 kernel；288B/token FP4 格式（`tilelang_ops/cache_common.py:119`） | 对拍通过 |
| **4.4** 稀疏 indexer | 长上下文 attention 访存随长度线性涨 | 只读 top-k | index-K FP4 68B/token；prefill 用 `fp8_fp4_mqa_logits` 分块 + DeepSelect topk（`v4_modules/index_attn.py:327`）；decode 用 paged 版本 + `convert_kv_index`；candidate block 两级稀疏（`:204`） | 对拍通过 |
| **4.5** YoCo + bounded replay | 后 20 层的 KV 和前 20 层强相关，prefill 算了两遍 | consumer 层 KV 从第一个 consumer 层的 hidden 投影 | `n_prefill_layer` 分界；`shared_hidden`；`fill_yoco_truncation_metadata` 只保留最后 140 行（`prepare_tensor_input_v4.cpp:203`）；compact 模式在边界 gather（`v4_modules/yoco.py:93`）；mask 模式用于等价测试 | compact 和 mask 模式结果一致；prefill 计算接近减半 |
| **4.6** 跨层共享 | consumer 层各自存 KV 浪费一半容量 | 别名 | `SharedLayerTopology` 三张映射（KV / topk / index-K，`v4_modules/cross_layer.py:22`）；cache engine 给共享层别名张量（`cache_engine.py:429`） | KV 显存占用减半 |
| **4.7** mHC | 四路残差每个子层都要读写 4×H | 访存 | `hc_lazy` 把上一个子层的 post 和这个子层的 pre 融合成一次 `mega_mhc`（`v4_modules/hc.py:229`）；Sinkhorn 归一化 | 对拍通过 |
| **4.8** engram | n-gram 表比 HBM 大 | 容量 | 压缩 tokenizer（`v4_modules/engram_metadata.py:16`）；表放 host，DeepEP RDMA 异步 fetch，第 i 层的取数和前 i 层重叠（`models/deepseek.py:180`）。单卡先做本地版本 | 对拍通过 |
| **4.9** Mega MoE + FP4 专家 | 3.2 的 MoE 是多个 kernel，B200 上访存和 launch 都浪费 | 融合 | `fp8_fp4_mega_moe` 一个 kernel 做 dispatch/GEMM/SwiGLU/combine（`v4_modules/mlp.py:525`）；`SymmBuffer`；norm kernel 直接写进 sym buffer | 对拍通过；吞吐提升 |
| **4.10** DFlash + 置信度头 | 2.8 的 EAGLE-3 头是 Qwen3 的，V4 用不了；而且它没有置信度输出，2.9 只能用接受率近似 | V4 自带的 draft 头和置信度头 | `DFlash.decode_forward`：draft 层吃 target 特征，Markov 头，置信度头（`models/mtp/dflash.py:279`）；独立 CUDA graph（`dflash_cuda_graph_runner.py`）；2.9 的 GPU 重分现在用真正的置信度 | 接受率和 DeepInfer 一致 |
| **4.11** 自己写 kernel | 胶水里还有几个小算子是 torch 拼的 | launch 和访存 | 用 TileLang 重写 `fused_norm_rope_hadamard_cast_store_kvcache`、`gather_input_ids_zos`、`kv_index_cat` 中的至少一个，对照 `tilelang_ops/cuda/` | 数值一致；耗时不高于原版 |

### 第 5 章：分布式（8×H100 用 V2-Lite 做机制，最后 8×B200 跑 V4.1-Flash）

轴 6。做完达到 **P3 系统对齐**。

| 步 | 现象 | 本质 | DeepInfer 的解法 | 验收 |
|---|---|---|---|---|
| **5.1** DP + EP | 单卡上每个专家分到的 token 太少，读一遍专家权重只服务几个 token | 把专家分散到多卡，提高每个专家的有效 batch | 去掉 TP；`ParallelProcessGroup` 的 dp/ep/cp/enp 分组（`parallel_utils/parallel_state.py:51`）；一卡一进程，调度器在每个 DP rank 的 cp0 进程；naive all-to-all | 每专家 batch 提升；吞吐随 EP 增长 |
| **5.2** DeepEP | naive all-to-all 占满 SM，和计算串行 | 通信和计算抢 SM | `EPBuffer` dispatch/combine（`v4_modules/mlp.py:460-510`）；`limit_num_sms` 把 shared expert 限在少量 SM 上和 combine 并行，shared 输出作为 combine 的 bias（`:588`）；`userbuffer` 单例、有序销毁（`parallel_utils/userbuffer.py:227`） | 通信时间被隐藏 |
| **5.3** DP 步型一致 | 一个 rank 做 prefill、另一个做 decode，all-to-all 卡死 | 集合通信要求每步同型 | gloo 默认通信组、NCCL 懒初始化子组（`base.py:334`、`parallel_state.py:173`）；ALL/ANY 两种投票点 + 第二轮确认（`scheduler.hpp:296-341`）；空 prompt 步和 placeholder 步；`has_unfinished_requests` 的 CPU allreduce；block 数跨 rank 取平均 | 混合负载下不卡死；ZOS 仍然每步一次同步 |
| **5.4** EPLB | 热点专家所在的 rank 拖慢整步 | 负载不均 | 统计由 dispatch kernel 免费写（`mlp.py:474`）；policy 以"每步峰值"为目标的向量化贪心（`eplb/policy.py:12`，无 GPU 可单测）；symm_mem 单层镜像 + 两个 barrier 的 shuffle（`eplb/shuffle.py:178`），P2P 兜底 | `moe_lb` 接近 1.05 |
| **5.5** CP | 超长 prompt 单卡算不完 | 按 token 切分 prefill | zigzag 镜像切分（`tilelang_ops/prepare_cp_mirror.py:126`）、`cp_align`、拷贝引擎 all-gather 零 SM（`parallel_utils/cp.py:42`）、`all_gather` 返回可调用延迟等待 | 长 prompt TTFT 随 CP 下降 |
| **5.6** 引擎类型与 PD 接缝 | P 和 D 卡在不同资源上却挤在一批卡上 | 分开配卡 | `EngineType` PROMPT/COMPLETION/NORMAL（`scheduler.hpp:286`）；`prompted` 队列作为 COMPLETION 的入口；`try_allocate_seq` + `hit_tokens` 注入；`do_copy_state` 和 `state_size_to_disk`；虚钩子 `do_save_kvcache` 等（`llm_engine.py:801`）。做机内 KV 传输的最小实现 | 1P+1D 在单机跑通 |
| **5.7** V4.1-Flash 全模型（8×B200） | 前面都是分块对拍，没有端到端数字 | 期末考试 | 装配第 4 章全部组件 + 第 5 章的并行，跑 `examples/offline_inference.py` 同样的配置 | 逐样本 PPL 对上 `offline_decode_ppl.json`；吞吐达标 |

### 第 6 章：开源版没有的系统层（可选，简化实现）

DeepInfer 开源版只留了钩子，这部分只能照官方文档做简化版，不要求和代码对齐。

| 步 | 内容 | 文档出处（`reference/DeepInfer/doc/inference_deployment_qa.md`） |
|---|---|---|
| 6.1 | KV 分层存储：host + 本地盘，hash 链前缀匹配，异步 fetch，逐层 swap-in | KV cache storage |
| 6.2 | E/P/D 分离服务（进程级） | Inference system overview |
| 6.3 | DP 负载均衡器 | Data-parallel load balancing |
| 6.4 | VE 分离：只加载 ViT 权重的独立进程（`examples/_ve_helper.py`） | Inference system overview |
| 6.5 | 多协议编解码与流式解析（对照 deepseek-recipe） | DeepSeek Recipe |

---

## 5. 依赖图

```
1.1 ──► 2.1 ──► 2.2 ──► 2.3 ──► 2.5 ──► 2.6 ──► 2.7 ──► 2.8 ──► 2.9
       │                                   │       │
       └──► 2.4 (任意时候)                  │       └──► 2.10 ──► 2.11
                                           │               └──► 2.12
                                           ▼
                                      3.1 ──► 3.2 ──► 3.3
                                                     │
                     ┌───────────────────────────────┤
                     ▼                               ▼
              4.1 ──► 4.2 ──► 4.3 ──► 4.4 ──► 4.5    5.1 ──► 5.2 ──► 5.3 ──► 5.4
                                    │    │                      │
                                    │    └──► 4.6 ──► 4.7 ──► 4.8  └──► 5.5 ──► 5.6
                                    │                     │              │
                                    └──► 4.9 ──► 4.10 ──► 4.11              │
                                                          │              │
                                                          └──────► 5.7 ◄──┘
                                                                   │
                                                                   ▼
                                                              6.1 … 6.5
```

第 4 章和第 5 章可以交错做（第 4 章要 B200，第 5 章要 8×H100），谁先租到卡先做谁。

## 6. 提前埋下的架构种子

下面这些设计被后面好几步共同依赖。在第一次出现的那一步就按 DeepInfer 的最终形态来做，否则后面要重写。

| 种子 | 第一次出现 | 后面谁依赖它 | DeepInfer 位置 |
|---|---|---|---|
| 调度元数据是**带指针的池对象**，不是快照。`SequenceMetadata` 指向 `Sequence`，采样前可以 `refresh_for_sampling` | 2.5 | 2.7（ZOS 要在采样前改 seqlen）、2.8/2.9（MTP 槽位） | `sequence.hpp:333`、`:408` |
| 请求状态里有 **inflight 计数**，`get_num_blocks` 和 `seqlen_k` 都按上界算 | 2.1 预留字段，2.7 启用 | 2.7、2.8 | `sequence.hpp:244`、`:291` |
| 输入准备写进**预分配 pinned buffer**，worker 接收的是 tensor 不是 Python 对象 | 2.5 | 2.6、2.7（`input_on_device`）、5.1（CP worker） | `csrc/engine/input_metadata/` |
| **两种分配器**：paged block 和每序列 state 槽位是同一个 `BlockManager` 类的两个实例 | 2.1 预留，4.1 启用 | 4.2（SWA 环）、4.3（压缩状态）、5.6（state 拷贝） | `block_manager.hpp:119`、`llm_engine.py:265` |
| 采样用 **counter-based RNG**，按 `(seq_seed, position, tag)` 取数 | 2.10 | 2.12、2.8 的回放、4.10 | `tilelang_ops/random_sampling.py` |
| **`forbid_cuda_sync` 守卫**包住 forward 和采样 | 2.7 | 之后每一步 | `deepinfer/utils.py:119` |
| 调度器的 DP 投票用 **CPU 通信组**，`dp_size == 1` 时退化 | 2.2 写接口，5.3 启用 | 5.3 | `scheduler.hpp:169`、`llm_engine.py:313` |

## 7. 硬件与时间

| 章 | 硬件 | 节数 | 预计时间 |
|---|---|---|---|
| 1 + 2 | 1×H100 | 13 | 60–90 h |
| 3 | 1×H100 | 3 | 15–25 h |
| 4 | 1×B200 | 11 | 50–80 h |
| 5 | 8×H100，最后一节 8×B200 | 7 | 40–60 h |
| 6 | 不定 | 5 | 可选 |

合计 34 节（不含第 6 章），约 170–260 小时。每一步加上理论大约 3–8 小时。GPU 价格开工前查 Runpod；5.7 要一次性集中几天做。

两个还没验证的前提：(1) 从 V4.1-Flash 的 safetensors 只读前几层权重、让 DeepInfer 也只跑前几层做对拍是否可行，进入第 4 章前确认，不行就用随机权重加小配置；(2) `bf16_mega_moe` 是否支持 Hopper，影响 5.2 在 H100 上能做到哪一步。

## 8. 进度

各节的文档写完就打勾（文档说明这个 PR 要做什么）。参考实现由维护者验证后另打 `<节>-done` tag。

- [x] 1.1 观测与正确性（[文档](1.1-观测与正确性.md)）
- [x] 2.1 请求状态机（[文档](2.1-请求状态机.md)）
- [x] 2.2 prefill/decode 交替（[文档](2.2-prefill-decode交替.md)）
- [x] 2.3 抢占（[文档](2.3-抢占.md)）
- [x] 2.4 fork 与 COW（[文档](2.4-fork与COW.md)）
- [x] 2.5 CPU 开销与对象池（[文档](2.5-CPU开销与对象池.md)）
- [x] 2.6 调度器下沉 C++（[文档](2.6-调度器下沉C++.md)）
- [x] 2.7 ZOS（[文档](2.7-ZOS.md)）
- [x] 2.8 投机解码（定长）（[文档](2.8-投机解码.md)）
- [x] 2.9 变长验证（[文档](2.9-变长验证.md)）
- [x] 2.10 采样器（[文档](2.10-采样器.md)）
- [x] 2.11 结构化输出（[文档](2.11-结构化输出.md)）
- [x] 2.12 batch-invariant（[文档](2.12-batch-invariant.md)）
- [x] 3.1 MLA（[文档](3.1-MLA.md)）
- [x] 3.2 MoE（[文档](3.2-MoE.md)）
- [x] 3.3 FP8（[文档](3.3-FP8.md)）
- [x] 4.1 多 cache 共存（[文档](4.1-多cache共存.md)）
- [x] 4.2 SWA 环 + sink（[文档](4.2-SWA环与sink.md)）
- [x] 4.3 MLA 压缩 KV（[文档](4.3-压缩KV.md)）
- [x] 4.4 稀疏 indexer（[文档](4.4-稀疏indexer.md)）
- [x] 4.5 YoCo + bounded replay（[文档](4.5-YoCo.md)）
- [x] 4.6 跨层共享（[文档](4.6-跨层共享.md)）
- [x] 4.7 mHC（[文档](4.7-mHC.md)）
- [x] 4.8 engram（[文档](4.8-engram.md)）
- [x] 4.9 Mega MoE + FP4 专家（[文档](4.9-MegaMoE与FP4.md)）
- [x] 4.10 DFlash + 置信度头（[文档](4.10-DFlash.md)）
- [x] 4.11 自己写 kernel（[文档](4.11-自己写kernel.md)）
