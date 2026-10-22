<p align="center">
  <img src="assets/cover.png" alt="vllm2deepinfer" width="100%">
</p>

<h1 align="center">vllm2deepinfer</h1>

<p align="center">
  从 nano-vllm 出发，一个 PR 一个 PR 地把它搓成 DeepInfer。<br>
  边学边做，弄懂推理引擎的 why 和 how。
</p>

<p align="center">
  <a href="https://github.com/justkyriecai/vllm2deepinfer/stargazers"><img src="https://img.shields.io/github/stars/justkyriecai/vllm2deepinfer?style=flat-square" alt="stars"></a>
  <a href="https://github.com/justkyriecai/vllm2deepinfer/network/members"><img src="https://img.shields.io/github/forks/justkyriecai/vllm2deepinfer?style=flat-square" alt="forks"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-MIT-blue?style=flat-square" alt="code license"></a>
  <a href="doc/README.md"><img src="https://img.shields.io/badge/docs-CC%20BY%204.0-lightgrey?style=flat-square" alt="docs license"></a>
</p>

> **状态**：路线图已定稿，正在写第 1 章。进度见 [doc/README.md](doc/README.md#8-进度)。

## 关于

[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 用 1.4k 行 Python 写出了一个能跑的推理引擎，读完它你知道 continuous batching 和 paged KV 是怎么回事。[DeepInfer](https://github.com/deepseek-ai/DeepInfer) 是 DeepSeek 生产推理引擎的开源版，46k 行 Python、6k 行 C++、2 万多行 kernel，读它像在读一座城市的管网图。

两者之间差了什么？差的部分，每一处都是为了解决一个具体的瓶颈：GPU 每步只剩几毫秒时 CPU 成了瓶颈，于是有了 C++ 调度器和异步调度；长上下文 KV 放不下，于是有了 MLA、压缩、稀疏 indexer、YoCo；Wide-EP 要求每个 rank 每步同型，于是调度器里出现了投票。

这个仓库把这段差距拆成 **34 节增量 PR**（外加 5 节可选的系统层）。每一节先用实验把问题在当前代码上复现出来，追问到本质，再对照 DeepInfer 的做法自己动手解决，最后用同一个实验验收。做完一节，下一个瓶颈自己浮出来。

> 本项目是个人学习项目，与 DeepSeek 官方没有关联。

## 你会学到什么

- **引擎控制面**：请求状态机、prefill/decode 交替、抢占、fork 与 copy-on-write、对象池、C++ 调度器、零开销异步调度（ZOS）
- **解码加速**：投机解码的槽位和拒绝采样、按置信度的变长验证、CUDA graph 分桶、确定性采样、batch-invariant
- **KV 与显存**：paged block 与 state 槽位两种分配器、多种 cache 共存的字节布局、FP4/FP8 格式、跨层共享
- **DeepSeek-V4.1-Flash 的每个组件**：MLA 压缩 KV、FP4 稀疏 indexer、SWA 环、YoCo + bounded replay、mHC、engram、Mega MoE、DFlash
- **分布式**：DP attention + EP MoE、DeepEP 与 SM 分区、DP 步型一致、EPLB、zigzag CP、P/D 引擎拆分
- 以及一个贯穿全程的习惯：**先看见问题，再动手**

## 前置要求

- **Python**：能读写几百行的工程代码；会用 pytest
- **PyTorch**：知道 tensor、stream、CUDA graph 大概是什么；写过模型前向更好
- **推理基础**：读过 vLLM 论文或 nano-vllm 的代码。没读过也行，第 1 章会补
- **C++**（第 2 章后半起）：能看懂并改写几百行 C++；会用 pybind11 更好
- **硬件**：第 1–3 章一张 H100；第 4 章一张 B200；第 5 章 8×H100，最后一节 8×B200。没有卡怎么办见 FAQ
- **一个 agent**：Claude Code、Codex、Cursor 之类。本仓库的教学方式是 agent 当导师、你写代码

## 路线

完整版在 [doc/README.md](doc/README.md)：七条差距轴、设计原则、三个对齐检查点、每一节的现象/本质/DeepInfer 解法/验收、依赖图、提前埋下的架构种子。读路线前建议先看 [DeepInfer 架构地图](doc/deepinfer-map.md)。

| 章 | 节 | 内容 | 硬件 | 状态 |
|---|---|---|---|---|
| **1 地基** | 1.1 | 观测工具与正确性回归 | 任意 GPU | 📝 |
| **2 引擎控制面** | 2.1–2.12 | 请求状态机 → P/D 交替 → 抢占 → fork → 对象池 → C++ 调度器 → ZOS → 投机解码（EAGLE-3）→ 变长验证 → 采样器 → 结构化输出 → batch-invariant | 1×H100，Qwen3-8B | ⬜ |
| **3 MLA + MoE** | 3.1–3.3 | MLA → MoE → FP8 | 1×H100，DeepSeek-V2-Lite | ⬜ |
| **4 V4.1-Flash 逐层** | 4.1–4.11 | 多 cache 共存 → SWA 环 → 压缩 KV → 稀疏 indexer → YoCo → 跨层共享 → mHC → engram → Mega MoE → DFlash → 自己写 kernel | 1×B200 | ⬜ |
| **5 分布式** | 5.1–5.7 | DP+EP → DeepEP → 步型一致 → EPLB → CP → P/D 引擎拆分 → 全模型期末考试 | 8×H100；5.7 用 8×B200 | ⬜ |
| **6 系统层**（可选） | 6.1–6.5 | KV 分层存储、E/P/D 服务、负载均衡、扩缩容、Recipe | 不定 | ⬜ |

📝 写作中　⬜ 计划中　✅ 完成

三个检查点：第 2 章结束时引擎结构和 DeepInfer 对齐（单卡可验）；第 4 章结束时 V4 逐层数值对齐；5.7 用 DeepInfer 自带的 PPL 黄金基线当期末考试。

## 快速开始

```bash
git clone https://github.com/<你>/vllm2deepinfer.git && cd vllm2deepinfer   # 先 fork
bash scripts/setup_reference.sh      # 拉取 reference/ 下的 DeepInfer（含 vendor 子模块）和资料清单
```

然后在仓库根目录打开你的 agent，说"开始 1.1"。Claude Code 里也可以直接 `/start-step 1.1`。

每一节的流程：

| 环节 | 你做什么 | agent 做什么 |
|---|---|---|
| 发现 | 预测实验结果，亲手跑 | 给方向和实验，问"哪个数字不对劲" |
| 分析 | 说出根本原因 | 一层层追问，这时才讲理论 |
| 解决 | **自己写代码** | 带你读 DeepInfer 的做法，标出 TODO 的位置，卡住时按层级给提示 |
| 验收 | 解释数字为什么说明问题解决了 | `/check-step`：重跑同一个实验 + 回归测试 |
| 对照 | 读 diff | `/compare-step`：和参考实现、DeepInfer 对比，收一张卡片 |

每一节有起止 tag（`2.3-start` / `2.3-done`），`2.3-start` 就是 2.2 的参考答案。卡住了或想跳过，直接从下一节的 tag 开始。做完在**你自己的 fork** 里提 PR（`learn/2.3` → `main`）当存档，不要提到本仓库。

### 项目结构

```
nanovllm/                 引擎代码，随路线逐节演进
tests/                    每一节的验收测试（test_2_3.py）
doc/
  README.md               路线图
  deepinfer-map.md        DeepInfer 架构地图，带 文件:行号
  <节>-<主题>.md          每一节的文档：现象 → 分析 → 解决 → 验收
  cards/                  每节结束收的卡片
reference/
  DeepInfer/              DeepInfer 源码（submodule，固定版本），阅读和做实验用
  gpu-perf-engineering-resources/   理论资料清单（submodule）
AGENTS.md                 给 agent 的教学规则；CLAUDE.md 指向它
.claude/skills/           start-step / check-step / compare-step / learn-codebase
```

## FAQ

**没有 GPU 能学吗？**
第 1 章和第 2 章前半（2.1–2.4）的调度逻辑可以在 CPU 上跑测试，但"发现问题"那一环要看真实的 TBT 和 nsys，没有卡就只能读文档和卡片。建议至少租一张 H100 做第 2 章，这是全路线性价比最高的部分。

**没有 B200 呢？**
第 1–3 章和 5.1–5.6 都在 H100 上。第 4 章和 5.7 需要 B200，因为 DeepInfer 只支持 SM100，对拍必须在它能跑的卡上做。可以先做完其余部分，B200 集中租几天。

**为什么不直接读 DeepInfer？**
可以读，`doc/deepinfer-map.md` 就是为此写的。但读和写学到的东西不一样：读代码可以跳过"为什么一定要这样"，写代码跳不过去。

**一定要用 agent 吗？**
不一定。每一节的文档本身就是完整教程，可以自己读、自己做。agent 只是让"先发现、后解决"这个过程有人陪着走。

**参考答案就在 main 上，怎么防止自己偷看？**
靠自觉。`start-step` 会把你 checkout 到 `<节>-start`，那里没有这一节的答案。

**我可以给本仓库提 PR 吗？**
欢迎，见下面。作业 PR 请提到你自己的 fork。

## 贡献

- 文档错误、行号失效、实验跑不出文档里的现象：直接开 issue 或 PR
- 想给某一节写更好的实验或测试：开 issue 说一下思路再动手
- 升级 `reference/DeepInfer` 的 pin 要单独 PR，并同步检查 `doc/` 里所有引用的行号

## 致谢

- 起点代码来自 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（MIT，作者 Xingkai Yu），原始 README 见 [doc/upstream-nano-vllm-README.md](doc/upstream-nano-vllm-README.md)
- 终点和参考实现来自 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)（Apache-2.0）及其 vendor 库 FlashMLA、DeepGEMM、DeepEP、DeepSelect
- 理论资料优先引用 [gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources)（MIT）
- README 的组织方式参考了 [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang)

## 许可

代码沿用 nano-vllm 的 MIT 许可（见 [LICENSE](LICENSE)）；`doc/` 下的文档采用 CC BY 4.0。第三方来源见 [NOTICE](NOTICE)。
