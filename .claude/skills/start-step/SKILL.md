---
name: start-step
description: 开始 nanovllm2deepinfer 路线中的某一节（如 1.1、2.3、4.5）：准备分支，先用实验把问题在当前代码上复现出来，引导学习者分析到本质，再讨论解法、标出 TODO 的位置。学习者说"开始 2.2""做下一步""我要学 ZOS 那一步"时使用。
---

# start-step

参数：节号（如 `2.3`），例如 `2.2`。没给编号时，读学习者的进度分支 `<名字>-main` 上已经有哪些 `tests/test_X_Y.py`，推断下一步，并向学习者确认。

## 1. 准备环境

学习者的进度分支是 `<名字>-main`（维护者的是 `kyrie-main`）。这一节从它上面切出来：

```bash
git status --short                       # 有未提交的改动就先停下，问学习者怎么处理
git checkout <名字>-main
git fetch upstream && git merge upstream/main     # 文档和 skill 始终用最新版；main 上没有实现，不会冲突
git checkout -b learn/<节>
git submodule update --init reference/DeepInfer
```

- 还没有 `<名字>-main` 的（第一次用）：`git checkout -b <名字>-main main && git push -u origin <名字>-main`，再切 `learn/<节>`。
- 上一节还没做完就想跳到这一节：`git fetch upstream --tags && git checkout <上一节>-done -- nanovllm tests`，把维护者的实现拿过来当起点，并说明这一点。tag 不存在就只能从当前进度往下做。
- 如果 `reference/DeepInfer` 是空的或没有 `upstream`，运行 `bash scripts/setup_reference.sh`。

## 2. 读材料

- `doc/<节>-*.md`：这一步的文档。如果还没写，就用 `doc/README.md` 路线图里这一步的描述，再到 `reference/DeepInfer` 里找到对应代码，现场备课（结构参照 `doc/_template.md`）。
- 上一步的卡片（`doc/cards/`），用来做衔接。
- `nanovllm/` 里要改的文件，先自己读一遍，确认行号。

## 3. 发现问题（这一步最重要，不要赶）

学习者是新手，**要给方向，但结论让他自己说**。

1. 用一两句话交代背景：上一步解决了什么，现在的引擎处在什么状态。
2. 告诉他这一步要关注哪一类问题（比如"看看长 prompt 进来时 decode 怎么样"），然后给出文档第 1 节的实验命令（负载场景加指标）。如果文档还没写，就按路线图里"现象"一列现场设计一个实验。
3. **先请学习者预测**：跑之前，他觉得会看到什么？
4. 学习者跑完后，问他：哪个数字不对劲？和你的预测差在哪？
5. 学习者没看出来时，逐步缩小范围：先提示看哪个指标，再提示看哪段时间，最后提示看哪一个请求。

问题没有被复现出来之前，不进入下一节。

## 4. 分析问题

从现象一层层追问到本质，每次只问一个问题，让学习者自己回答：

1. 这个现象是哪段代码造成的？带他去读 `nanovllm/` 里的对应位置，粒度按 `.claude/skills/learn-codebase/SKILL.md`。
2. 根本原因是什么？卡在哪种资源上？
3. **这时候才引出理论**：这个现象在哪篇论文里讨论过（优先从 `reference/gpu-perf-engineering-resources/` 里找），用三到五句话说清楚。
4. 量化：用一个简单算式估一下，这个问题吃掉了多少吞吐或延迟，和实测数字对一对。

## 5. 看 DeepInfer 怎么解决，布置任务

1. 先请学习者想一想可以怎么解决，然后带他读 DeepInfer 的做法（`reference/DeepInfer/...`），讲清楚它为什么这样选。**实现方向以 DeepInfer 为准**，其他方案只用来做对比。
2. 在 `nanovllm/` 里要改的位置加上 TODO 注释和函数签名，函数体留空或写 `raise NotImplementedError`。**不写实现。**
3. 如果 `tests/test_<节>.py`（点换下划线） 还不存在，就把第 3 节的实验写成测试或 bench 断言，并确认它在当前代码上是失败的。
4. 列出任务清单，并告诉学习者：写完之后用 `check-step <节>` 验收。

## 6. 实现过程中

遵守 `AGENTS.md` 里的导师模式：提示分层给（方向 → 位置 → 伪代码 → 答案），只有学习者明确要求时才给出实现。
