---
name: compare-step
description: 把学习者在 nanovllm2deepinfer 某一节的实现和参考实现（<节>-done tag）以及 DeepInfer 源码做对照，解释每一处差异背后的 why，最后收这一步的卡片。学习者说"对比一下答案""和 DeepInfer 比比""这一步收个尾"时使用。
---

# compare-step

参数：节号（如 `2.3`）。

## 1. 和参考实现对比

参考实现是维护者 `kyrie-main` 上的 `<节>-done` tag，在上游：

```bash
git fetch upstream --tags
git diff <节>-done -- nanovllm/       # 学习者的实现 vs 参考实现
```

维护者还没做到这一节（tag 不存在）时，看 `LEARNERS.md` 里有没有走在前面的人，拿他们 fork 里的分支来比；都没有就跳过这一步，只和 DeepInfer 比。

把差异分成三类，一类一类讲：

| 类型 | 怎么处理 |
|---|---|
| 等价写法 | 一句话带过 |
| 学习者漏掉的边界情况 | 构造一个能触发它的输入，让学习者自己判断会出什么错 |
| 不同的设计取舍 | 说清楚两种做法各自的代价，没有绝对的对错 |

## 2. 和 DeepInfer 对比

- 找到 `reference/DeepInfer` 里对应的代码（文档"DeepInfer 对照"一节有位置），按 `.claude/skills/learn-codebase/SKILL.md` 的粒度讲解。
- 重点讲 DeepInfer **多做了什么、少做了什么，以及为什么**。比如：它为什么要 DP 投票，我们单卡为什么不需要；它为什么用 C++ 写调度器。
- 有些差异是因为这一步还没做到（后面的章节会补上），这种要标出来是哪一步。

## 3. 收卡片

格式按 `AGENTS.md`：要记住的 why（3–6 条，每条写"不这样做会怎样"和"代价"）、调用链（我们的实现和 DeepInfer 各一条）、可以忘的细节、2–3 道自测题。

卡片全文发在对话里，同时存到 `doc/cards/<节>-<主题>.md`。
