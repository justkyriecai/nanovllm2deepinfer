# 一起学习的小伙伴

在这里登记你的 fork 和进度。开坑加一行、进度改一行，直接提 PR 到 `main`，这是本仓库最欢迎的 PR。

| 谁 | 进度分支 | 做到哪了 | 硬件 | 开坑 | 一句话 |
|---|---|---|---|---|---|
| [kyrie](https://github.com/justkyriecai) | [`kyrie-main`](https://github.com/justkyriecai/nanovllm2deepinfer/tree/kyrie-main) | 2.2 做完（tag `2.2-done`） | Modal L4 | 2026-10 | 2.2：prefill 攒起来隔 70 步集中做，短请求 TBT p99 145 → 61 ms |

## 怎么登记

1. fork 本仓库，从 `main` 切出 `<你>-main`，push 到你的 fork（见 [README 的快速开始](README.md#快速开始)）。
2. 在上表加一行：名字链到你的 GitHub，进度分支链到你 fork 里的 `<你>-main`。
3. 提 PR 到本仓库的 `main`，标题就写 `learners: add <你>`。

进度变了（做完一章、换了硬件、暂停了）就改自己那一行再提一个 PR，标题 `learners: <你> at 2.5` 之类。只改自己的行。

## 约定

- 每个人的实现只在自己 fork 的 `<你>-main` 上，不往本仓库的 `main` 提。`main` 永远是没有实现的原始版本。
- 想看别人的做法，就 clone 表里的分支。走在前面的人的 `<节>-done` tag 可以当参考答案，`compare-step` 会用到。
- 做的过程中发现文档有错、行号失效、某一节按文档做不出来，开 issue 或直接 PR 到 `main`，所有人受益。
