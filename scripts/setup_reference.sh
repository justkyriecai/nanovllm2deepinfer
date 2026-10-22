#!/usr/bin/env bash
# 准备 reference/ 下的参考代码（DeepInfer 及其 vendor 子模块）和资料清单，并拉取各步骤的 tag。
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

UPSTREAM_URL="https://github.com/justkyriecai/vllm2deepinfer.git"
if ! git remote get-url upstream >/dev/null 2>&1 && [ "$(git remote get-url origin)" != "$UPSTREAM_URL" ]; then
  git remote add upstream "$UPSTREAM_URL"
fi
git fetch --tags "$(git remote get-url upstream >/dev/null 2>&1 && echo upstream || echo origin)"

git submodule update --init --depth 1 reference/DeepInfer reference/gpu-perf-engineering-resources
# vendor 下的 kernel 库（FlashMLA、DeepGEMM、DeepEP、DeepSelect 等）只拉第一层，不递归进它们自己的子模块
git -C reference/DeepInfer submodule update --init --depth 1

git -C reference/DeepInfer log --oneline -1
git -C reference/DeepInfer submodule status
