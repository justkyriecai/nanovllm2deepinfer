# 测试

每一节的验收测试放在 `test_<节>.py`，节号里的点换成下划线，例如第 2 章第 2 节是 `test_2_2.py`。

- 这一节开始时（`<节>-start`），它应该是失败的；完成后（`<节>-done`）应该通过。
- 前面各节的测试要一直保持通过，用来防止回归。
- 需要 GPU 的测试用 `pytest.mark.skipif(not torch.cuda.is_available(), ...)` 标出来，没有 GPU 的机器上也能跑其余部分。
