# 验证记录

后续已在 redpredict2 完成真实 FX、cubin 和带权重模型的 forward 验证：5 种形状，融合/不融合共 140 次对比，最大绝对误差为 0。完整环境、工件和命令见 [remote_validation.md](remote_validation.md)。以下保留初版交付时的本地验证记录。

2026-10-06，开发机器为 macOS，`/usr/bin/python3` 版本 3.9.6。

该解释器没有安装 PyTorch 或 Triton，项目和父目录也没有 `.venv`。本次没有安装或修改任何 PyTorch、CUDA、Triton 依赖，没有构建 PyTorch，也没有运行远程 GPU 任务。

## 已执行

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -p test_core.py -v
```

12 个 core tests 通过，覆盖：

- 两个节点融合为一个 pointwise kernel，中间 load/store 和分配消除。
- 关闭融合时保留两个 kernel 和中间 buffer。
- 收缩造成依赖环时拒绝融合。
- 返回的中间值、组外 consumer 需要保留 store。
- 最后一次读取之后才复用存储，不复用仍需返回的结果。
- demo 不做 CSE/DCE；未知依赖、非 identity indexing 被拒绝。
- 拓扑顺序、handler 恢复、空形状以及输出结构。
- 20 个随机 add/mul DAG：分别启用/关闭融合，用标量宿主解释执行生成的函数体和物理 slot 复用，与原 IR 的计算结果比较。

生成源码和 wrapper 使用 Python AST 进行语法检查。core test 中的标量 handler 和生成源码的宿主解释只验证调度/代码生成数据流，不代表 Triton kernel 已经执行，也不证明 GPU 浮点数值行为。

## 初版本地未执行的项目

- 真实 PyTorch FX interpreter/lowering 的集成测试。
- Triton 3.3.x 的 ASTSource 编译与 CompiledKernel 文件加载。
- cubin 内容、GPU 数值、stream 行为与目录迁移的真实运行校验。

这些测试写在 `tests/test_fx_cuda.py`。后续远端通过独立的 `examples/validate_model.py` 覆盖了本次所需的 FX、编译、权重和 cubin 重载，以及 forward 精度；完整测试文件仍未执行，未将它计为通过。

初版的 `artifacts/ir_fused` / `artifacts/ir_unfused` 是本机从手写 IR 生成的教学源码。当时没有生成 cubin。后续远端实际生成并验证的 cubin 保存在 `artifacts/model_20261006_02/`。
