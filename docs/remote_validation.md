# redpredict2 模型验证记录

2026-10-06：**真实 FX 导出、kernel/cubin 生成、带权重 forward 已通过。**

## 模型与编译路径

模型为 `examples/validate_model.py` 中的 `PointwiseResidual`：

```python
hidden = x * scale + bias
output = hidden * gate + residual
```

`scale/bias/gate` 是三个 `nn.Parameter`，从保存的 `weights.pt` 重载。为符合 demo 的能力边界，它们与输入形状相同，不使用广播；模型只有 2 个 mul 和 2 个 add。

`torch.export.export()` 得到 ATen FX graph，参数被提升为 placeholder：

```text
p_scale, p_bias, p_gate, x, residual
  -> aten.mul.Tensor
  -> aten.add.Tensor
  -> aten.mul.Tensor
  -> aten.add.Tensor
  -> output
```

测试根据 `graph_signature.input_specs` 保存参数绑定顺序，不假设权重输入顺序。编译前也检查导出的 GraphModule 与原始模型数值一致。

`KernelForward` 从 `manifest.json`、`schedule.json` 读取参数顺序、物理存储规划和 kernel 列表，用 `runtime.load_kernel()` 加载 cubin，直接绑定权重/输入、分配 slot、依次 launch。**模型测试没有导入生成的 wrapper.py，也没有打包 pt2。**

## 环境

| 项目 | 实测值 |
| --- | --- |
| SSH | `ssh redpredict2` |
| GPU | 物理 GPU 0，NVIDIA L20，SM89 |
| Python | `/opt/miniconda3/bin/python`，3.12.9 |
| PyTorch | `2.7.0+cu126` |
| Triton | `3.3.0` |
| torch 实际导入路径 | `/opt/miniconda3/lib/python3.12/site-packages/torch/__init__.py` |
| demo | `/root/gaoxinlong/Desktop/repo/inductor_demo` |
| PyTorch 参考 checkout | `/root/gaoxinlong/Desktop/repo/pytorch` |
| checkout commit | `2b3ec34829036a65cd9d1398ea72a0167dc37470` |

未安装、升级或修改依赖，未构建或修改 PyTorch。实际执行使用已有的 PyTorch 2.7 wheel，不是把参考 checkout 编译后运行。测试前两次 GPU 0 均为 0 MiB、0% 利用率；测试结束后回到 0 MiB。

## 结果

| shape | 融合 kernel / storage slots | 不融合 kernel / storage slots | 首次 forward 对比 | 新进程对比 |
| --- | --- | --- | --- | --- |
| `(1,)` | 1 / 1 | 4 / 2 | 14 通过 | 14 通过 |
| `(255,)` | 1 / 1 | 4 / 2 | 14 通过 | 14 通过 |
| `(256,)` | 1 / 1 | 4 / 2 | 14 通过 | 14 通过 |
| `(1025,)` | 1 / 1 | 4 / 2 | 14 通过 | 14 通过 |
| `(17, 257)` | 1 / 1 | 4 / 2 | 14 通过 | 14 通过 |

- 总计 **140 次 forward 对比通过**，全部使用 `rtol=0, atol=0`；最大绝对误差为 **0.0**，不相等元素数为 **0**。
- 每轮包含保存的示例输入、4 个随机种子、全零输入、构造的抵消输入；融合/不融合分别比较。
- 首轮随机种子为 `0, 1, 17, 42`；新进程为 `3, 5, 7, 1234`。两轮使用相同权重工件，另有部分相同的边界输入。
- 成功导出 **25 个 cubin**，检查 ELF header 并记录每个文件的 SHA-256。关闭 `enable_fp_fusion`，保持 FP32 两次舍入的乘加语义。
- 新进程使用全新的 `TRITON_CACHE_DIR`，并 mock `triton.compile` 为报错。在这个条件下仍通过所有对比；新 cache 有 3 个宿主 `.so`、0 个 `.cubin`，因此 GPU kernel 来自保存的 cubin。
- 远端 12 项 `test_core.py` 测试通过。原有 `a+b*c` 冒烟测试也实际生成 cubin 并通过 eager 精度比较。

这些是 FP32、连续静态形状、add/mul 的正确性结果，未测性能，没有推导吞吐或延迟收益。完整 `tests/test_fx_cuda.py` 未执行，环境缺少 `expecttest`；本次使用可独立执行的模型验证脚本。未运行 compute-sanitizer，远端 shell 的 PATH 中没有找到该工具。

## 文件位置

远端成功运行的目录：

```text
/root/gaoxinlong/Desktop/repo/inductor_demo/artifacts/model_20261006_02/
```

同一批工件已同步到本地 `artifacts/model_20261006_02/`，不包含临时的 Triton replay cache。

```text
environment.json
cases.json
forward_report.json
fresh_process_report.json
shape_17x257/
  weights.pt
  example_inputs.pt
  bindings.json
  fx_graph.txt
  fx_graph.py
  fused/
    kernels.py
    schedule.json
    manifest.json
    triton_pw_0/triton_pw_0.cubin
  unfused/
    kernels.py
    schedule.json
    manifest.json
    triton_pw_0/... 至 triton_pw_3/...
```

其他 shape 使用相同结构。`fx_graph.py` 保存 FX 的 forward 源码供阅读，实际编译消费 `torch.export` 返回的 GraphModule 对象。

## 实际执行命令

首次编译与验证：

```bash
cd /root/gaoxinlong/Desktop/repo/inductor_demo
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /opt/miniconda3/bin/python \
  examples/validate_model.py --output artifacts/model_20261006_02
```

独立进程、新 cache、不同随机种子重载：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
TRITON_CACHE_DIR=/root/gaoxinlong/Desktop/repo/inductor_demo/artifacts/model_20261006_02/replay_cache \
/opt/miniconda3/bin/python examples/validate_model.py \
  --replay artifacts/model_20261006_02 \
  --seeds 3 5 7 1234 --report-name fresh_process_report.json
```

这些目录/报告已经存在；再次执行时请更换 `--output`，或更换 replay cache 和 `--report-name`。脚本拒绝覆盖已有报告。

## 首次尝试遇到的问题

`artifacts/model_20261006_01` 是未完成的首次尝试。给 `torch.export` 的 GraphModule 额外加了 `torch.save/torch.load` 往返，加载时触发 PyTorch 2.7 `_ModuleStackTracer.__init__()` 缺少 `scope_root`。这个失败发生在 demo 编译之前。

最终脚本直接把真实导出的 GraphModule 传给 demo，保留可读 FX 文件，并让另一个进程只重载权重与 cubin。这遵循本次关注 kernel 生成/执行而不关注模型打包的范围；没有为此修改 demo 编译器或 PyTorch。
