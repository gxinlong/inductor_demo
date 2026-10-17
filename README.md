# Inductor Demo

一个以 **scheduling 和 Triton 代码生成**为重点的教学编译器：

```text
FX Graph / GraphModule
  -> GraphLowering + lowerings 注册表
  -> GraphIR: InputBuffer + ComputedBuffer(Pointwise.inner_fn)
  -> SchedulerNode + ReadWrites
  -> 拓扑排序 / can_fuse / 贪心融合
  -> FusedSchedulerNode
  -> CUDACombinedScheduling.choose_node_backend
  -> TritonScheduling / SIMDScheduling
  -> node_schedule -> TritonKernel 的 loads/compute/stores
  -> kernels.py + memory plan + wrapper.py
  -> Triton ASTSource -> triton.compile -> cubin
  -> wrapper 加载 cubin，在 PyTorch 当前 CUDA stream 上运行
```

没有调用 `torch.compile`、`torch._inductor` 或把编译委托给真实 Inductor。FX 是前端，Triton 是后端；中间的 lowering、依赖、融合、代码生成、存储规划由这个项目实现。

已在 redpredict2 的 L20 上完成 [带权重模型的远端验证](docs/remote_validation.md)：5 种形状，融合/不融合及独立进程重载共 140 次 forward 比较，最大绝对误差为 0。

## 先看最小例子

在项目根目录运行，或者先在已有 Python 环境中安装本项目。使用 `PYTHONPATH=.` 就不需要安装项目本身。

```python
import torch
from inductor_demo import TensorSpec, compile_fx

gm = torch.fx.symbolic_trace(lambda a, b, c: a + b * c)
result = compile_fx(
    gm,                             # 也接受 gm.graph
    [TensorSpec((1025,))] * 3,       # 或三个 CUDA float32 example tensors
    "artifacts/example",            # 必须不存在或为空
    emit_cubin=True,
)

print(result.source_path.read_text())
print(result.cubin_paths)

inputs = [torch.randn(1025, device="cuda") for _ in range(3)]
actual = result.load()(*inputs)
torch.testing.assert_close(actual, inputs[0] + inputs[1] * inputs[2], rtol=0, atol=0)
```

这个例子有两个 `ComputedBuffer`，融合后只有一个 kernel、三个输入 load、一个输出 store，以及一个输出 storage slot。中间乘法结果作为 kernel 内的临时值传给加法，不分配显存。

生成的函数主体类似：

```python
@triton.jit
def triton_pw_0(arg1, arg2, arg0, buf1):
    xindex = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = xindex < 1025
    tmp0 = tl.load(arg1 + xindex, mask=mask, other=0.0)
    tmp1 = tl.load(arg2 + xindex, mask=mask, other=0.0)
    tmp3 = tl.load(arg0 + xindex, mask=mask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 + tmp2
    tl.store(buf1 + xindex, tmp4, mask=mask)
```

`tmp3` 在 `tmp2` 前出现，是因为最终按 `loads -> compute -> stores` 拼接。变量编号是生成顺序，不是机器指令编号。

## 环境与运行方式

| 环境 | 能验证的部分 |
| --- | --- |
| Python 3.9+，无第三方依赖 | 从手写 IR 开始的 scheduler、融合、内存规划、源码生成 |
| 安装了 PyTorch 的 Python | 从真实 FX graph 开始生成源码和 wrapper，无需 GPU |
| Linux + Triton 3.3.x | 指定 `sm`，离线编译 cubin；需要 Triton 支持该架构 |
| Linux + NVIDIA CUDA + PyTorch + Triton 3.3.x | 编译、加载 cubin、GPU 数值校验 |

CUDA 依赖组合以 **PyTorch 2.7.0 + cu128 / Triton 3.3.0** 为目标。已有其他环境时不要直接覆盖；可以使用独立环境。在选定环境内，安装命令示例为：

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[test]'
```

Linux 的上述 PyTorch wheel 会带上匹配的 Triton。cubin 加载使用 Triton 内部 `CompiledKernel` 接口，因此代码明确限制在 3.3.x，且导出与加载要求相同 patch 版本。首次启动可能需要 Triton 编译宿主 launcher，因此仍需要可用的 C 编译器；这不重新编译 GPU kernel。

### 1. 当前 Mac 上，不安装依赖也能观察 scheduler

```bash
PYTHONPATH=. python3 examples/inspect_scheduler.py --output artifacts/my_ir_fused
PYTHONPATH=. python3 examples/inspect_scheduler.py --output artifacts/my_ir_unfused --unfused
PYTHONPATH=. python3 -m unittest discover -s tests -p test_core.py -v
```

这里使用手写的等价 IR，**不经过 FX，也不编译或运行 GPU 代码**。已生成的示例在 `artifacts/ir_fused` 和 `artifacts/ir_unfused`；重复运行以上命令时请换一个新的输出目录。

### 2. 已有 PyTorch、没有 CUDA：从真实 FX 生成源码

```bash
PYTHONPATH=. python examples/pointwise.py --source-only --output artifacts/fx_source
```

此时 `manifest.json` 中 `compiled=false`，`cubin=null`。不会生成伪造或空的 cubin。wrapper 可以阅读，但运行前还需要完成编译。

### 3. Linux/CUDA：端到端编译、运行与对比

```bash
PYTHONPATH=. python examples/pointwise.py --output artifacts/cuda_fused --run
PYTHONPATH=. python examples/pointwise.py --output artifacts/cuda_unfused --unfused --run
PYTHONPATH=. python tests/test_fx_cuda.py -v
```

第二条会生成两个 kernel，显式保留中间 `buf0`。可对照两组 `kernels.py`、`wrapper.py` 和 `schedule.json`。

`test_fx_cuda.py` 使用 PyTorch 的 `TestCase`、`parametrize` 和 device-type runner：覆盖 FX target、融合/不融合、尾块 mask、空张量、标量形状、多维连续张量、返回中间值、输出别名、buffer 复用、当前 stream、输入检查与工件目录迁移。加载时临时禁止 `triton.compile`，检查没有 GPU 重编译回退。

也可以分两步编译，在有 Triton 的 Linux 上指定目标架构：

```python
from inductor_demo.runtime import compile_cubins
compile_cubins("artifacts/fx_source", sm=89)
```

`sm=89` 表示目标是 SM89，并不代表运行机器已验证支持。demo 加载时要求 GPU 架构与导出时一致。

### 4. 导出带权重模型的 FX，直接加载 cubin 做 forward

```bash
PYTHONPATH=. python examples/validate_model.py --output artifacts/my_model_run
PYTHONPATH=. python examples/validate_model.py --replay artifacts/my_model_run --report-name replay_report.json
```

模型为 `(x * scale + bias) * gate + residual`，三个权重与输入 shape 一致。`torch.export` 把权重提升为 FX placeholder；测试保存和重载权重，直接启动生成的 cubin，不依赖生成的 wrapper。脚本同时验证融合与不融合版本，并在 forward 阶段禁止 `triton.compile`。成功运行的工件和两份报告在 `artifacts/model_20261006_02/`。

## 输出是什么

```text
artifacts/example/
  fx_graph.txt                  # 只有从 FX 入口编译才有
  kernels.py                    # 可阅读的 Triton 源码
  wrapper.py                    # 输入检查、分配、kernel launch、返回
  schedule.json                 # IR、融合前后节点、融合评分、存储生命周期
  manifest.json                 # kernel 参数顺序、编译状态、SM、工件路径
  triton_pw_0/                  # 完成 cubin 编译后才有
    triton_pw_0.ttir
    triton_pw_0.ttgir
    triton_pw_0.llir
    triton_pw_0.ptx
    triton_pw_0.cubin
    triton_pw_0.json             # Triton 编译 metadata
```

中间表示的具体文件取决于 Triton 的 `binary.asm`。导出器保存其中的所有阶段。

wrapper 通过 `runtime.load_kernel()` 用源码签名、导出的 metadata 和 cubin 重建 `CompiledKernel`。**没有 JIT 回退**；缺少 cubin、版本或 SM 不一致时会报错。源码被读取来重建参数描述，不经过 `triton.compile`。这不是 PyTorch 的 `.pt2` 或 AOTI `.so` 打包格式；运行仍依赖本项目、PyTorch 和同版本 Triton。

## 按这个顺序阅读

| 顺序 | 文件 / 入口 | 关注的问题 |
| --- | --- | --- |
| 1 | `compile_fx.py: compile_fx / compile_ir` | 各阶段之间传什么对象？ |
| 2 | `graph.py`、`lowering.py`、`ir.py` | FX target 如何注册、如何变成 `inner_fn`？ |
| 3 | `dependencies.py: extract_read_writes` | 同一个计算描述如何提取访问关系？ |
| 4 | `scheduler.py: Scheduler.__init__ / can_fuse / fuse_nodes` | 什么能融合？融合后是什么？ |
| 5 | `codegen/cuda_combined_scheduling.py` | 按节点类型如何选择生成路径？ |
| 6 | `codegen/simd.py` | scheduler node 如何成为 kernel 内的 `node_schedule`？ |
| 7 | `codegen/triton.py` | load 如何被内部值替换？两阶段代码生成各做什么？ |
| 8 | `memory.py`、`codegen/wrapper.py` | 哪些 buffer 必须有显存？什么时候能复用？ |
| 9 | `runtime.py` | Triton 编译与 cubin 加载在哪里发生？ |

详细机制、断点和真实源码对应关系见 [docs/walkthrough.md](docs/walkthrough.md)。本次验证范围见 [docs/validation.md](docs/validation.md)。

## 明确的边界

- 仅静态形状、CUDA、连续 strided float32、tensor-tensor `add/mul`；add 的 `alpha` 只能为 1。
- 允许多维连续输入，以一维线性索引展开；每个二元算子的输入 shape 必须相同。
- 无广播、view、mutation、scalar operand、reduce、动态 shape、autograd、GEMM、模板 kernel、autotune。
- 保留 lowering 注册接口与按 node kind 的 backend 分发点；扩展算子还要补充依赖 handler 和代码生成 handler。
- 不实现 TensorBox/StorageBox 和延迟内联。每个 FX 算子直接生成一个 `ComputedBuffer`，这是为了暴露 scheduler 融合过程。真实 Inductor 可能在 lowering 时已经把这个表达式内联。
- demo 自己不做 CSE、DCE 或常量折叠；底层 Triton/LLVM 仍正常做编译优化。
- 融合不查 `add+mul` pattern 表。依靠访问关系、同计算域以及收缩后的依赖图无环性，按估算 IO 收益贪心选择；没有 benchmark。
- `node_schedule` 只是 kernel 内的计算节点顺序。SASS 级的指令选择、调度和寄存器分配属于 Triton/LLVM/ptxas。
- 固定 BLOCK=256、num_warps=4，编译时关闭 `enable_fp_fusion`，避免把乘加折叠为 FMA 导致不同的舍入结果。未做性能承诺。
