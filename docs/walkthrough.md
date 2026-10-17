# 从 a + b * c 跟到 cubin

## 1. Lowering 创建计算描述

`lowering.lowerings` 的 key 是 FX node 的 `target`，不是 FX node 的名字。默认注册：

```text
operator.add / torch.add / aten.add.Tensor -> add
operator.mul / torch.mul / aten.mul.Tensor -> mul
```

`GraphLowering` 是 FX `Interpreter` 的子类。`call_function` 不执行原始 torch 算子，而是调用注册的 lowering。传入的是 `InputBuffer/ComputedBuffer`，返回的是 `ComputedBuffer`。

对例子，IR 为：

```text
arg0, arg1, arg2: InputBuffer
buf0: ComputedBuffer(Pointwise(inner_fn = load(arg1, i) * load(arg2, i)))
buf1: ComputedBuffer(Pointwise(inner_fn = load(arg0, i) + load(buf0, i)))
outputs = buf1
```

这里 `inner_fn(i)` 描述一个位置的计算；它不包含 Python 循环，也没有启动 CUDA kernel。`ranges` 保存原始形状，codegen 用连续布局把它展平成一维索引。

新增 lowering 时，可以使用 `@register_lowering(target)`。注册一个入口不意味着所有后续阶段已支持它；还需要明确该算子的 IR、读写模式和 codegen。

## 2. 同一个 inner_fn，两个 handler

`virtualized.V` 用 `ContextVar` 保存当前 ops handler。它只保留 `V.ops` 这一项，没有真实 Inductor 的其他全局状态。

```python
with V.set_ops_handler(DependencyHandler()):
    buffer.codegen("i")

with V.set_ops_handler(triton_kernel):
    scheduler_node.codegen("xindex")
```

第一次执行 `inner_fn` 时：

- `load("arg1", "i")` 记录 `MemoryDep("arg1", "i")`。
- `mul()` 不做乘法，返回值也不用于数值计算。
- `store("buf0", "i", value)` 记录写 `buf0`。

第二次执行时：

- `load()` 返回诸如 `"tmp0"` 的表达式变量名，并追加 `tl.load` 源码。
- `mul()` 追加 `tmp2 = tmp0 * tmp1`，返回 `"tmp2"`。
- `store()` 把逻辑 buffer 绑定到该临时值；只有需要物化时才追加 `tl.store`。

这就是 define-by-run 计算描述：Python 函数本身不变，handler 决定一次调用的含义。demo 未实现真实 Inductor 的 `LoopBody` FX 捕获和 CSEProxy。

## 3. Scheduler 初始化

`Scheduler.__init__` 按以下顺序工作：

1. `create_scheduler_node`：包装每个 `ComputedBuffer`，执行依赖 handler，获得 `ReadWrites`。
2. `compute_dependencies`：检查 producer/input 是否存在，建立 buffer -> consumer 集合。
3. `topological_sort`：稳定的 Kahn 拓扑排序。
4. `fuse_nodes`：寻找并合并融合候选。

读写索引只允许同一个 `i`，不支持修改输入或跨索引访问。因此无需引入 StarDep/WeakDep、别名分析或复杂 loop 重排。要增加 view/reduction，必须扩展这个合法性边界。

## 4. can_fuse 到底判断什么

demo 的条件为：

- 合并后成员数量不超过 `max_fused_nodes`。
- 两组共享至少一个读写 buffer，可包含 producer-consumer 或共享输入的 sibling。
- backend 支持这两组节点；当前只有 pointwise。
- 所有成员的 shape/dtype/device 一致，索引已限定为一一对应。
- 把两组收缩成一个节点之后，图仍然无环。

最后一个条件不能省略。例如 `A -> B -> C`，同时还有 `A -> C`。直接融合 A/C 会使新节点既需要在 B 之前又需要在 B 之后；应拒绝这一个 pair，但 A/B/C 最终整体融合仍可能合法。

`get_possible_fusions` 为了教学直接枚举所有 pair；真实 Inductor 会按共享 buffer 等信息缩小候选集合。评分优先估算消除 producer-consumer IO 的收益，再考虑共享输入。估算不扣除外部消费者仍需的 store，也不运行性能测量；它只是排序启发式。

每次取最高分 pair，生成 `FusedSchedulerNode`，重排后重新搜候选。成员仍然是原来的 `SchedulerNode`，原 IR 没被替换为一个新的 `Pointwise` 大节点。

## 5. node_schedule 与代码生成的两步

```text
Scheduler.nodes
  [FusedSchedulerNode]
        |
        +-- members = [SchedulerNode(buf0), SchedulerNode(buf1)]
                           |
                           v
                  generate_node_schedule
                           |
                           v
                    [buf0 node, buf1 node]
```

这里只做 pointwise，因此 `node_schedule` 是成员的拓扑有序列表，没有 reduction region marker。它组织单个 kernel 内的计算顺序。

`SIMDScheduling.codegen_node_schedule_with_kernel()` 用 Triton handler 执行这些成员，填入：

```text
kernel.loads
kernel.compute
kernel.stores
kernel.values      # 已经在这个 kernel 中计算出来的逻辑 buffer
```

执行 buf0 后，`kernel.values["buf0"] = "tmp2"`。buf1 再读取 buf0 时直接得到 `tmp2`，不发射显存 load。

`TritonKernel.codegen_kernel()` 再组合函数签名、`program_id/arange/mask` 和三个代码区，返回完整的源码。它没有调用 Triton 编译器。

`values` 是 producer-consumer 值传递，不是表达式 CSE 表。两个独立但相同的乘法仍会生成两条乘法；重复输入 load 也不做消重。底层编译器可能随后优化这些表达式。

## 6. 哪些 buffer 有显存

`Scheduler.materialized_outputs(group)` 保留三类写入：

1. 图的返回值。
2. 组外仍有 consumer 的值。
3. 没有 consumer 的值，也保留：demo 不做 DCE。

只有全部消费者都在组内、且不是图输出的值才能完全使用局部临时值。

`memory.plan_memory()` 计算每个需要显存的 buffer 的最后使用 kernel，然后以相同 `TensorSpec` 为条件复用 slot：

```text
不融合的三节点链：
K0: write buf0 -> slot0
K1: read buf0, write buf1 -> slot1; K1 后释放 slot0
K2: read buf1, write buf2 -> 复用 slot0
```

不能在 K1 启动之前把 slot0 同时分给 buf1，因为 K1 还需要读 buf0。图输出的生命周期延长到所有 kernel 之后；输入始终由调用者持有，从不参与复用。

wrapper 开头为物理 slots 调用 `torch.empty`，再建立逻辑 buffer 到 slot 的别名。它在整个调用期间持有 slots，结束后临时对象交回 PyTorch caching allocator。这里的 `release` 是规划器认为一个 slot 可复用，不是立刻执行 CUDA free，更不是分配固定设备地址。

多个 kernel 顺序提交到同一个当前 stream。正是这个执行顺序保证 slot 的复用安全；demo 不做跨 stream 并行。

## 7. 从源码到 cubin，以及 wrapper

`compile_cubins()`：

1. 导入生成的 `kernels.py`，得到 Triton JITFunction 描述。
2. 为每个函数构造 `ASTSource`，参数只有 fp32 tensor pointers。shape、BLOCK 固定在源码中。
3. 构造 `GPUTarget("cuda", sm, 32)`。
4. `triton.compile()`，设置 4 warps、1 CTA、禁用 FP fusion。
5. 写出 `binary.asm` 中的各级 IR、PTX、cubin 和 metadata。

`runtime.load_kernel()` 从这些文件重建 `CompiledKernel`，其第一次调用会让 Triton 初始化 CUDA module/function 和宿主 launcher。wrapper 使用这个已编译对象启动 kernel，没有 `JITFunction[grid]` 或 `triton.compile` 回退。

SASS 级指令调度发生在下游编译器中。demo 的排序与 `loads/compute/stores` 拼接只是源代码组织，不能等同于最终机器指令顺序。

## 8. 与当前 PyTorch 源码对应

参考 checkout：`/Users/gaoxinlong/workspace/github/pytorch`，commit `2b3ec34829036a65cd9d1398ea72a0167dc37470`。对应的是职责，不是逐行复制。

| Demo | PyTorch |
| --- | --- |
| `compile_fx.py` / `GraphLowering` | `torch/_inductor/compile_fx.py` / `graph.py` |
| `lowerings` / `register_lowering` | `torch/_inductor/lowering.py` |
| `Pointwise` / `ComputedBuffer` | `torch/_inductor/ir.py` |
| `V.set_ops_handler` | `torch/_inductor/virtualized.py` |
| `MemoryDep` / `extract_read_writes` | `torch/_inductor/dependencies.py` |
| `SchedulerNode` / `FusedSchedulerNode` / `can_fuse` | `torch/_inductor/scheduler.py` |
| `CUDACombinedScheduling` | `torch/_inductor/codegen/cuda_combined_scheduling.py` |
| `generate_node_schedule` / `codegen_node_schedule_with_kernel` | `torch/_inductor/codegen/simd.py` |
| `TritonKernel` / `codegen_kernel` | `torch/_inductor/codegen/triton.py` |
| `plan_memory` / wrapper | `torch/_inductor/codegen/wrapper.py` / `memory_planning.py` 等 |
| `compile_cubins` | `torch/_inductor/runtime/triton_heuristics.py` 中 ASTSource/compile 路径 |

两个最重要的教学简化：一是每个 FX 算子立即进入 `ComputedBuffer`；二是普通 pointwise 用一个显式 `V.ops` handler 直接生成字符串。真实实现还有 TensorBox/StorageBox、LoopBody、CSEProxy、多级配置和多种 IR。

## 9. VS Code / PyCharm 断点

建议给 Python 解释器配置好对应环境，然后调试 `examples/pointwise.py`：

```text
--source-only --output artifacts/debug_session_1
```

| 断点 | Watch / Evaluate |
| --- | --- |
| `GraphLowering.call_function` | `target`、`args`、`self.ir.operations` |
| `Scheduler.compute_dependencies` 之后 | `self.users`、`self.nodes[0].read_writes` |
| `Scheduler.can_fuse` | `first.reads`、`first.writes`、`second.reads` |
| `Scheduler.fuse_nodes` 之后 | `self.nodes[0].get_nodes()` |
| `SIMDScheduling.codegen_node_schedule_with_kernel` | `node_schedule`、`kernel.values` |
| `TritonKernel.codegen_kernel` 入口 | `self.loads`、`self.compute`、`self.stores` |
| `compile_cubins` 的 compile 返回后 | `binary.asm.keys()`、`binary.metadata` |

每次选择新的输出目录，避免把一次旧编译的 cubin 误当成新源码的结果。
