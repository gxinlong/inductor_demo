from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .codegen.wrapper import generate_wrapper
from .ir import TensorSpec
from .memory import plan_memory
from .runtime import compile_cubins, load_python_module
from .scheduler import Scheduler


@dataclass
class CompileResult:
    directory: Path
    scheduler: Scheduler
    memory_plan: object
    kernels: list

    @property
    def source_path(self):
        return self.directory / "kernels.py"

    @property
    def wrapper_path(self):
        return self.directory / "wrapper.py"

    @property
    def cubin_paths(self):
        manifest = json.loads((self.directory / "manifest.json").read_text())
        return [self.directory / entry["cubin"] for entry in manifest["kernels"] if entry["cubin"]]

    def load(self):
        return load_python_module(self.wrapper_path).call


def compile_fx(graph, example_inputs, output_dir, *, fusion=True, max_fused_nodes=32, emit_cubin=True, sm=None):
    import torch
    from .graph import GraphLowering

    if isinstance(graph, torch.fx.Graph):
        graph = torch.fx.GraphModule(torch.nn.Module(), graph)
    if not isinstance(graph, torch.fx.GraphModule):
        raise TypeError("input must be an FX Graph or GraphModule")
    graph.graph.lint()
    if sum(node.op == "placeholder" for node in graph.graph.nodes) != len(example_inputs):
        raise ValueError("provide one example tensor or TensorSpec per placeholder")
    specs = [value if isinstance(value, TensorSpec) else TensorSpec.from_tensor(value) for value in example_inputs]
    lowered = GraphLowering(graph).run(*specs, enable_io_processing=False)
    result = compile_ir(
        lowered, output_dir, fusion=fusion, max_fused_nodes=max_fused_nodes,
        emit_cubin=False,
    )
    (result.directory / "fx_graph.txt").write_text(str(graph.graph) + "\n")
    if emit_cubin:
        compile_cubins(result.directory, sm=sm)
    return result


def compile_ir(graph, output_dir, *, fusion=True, max_fused_nodes=32, emit_cubin=False, sm=None):
    if not graph.inputs:
        raise ValueError("v1 requires at least one tensor input")
    scheduler = Scheduler(graph, fusion=fusion, max_fused_nodes=max_fused_nodes)
    kernels = scheduler.codegen()
    plan = plan_memory(scheduler)
    source = "import triton\nimport triton.language as tl\n\n\n" + "\n\n".join(kernel.source for kernel in kernels)
    wrapper = generate_wrapper(graph, kernels, plan)
    ast.parse(source)
    ast.parse(wrapper)
    directory = Path(output_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"use an empty output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernels.py").write_text(source)
    (directory / "wrapper.py").write_text(wrapper)
    trace = {
        "ir": [{"name": op.name, "op": op.data.op, "ranges": op.data.ranges, "origin": op.origin} for op in graph.operations],
        "before_fusion": scheduler.before_fusion,
        "fusion_log": scheduler.fusion_log,
        "after_fusion": scheduler.describe_nodes(),
        "memory": asdict(plan),
    }
    (directory / "schedule.json").write_text(json.dumps(trace, indent=2) + "\n")
    manifest = {
        "format": 1, "compiled": False, "sm": None, "triton_version": None,
        "kernels": [{key: value for key, value in asdict(kernel).items() if key != "source"} | {"cubin": None} for kernel in kernels],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if emit_cubin:
        compile_cubins(directory, sm=sm)
    return CompileResult(directory, scheduler, plan, kernels)
