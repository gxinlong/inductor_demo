from ..ir import ComputedBuffer, InputBuffer


def render_output(tree):
    if isinstance(tree, (InputBuffer, ComputedBuffer)):
        return tree.name
    items = ", ".join(render_output(item) for item in tree)
    if isinstance(tree, list):
        return f"[{items}]"
    return f"({items}{',' if len(tree) == 1 else ''})"


def generate_wrapper(graph, kernels, plan):
    lines = [
        "from pathlib import Path",
        "import torch",
        "from inductor_demo.runtime import check_inputs, load_kernel",
        "",
        "_ROOT = Path(__file__).resolve().parent",
        "_kernels = {}",
        "",
        "def call(*args):",
        f"    check_inputs(args, {[list(buffer.layout.shape) for buffer in graph.inputs]!r})",
    ]
    for i, buffer in enumerate(graph.inputs):
        lines.append(f"    {buffer.name} = args[{i}]")
    lines.extend([
        "    device = args[0].device",
        "    with torch.cuda.device(device):",
    ])
    for i, kernel in enumerate(kernels):
        lines.extend([
            f"        key_{i} = (device.index, {kernel.name!r})",
            f"        if key_{i} not in _kernels:",
            f"            _kernels[key_{i}] = load_kernel(_ROOT, {kernel.name!r})",
            f"        kernel_{i} = _kernels[key_{i}]",
        ])
    if not plan.slots and not kernels:
        lines.append("        pass")
    for slot, spec in enumerate(plan.slots):
        lines.append(f"        slot{slot} = torch.empty({spec.shape!r}, device=device, dtype=torch.float32)")
    for name, slot in plan.assignments.items():
        lines.append(f"        {name} = slot{slot}")
    for i, kernel in enumerate(kernels):
        if kernel.numel:
            grid = (kernel.numel + kernel.block - 1) // kernel.block
            lines.append(f"        kernel_{i}[({grid}, 1, 1)]({', '.join(kernel.args)})")
    lines.append(f"    return {render_output(graph.outputs)}")
    return "\n".join(lines) + "\n"
