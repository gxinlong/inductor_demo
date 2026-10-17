from dataclasses import dataclass

from .simd import SIMDScheduling


@dataclass
class KernelSource:
    name: str
    args: list[str]
    outputs: list[str]
    numel: int
    block: int
    num_warps: int
    source: str


class TritonKernel:
    def __init__(self, name, node_schedule, outputs):
        self.name = name
        self.numel = node_schedule[0].node.layout.numel
        self.outputs = outputs
        self.inputs = []
        self.values = {}
        self.loads = []
        self.compute = []
        self.stores = []
        self.next_temp = 0
        self.block = 256
        self.num_warps = 4

    def new_temp(self):
        name = f"tmp{self.next_temp}"
        self.next_temp += 1
        return name

    def load(self, name, index):
        if name in self.values:
            return self.values[name]
        if name not in self.inputs:
            self.inputs.append(name)
        value = self.new_temp()
        self.loads.append(f"{value} = tl.load({name} + {index}, mask=mask, other=0.0)")
        return value

    def add(self, lhs, rhs):
        value = self.new_temp()
        self.compute.append(f"{value} = {lhs} + {rhs}")
        return value

    def mul(self, lhs, rhs):
        value = self.new_temp()
        self.compute.append(f"{value} = {lhs} * {rhs}")
        return value

    def store(self, name, index, value):
        self.values[name] = value
        if name in self.outputs:
            self.stores.append(f"tl.store({name} + {index}, {value}, mask=mask)")

    def codegen_kernel(self):
        args = self.inputs + self.outputs
        body = [
            f"xindex = tl.program_id(0) * {self.block} + tl.arange(0, {self.block})",
            f"mask = xindex < {self.numel}",
            *self.loads, *self.compute, *self.stores,
        ]
        source = f"@triton.jit\ndef {self.name}({', '.join(args)}):\n"
        source += "\n".join(f"    {line}" for line in body) + "\n"
        return KernelSource(self.name, args, self.outputs, self.numel, self.block, self.num_warps, source)


class TritonScheduling(SIMDScheduling):
    kernel_type = TritonKernel
