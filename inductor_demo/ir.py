from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .virtualized import V


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str = "float32"
    device: str = "cuda"

    def __post_init__(self):
        if not isinstance(self.shape, tuple) or any(type(n) is not int or n < 0 for n in self.shape):
            raise ValueError("shape must be a tuple of nonnegative, static integers")
        if self.dtype != "float32":
            raise NotImplementedError("v1 supports float32 only")
        if self.device != "cuda":
            raise NotImplementedError("v1 supports CUDA only; device index is chosen at runtime")
        if self.numel >= 2**31:
            raise NotImplementedError("v1 uses 32-bit flat indexing")

    @property
    def numel(self):
        return math.prod(self.shape)

    @classmethod
    def from_tensor(cls, tensor):
        if tensor.device.type != "cuda" or tensor.dtype.__str__() != "torch.float32":
            raise ValueError("example tensors must be CUDA float32; use TensorSpec for source-only compilation")
        if not tensor.is_contiguous() or tensor.layout.__str__() != "torch.strided":
            raise ValueError("v1 requires contiguous, strided tensors")
        if tensor.requires_grad:
            raise ValueError("v1 is inference-only")
        return cls(tuple(tensor.shape))


@dataclass
class InputBuffer:
    name: str
    layout: TensorSpec


@dataclass
class Pointwise:
    ranges: tuple[int, ...]
    inner_fn: Callable
    op: str


@dataclass
class ComputedBuffer:
    name: str
    layout: TensorSpec
    data: Pointwise
    origin: str
    kind: str = field(default="pointwise", init=False)

    def codegen(self, index):
        value = self.data.inner_fn(index)
        V.ops.store(self.name, index, value)


def output_buffers(tree):
    if isinstance(tree, (InputBuffer, ComputedBuffer)):
        return [tree]
    if isinstance(tree, (tuple, list)):
        return [buffer for item in tree for buffer in output_buffers(item)]
    raise NotImplementedError("outputs must be tensors, or nested tuples/lists of tensors")


@dataclass
class GraphIR:
    inputs: list[InputBuffer] = field(default_factory=list)
    operations: list[ComputedBuffer] = field(default_factory=list)
    outputs: Any = ()

    def register_pointwise(self, op, lhs, rhs, origin):
        if not isinstance(lhs, (InputBuffer, ComputedBuffer)) or not isinstance(rhs, (InputBuffer, ComputedBuffer)):
            raise NotImplementedError("v1 supports tensor-tensor operations only")
        if lhs.layout != rhs.layout:
            raise NotImplementedError("v1 requires equal shapes/dtypes/devices; no broadcasting")

        def inner_fn(index):
            left = V.ops.load(lhs.name, index)
            right = V.ops.load(rhs.name, index)
            return getattr(V.ops, op)(left, right)

        buffer = ComputedBuffer(
            f"buf{len(self.operations)}", lhs.layout,
            Pointwise(lhs.layout.shape, inner_fn, op), origin,
        )
        self.operations.append(buffer)
        return buffer

    @property
    def buffers(self):
        return {buffer.name: buffer for buffer in [*self.inputs, *self.operations]}

    @property
    def output_names(self):
        return {buffer.name for buffer in output_buffers(self.outputs)}
