import torch

from .ir import GraphIR, InputBuffer, TensorSpec, output_buffers
from .lowering import lowerings


class GraphLowering(torch.fx.Interpreter):
    def __init__(self, gm):
        super().__init__(gm)
        self.ir = GraphIR()
        self.current_node = None

    def run_node(self, node):
        self.current_node = node
        return super().run_node(node)

    def placeholder(self, target, args, kwargs):
        spec = super().placeholder(target, args, kwargs)
        if not isinstance(spec, TensorSpec):
            raise TypeError("each FX placeholder needs a TensorSpec")
        buffer = InputBuffer(f"arg{len(self.ir.inputs)}", spec)
        self.ir.inputs.append(buffer)
        return buffer

    def call_function(self, target, args, kwargs):
        if target not in lowerings:
            raise NotImplementedError(f"no lowering for {target}; register it in lowering.py")
        return lowerings[target](self.ir, *args, origin=self.current_node.name, **kwargs)

    def output(self, target, args, kwargs):
        self.ir.outputs = args[0]
        output_buffers(self.ir.outputs)
        return self.ir

    def get_attr(self, target, args, kwargs):
        raise NotImplementedError("get_attr/constants are outside v1")

    def call_method(self, target, args, kwargs):
        raise NotImplementedError("use functional add/mul, not call_method")

    def call_module(self, target, args, kwargs):
        raise NotImplementedError("call_module is outside v1")
