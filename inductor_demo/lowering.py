import operator

import torch


lowerings = {}


def register_lowering(*targets):
    def register(fn):
        for target in targets:
            if target in lowerings:
                raise ValueError(f"lowering already registered: {target}")
            lowerings[target] = fn
        return fn
    return register


@register_lowering(operator.add, torch.add, torch.ops.aten.add.Tensor)
def add(graph, lhs, rhs, *, origin, alpha=1):
    if alpha != 1:
        raise NotImplementedError("v1 supports add(alpha=1) only")
    return graph.register_pointwise("add", lhs, rhs, origin)


@register_lowering(operator.mul, torch.mul, torch.ops.aten.mul.Tensor)
def mul(graph, lhs, rhs, *, origin):
    return graph.register_pointwise("mul", lhs, rhs, origin)
