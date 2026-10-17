"""Enter at IR without torch/triton; this is not an FX or GPU test."""

import argparse

from inductor_demo import TensorSpec, compile_ir
from inductor_demo.ir import GraphIR, InputBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--unfused", action="store_true")
    options = parser.parse_args()
    spec = TensorSpec((1025,))
    a, b, c = [InputBuffer(f"arg{i}", spec) for i in range(3)]
    graph = GraphIR(inputs=[a, b, c])
    product = graph.register_pointwise("mul", b, c, "mul")
    graph.outputs = graph.register_pointwise("add", a, product, "add")
    result = compile_ir(graph, options.output, fusion=not options.unfused)
    print(result.source_path.read_text())
    print((result.directory / "schedule.json").read_text())


if __name__ == "__main__":
    main()
