import argparse

import torch

from inductor_demo import TensorSpec, compile_fx


def expression(a, b, c):
    return a + b * c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=1025)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--unfused", action="store_true")
    parser.add_argument("--sm", type=int, help="offline cubin target, e.g. 80 or 89")
    parser.add_argument("--run", action="store_true", help="load cubins and compare with eager PyTorch")
    options = parser.parse_args()
    if options.run and options.source_only:
        parser.error("--run requires compiled cubins")
    gm = torch.fx.symbolic_trace(expression)
    specs = [TensorSpec((options.size,)) for _ in range(3)]
    result = compile_fx(
        gm, specs, options.output, fusion=not options.unfused,
        emit_cubin=not options.source_only, sm=options.sm,
    )
    print(gm.graph)
    print(result.source_path.read_text())
    print(f"kernels={len(result.kernels)}, storage_slots={len(result.memory_plan.slots)}")
    print(f"artifacts: {result.directory}")
    if options.run:
        torch.manual_seed(0)
        inputs = [torch.randn(spec.shape, device="cuda") for spec in specs]
        actual = result.load()(*inputs)
        torch.testing.assert_close(actual, expression(*inputs), rtol=0, atol=0)
        torch.cuda.synchronize()
        print("CUDA result matches eager PyTorch exactly")


if __name__ == "__main__":
    main()
