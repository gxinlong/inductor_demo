"""Export a weighted pointwise model, compile it, and run saved cubins directly."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from unittest import mock

import torch
import triton

from inductor_demo import compile_fx
from inductor_demo.ir import output_buffers
from inductor_demo.runtime import load_kernel


class PointwiseResidual(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(shape, dtype=torch.float32) * 0.25 + 1)
        self.bias = torch.nn.Parameter(torch.randn(shape, dtype=torch.float32) * 0.125)
        self.gate = torch.nn.Parameter(torch.randn(shape, dtype=torch.float32) * 0.125 + 0.5)
        self.requires_grad_(False)

    def forward(self, x, residual):
        hidden = x * self.scale + self.bias
        return hidden * self.gate + residual


class KernelForward:
    """An explicit launch loop for teaching; generated wrapper.py is not imported."""

    def __init__(self, directory, binding, weights, output_names, device):
        manifest = json.loads((directory / "manifest.json").read_text())
        schedule = json.loads((directory / "schedule.json").read_text())
        self.binding = binding
        self.weights = weights
        self.output_names = output_names
        self.entries = manifest["kernels"]
        self.plan = schedule["memory"]
        with torch.cuda.device(device):
            self.kernels = [load_kernel(directory, entry["name"]) for entry in self.entries]
        self.slots = [
            torch.empty(spec["shape"], dtype=torch.float32, device=device)
            for spec in self.plan["slots"]
        ]

    def __call__(self, x, residual):
        user_inputs = {"x": x, "residual": residual}
        args = [
            self.weights[item["target"]] if item["kind"] == "PARAMETER" else user_inputs[item["name"]]
            for item in self.binding
        ]
        buffers = {f"arg{i}": tensor for i, tensor in enumerate(args)}
        buffers.update({name: self.slots[slot] for name, slot in self.plan["assignments"].items()})
        with torch.cuda.device(x.device):
            for entry, kernel in zip(self.entries, self.kernels):
                if entry["numel"]:
                    grid = (entry["numel"] + entry["block"] - 1) // entry["block"]
                    kernel[(grid, 1, 1)](*(buffers[name] for name in entry["args"]))
        return tuple(buffers[name] for name in self.output_names)


def environment():
    major, minor = torch.cuda.get_device_capability()
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "triton_file": triton.__file__,
        "gpu": torch.cuda.get_device_name(),
        "sm": major * 10 + minor,
    }


def export_and_compile(root, shapes, device):
    root.mkdir(parents=True, exist_ok=False)
    (root / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    cases = []
    for shape in shapes:
        name = "shape_" + ("x".join(map(str, shape)) or "scalar")
        directory = root / name
        directory.mkdir()
        torch.manual_seed(2026)
        model = PointwiseResidual(shape).eval().to(device)
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save({"shape": shape, "state_dict": state}, directory / "weights.pt")
        examples = [torch.randn(shape, device=device) for _ in range(2)]
        torch.save([tensor.cpu() for tensor in examples], directory / "example_inputs.pt")
        exported = torch.export.export(model, tuple(examples))
        gm = exported.graph_module
        (directory / "fx_graph.txt").write_text(str(gm.graph) + "\n")
        (directory / "fx_graph.py").write_text(gm.code)
        binding = [
            {"kind": item.kind.name, "name": item.arg.name, "target": item.target}
            for item in exported.graph_signature.input_specs
        ]
        if any(item["kind"] not in {"PARAMETER", "USER_INPUT"} for item in binding):
            raise RuntimeError("unexpected graph input kind")
        (directory / "bindings.json").write_text(json.dumps(binding, indent=2) + "\n")
        parameters = dict(model.named_parameters())
        user_inputs = dict(zip(("x", "residual"), examples))
        args = [
            parameters[item["target"]].detach() if item["kind"] == "PARAMETER" else user_inputs[item["name"]]
            for item in binding
        ]
        torch.testing.assert_close(gm(*args)[0], model(*examples), rtol=0, atol=0)
        modes = {}
        for fusion in (True, False):
            mode = "fused" if fusion else "unfused"
            result = compile_fx(gm, args, directory / mode, fusion=fusion)
            if len(result.kernels) != (1 if fusion else 4):
                raise AssertionError(f"unexpected {mode} kernel count: {len(result.kernels)}")
            binaries = {}
            for path in result.cubin_paths:
                content = path.read_bytes()
                if content[:4] != b"\x7fELF":
                    raise AssertionError(f"invalid cubin ELF header: {path}")
                binaries[str(path.relative_to(directory))] = {
                    "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                }
            modes[mode] = {
                "kernels": len(result.kernels),
                "storage_slots": len(result.memory_plan.slots),
                "output_names": [buffer.name for buffer in output_buffers(result.scheduler.graph.outputs)],
                "cubins": binaries,
            }
        cases.append({"name": name, "shape": shape, "modes": modes})
        print(f"COMPILED {name}: fused=1 kernel, unfused=4 kernels", flush=True)
    (root / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")


def check_forward(root, seeds, device, report_name):
    report_path = root / report_name
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite: {report_path}")
    cases = json.loads((root / "cases.json").read_text())
    report = {"environment": environment(), "gpu_recompilation_disabled": True, "results": []}
    with mock.patch.object(triton, "compile", side_effect=AssertionError("unexpected GPU recompilation")):
        for case in cases:
            directory = root / case["name"]
            checkpoint = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
            model = PointwiseResidual(checkpoint["shape"]).eval()
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device)
            weights = {name: value.to(device) for name, value in checkpoint["state_dict"].items()}
            binding = json.loads((directory / "bindings.json").read_text())
            runners = {
                mode: KernelForward(directory / mode, binding, weights, info["output_names"], device)
                for mode, info in case["modes"].items()
            }
            samples = []
            examples = torch.load(directory / "example_inputs.pt", weights_only=True)
            samples.append(("saved_inputs", *(value.to(device) for value in examples)))
            for seed in seeds:
                torch.manual_seed(seed)
                samples.append((f"seed_{seed}", *[torch.randn(case["shape"], device=device) for _ in range(2)]))
            zero = torch.zeros(case["shape"], device=device)
            samples.append(("zeros", zero, zero))
            x = torch.ones(case["shape"], device=device)
            samples.append(("cancellation", x, -((x * model.scale + model.bias) * model.gate)))
            for sample, x, residual in samples:
                expected = model(x, residual)
                if not torch.isfinite(expected).all().item():
                    raise AssertionError("reference contains nonfinite values")
                for mode, runner in runners.items():
                    actual, = runner(x, residual)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    error = (actual - expected).abs()
                    report["results"].append({
                        "case": case["name"], "sample": sample, "mode": mode,
                        "numel": actual.numel(),
                        "max_abs_error": error.max().item() if actual.numel() else 0.0,
                        "mismatches": (actual != expected).count_nonzero().item(),
                    })
            print(f"PASS {case['name']}: {len(samples) * len(runners)} exact forward comparisons", flush=True)
    report["passed"] = True
    report["comparisons"] = len(report["results"])
    report["max_abs_error"] = max(result["max_abs_error"] for result in report["results"])
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"ALL PASS: {report['comparisons']} comparisons; max_abs_error={report['max_abs_error']}; {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="new export/compile directory")
    action.add_argument("--replay", type=Path, help="load saved weights and cubins in a fresh process")
    parser.add_argument("--shapes", nargs="+", default=["1", "255", "256", "1025", "17x257"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 17, 42])
    parser.add_argument("--report-name", default="forward_report.json")
    options = parser.parse_args()
    if Path(options.report_name).name != options.report_name:
        parser.error("--report-name must be a file name")
    if not torch.cuda.is_available() or torch.version.hip:
        parser.error("an NVIDIA CUDA device is required")
    device = torch.device("cuda", torch.cuda.current_device())
    with torch.inference_mode():
        if options.output:
            shapes = [() if value == "scalar" else tuple(map(int, value.split("x"))) for value in options.shapes]
            root = options.output.resolve()
            export_and_compile(root, shapes, device)
        else:
            root = options.replay.resolve()
        check_forward(root, options.seeds, device, options.report_name)


if __name__ == "__main__":
    main()
