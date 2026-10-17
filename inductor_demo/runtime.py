"""Small, version-bounded bridge to Triton's compiler and cubin launcher."""

import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def load_python_module(path):
    path = Path(path).resolve()
    name = "_inductor_demo_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def require_triton():
    import triton

    if sys.platform != "linux" or triton.__version__.split(".")[:2] != ["3", "3"]:
        raise RuntimeError("cubin export/load requires Linux and Triton 3.3.x (internal API boundary)")
    return triton


def make_ast_source(module, entry):
    from triton.compiler import ASTSource

    signature = {name: "*fp32" for name in entry["args"]}
    return ASTSource(getattr(module, entry["name"]), signature=signature, constexprs={})


def compile_cubins(directory, *, sm=None):
    """With an explicit SM, Triton can compile offline on a supported Linux host."""
    triton = require_triton()
    from triton.backends.compiler import GPUTarget

    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["compiled"]:
        raise ValueError("artifacts are already compiled; generate a fresh directory")
    if sm is None:
        import torch

        if not torch.cuda.is_available() or torch.version.hip:
            raise RuntimeError("CUDA is required to infer SM; alternatively pass sm=80, etc. on Linux")
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor
    if type(sm) is not int or sm <= 0:
        raise ValueError("sm must be a positive integer, e.g. 80 or 89")
    module = load_python_module(directory / "kernels.py")
    for entry in manifest["kernels"]:
        binary = triton.compile(
            make_ast_source(module, entry),
            target=GPUTarget("cuda", sm, 32),
            options={"num_warps": entry["num_warps"], "num_ctas": 1, "enable_fp_fusion": False},
        )
        folder = directory / entry["name"]
        folder.mkdir(exist_ok=True)
        files = {}
        for extension, content in binary.asm.items():
            path = folder / f"{entry['name']}.{extension}"
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content)
            files[path.name] = str(path.relative_to(directory))
        cubin = folder / f"{entry['name']}.cubin"
        if not cubin.is_file() or not cubin.stat().st_size:
            raise RuntimeError("Triton did not produce a cubin")
        metadata = binary.metadata._asdict()
        metadata["target"] = dataclasses.asdict(metadata["target"])
        path = folder / f"{entry['name']}.json"
        path.write_text(json.dumps(metadata, indent=2) + "\n")
        files[path.name] = str(path.relative_to(directory))
        entry.update(files=files, hash=binary.hash, cubin=str(cubin.relative_to(directory)))

    manifest.update(compiled=True, sm=sm, triton_version=triton.__version__)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_kernel(directory, name):
    """Recreate CompiledKernel from exported files; no triton.compile/JIT fallback."""
    triton = require_triton()
    import torch
    from triton.compiler.compiler import CompiledKernel

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["compiled"]:
        raise RuntimeError("source-only artifacts: call compile_cubins(directory, sm=...) first")
    if manifest["triton_version"] != triton.__version__:
        raise RuntimeError("load artifacts with the exact Triton version used to compile them")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor != manifest["sm"]:
        raise RuntimeError("demo requires the runtime GPU SM to match the compiled target")
    entry = next(item for item in manifest["kernels"] if item["name"] == name)
    module = load_python_module(directory / "kernels.py")
    metadata_group = {key: str(directory / path) for key, path in entry["files"].items()}
    return CompiledKernel(make_ast_source(module, entry), metadata_group, entry["hash"])


def check_inputs(args, shapes):
    import torch

    if len(args) != len(shapes) or not args:
        raise ValueError(f"expected {len(shapes)} tensor inputs")
    if not all(isinstance(tensor, torch.Tensor) for tensor in args):
        raise TypeError("all inputs must be torch.Tensor")
    device = args[0].device
    for tensor, shape in zip(args, shapes):
        if tensor.device.type != "cuda" or tensor.device != device or torch.version.hip:
            raise ValueError("all inputs must be on the same NVIDIA CUDA device")
        if tensor.dtype != torch.float32 or tensor.layout != torch.strided:
            raise ValueError("v1 requires strided float32 tensors")
        if list(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(f"expected contiguous tensor of shape {shape}")
        if tensor.requires_grad:
            raise ValueError("v1 is inference-only; detach inputs explicitly")
