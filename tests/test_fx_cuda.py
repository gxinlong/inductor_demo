"""Run explicitly in a PyTorch environment; these tests are not run by test_core."""

import json
from pathlib import Path
import tempfile
from unittest import mock

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests

from inductor_demo import TensorSpec, compile_fx
from inductor_demo.runtime import load_python_module


class TestFX(TestCase):
    def test_python_and_aten_targets(self):
        for function in (
            lambda a, b, c: a + b * c,
            lambda a, b, c: torch.add(a, torch.mul(b, c)),
            lambda a, b, c: torch.ops.aten.add.Tensor(a, torch.ops.aten.mul.Tensor(b, c)),
        ):
            gm = torch.fx.symbolic_trace(function)
            for graph in (gm, gm.graph):
                with tempfile.TemporaryDirectory() as directory:
                    result = compile_fx(graph, [TensorSpec((17,))] * 3, directory, emit_cubin=False)
                    self.assertEqual(len(result.kernels), 1)
                    self.assertEqual(result.kernels[0].source.count("tl.store("), 1)

    def test_unsupported_operator(self):
        gm = torch.fx.symbolic_trace(lambda a: torch.sin(a))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(NotImplementedError, "no lowering"):
                compile_fx(gm, [TensorSpec((3,))], directory, emit_cubin=False)


class TestCUDA(TestCase):
    @parametrize("shape", [(), (0,), (1,), (255,), (256,), (1025,), (7, 11)])
    @parametrize("fusion", [True, False])
    def test_pointwise(self, device, shape, fusion):
        import triton

        torch.manual_seed(0)
        gm = torch.fx.symbolic_trace(lambda a, b, c: a + b * c)
        inputs = [torch.randn(shape, device=device) for _ in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            result = compile_fx(gm, inputs, directory, fusion=fusion)
            self.assertEqual(len(result.cubin_paths), 1 if fusion else 2)
            for path in result.cubin_paths:
                self.assertEqual(path.read_bytes()[:4], b"\x7fELF")
            # Loading must work from exported files without compiling GPU code again.
            with mock.patch.object(triton, "compile", side_effect=AssertionError("unexpected recompilation")):
                actual = result.load()(*inputs)
            torch.cuda.synchronize()
            self.assertEqual(actual, gm(*inputs), atol=0, rtol=0)

    def test_reuse_outputs_alias_and_current_stream(self, device):
        def fn(a, b, c):
            x = a * b
            y = x + c
            z = y * a
            return z, a, z

        gm = torch.fx.symbolic_trace(fn)
        inputs = [torch.randn(1025, device=device) for _ in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            result = compile_fx(gm, inputs, directory, fusion=False)
            self.assertEqual(len(result.memory_plan.slots), 2)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                actual = result.load()(*inputs)
            stream.synchronize()
            self.assertEqual(actual, gm(*inputs), atol=0, rtol=0)
            self.assertIs(actual[1], inputs[0])
            self.assertIs(actual[0], actual[2])

    def test_returned_intermediate(self, device):
        def fn(a, b, c):
            product = b * c
            return product, a + product

        inputs = [torch.randn(513, device=device) for _ in range(3)]
        gm = torch.fx.symbolic_trace(fn)
        with tempfile.TemporaryDirectory() as directory:
            result = compile_fx(gm, inputs, directory)
            self.assertEqual(result.source_path.read_text().count("tl.store("), 2)
            self.assertEqual(result.load()(*inputs), gm(*inputs), atol=0, rtol=0)

    def test_runtime_guards(self, device):
        gm = torch.fx.symbolic_trace(lambda a, b: a + b)
        inputs = [torch.randn(8, device=device) for _ in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            result = compile_fx(gm, inputs, directory)
            wrapper = result.load()
            with self.assertRaisesRegex(ValueError, "shape"):
                wrapper(inputs[0][:4], inputs[1][:4])
            with self.assertRaisesRegex(ValueError, "inference-only"):
                wrapper(inputs[0].requires_grad_(), inputs[1])

    def test_relocated_artifacts(self, device):
        import shutil
        import triton

        gm = torch.fx.symbolic_trace(lambda a, b: a * b)
        inputs = [torch.randn(19, device=device) for _ in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original"
            relocated = Path(directory) / "relocated"
            compile_fx(gm, inputs, original)
            shutil.move(str(original), str(relocated))
            manifest = json.loads((relocated / "manifest.json").read_text())
            self.assertTrue(manifest["compiled"])
            with mock.patch.object(triton, "compile", side_effect=AssertionError("unexpected recompilation")):
                actual = load_python_module(relocated / "wrapper.py").call(*inputs)
            self.assertEqual(actual, gm(*inputs), atol=0, rtol=0)


instantiate_device_type_tests(TestCUDA, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
