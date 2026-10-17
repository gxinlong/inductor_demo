import ast
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

from inductor_demo import TensorSpec, compile_ir
from inductor_demo.ir import GraphIR, InputBuffer, ComputedBuffer, Pointwise
from inductor_demo.memory import plan_memory
from inductor_demo.scheduler import FusedSchedulerNode, Scheduler
from inductor_demo.virtualized import V


def graph_inputs(shape=(1025,)):
    inputs = [InputBuffer(f"arg{i}", TensorSpec(shape)) for i in range(3)]
    return GraphIR(inputs=inputs), inputs


class ScalarHandler:
    def __init__(self, values):
        self.values = values

    def load(self, name, index):
        return self.values[name]

    def store(self, name, index, value):
        self.values[name] = value

    def add(self, lhs, rhs):
        return lhs + rhs

    def mul(self, lhs, rhs):
        return lhs * rhs


class TestCore(unittest.TestCase):
    def test_fused_pointwise_and_wrapper(self):
        graph, (a, b, c) = graph_inputs()
        product = graph.register_pointwise("mul", b, c, "mul")
        graph.outputs = graph.register_pointwise("add", a, product, "add")
        with tempfile.TemporaryDirectory() as directory:
            result = compile_ir(graph, directory)
            self.assertEqual(len(result.scheduler.before_fusion), 2)
            self.assertEqual(len(result.kernels), 1)
            self.assertIsInstance(result.scheduler.nodes[0], FusedSchedulerNode)
            source = result.source_path.read_text()
            self.assertEqual(source.count("tl.load("), 3)
            self.assertEqual(source.count("tl.store("), 1)
            self.assertNotIn("buf0", source)
            self.assertEqual(result.memory_plan.assignments, {"buf1": 0})
            self.assertIn("xindex < 1025", source)
            ast.parse(result.wrapper_path.read_text())
            self.assertEqual(result.cubin_paths, [])
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertFalse(manifest["compiled"])
            with self.assertRaises(FileExistsError):
                compile_ir(graph, directory)

    def test_unfused_intermediate_storage(self):
        graph, (a, b, c) = graph_inputs()
        product = graph.register_pointwise("mul", b, c, "mul")
        graph.outputs = graph.register_pointwise("add", a, product, "add")
        scheduler = Scheduler(graph, fusion=False)
        self.assertEqual(len(scheduler.nodes), 2)
        self.assertEqual(len(plan_memory(scheduler).slots), 2)
        self.assertIn("tl.load(buf0", scheduler.codegen()[1].source)

    def test_contraction_cycle_is_rejected(self):
        graph, (a, b, c) = graph_inputs()
        x = graph.register_pointwise("add", a, b, "x")
        y = graph.register_pointwise("mul", x, c, "y")
        graph.outputs = graph.register_pointwise("add", x, y, "z")
        scheduler = Scheduler(graph, fusion=False)
        self.assertFalse(scheduler.can_fuse(scheduler.nodes[0], scheduler.nodes[2]))
        self.assertTrue(scheduler.can_fuse(scheduler.nodes[0], scheduler.nodes[1]))

    def test_returned_intermediate_keeps_store(self):
        graph, (a, b, c) = graph_inputs()
        x = graph.register_pointwise("mul", b, c, "x")
        y = graph.register_pointwise("add", a, x, "y")
        graph.outputs = (x, y, x, a)
        scheduler = Scheduler(graph)
        self.assertEqual(len(scheduler.nodes), 1)
        kernel = scheduler.codegen()[0]
        self.assertEqual(kernel.outputs, [x.name, y.name])
        self.assertEqual(kernel.source.count("tl.store("), 2)
        plan = plan_memory(scheduler)
        self.assertNotEqual(plan.assignments[x.name], plan.assignments[y.name])

    def test_external_user_keeps_materialization(self):
        graph, (a, b, c) = graph_inputs()
        x = graph.register_pointwise("mul", b, c, "x")
        y = graph.register_pointwise("add", a, x, "y")
        z = graph.register_pointwise("mul", x, c, "z")
        graph.outputs = (y, z)
        scheduler = Scheduler(graph, max_fused_nodes=2)
        self.assertEqual(len(scheduler.nodes), 2)
        self.assertIn(x.name, scheduler.materialized_outputs(scheduler.nodes[0]))
        self.assertIn(x.name, scheduler.nodes[1].reads)

    def test_memory_reuse_after_last_read(self):
        graph, (a, b, c) = graph_inputs()
        x = graph.register_pointwise("add", a, b, "x")
        y = graph.register_pointwise("mul", x, c, "y")
        graph.outputs = graph.register_pointwise("add", y, a, "z")
        plan = plan_memory(Scheduler(graph, fusion=False))
        self.assertEqual(len(plan.slots), 2)
        self.assertEqual(plan.assignments, {"buf0": 0, "buf1": 1, "buf2": 0})
        self.assertEqual(plan.last_use["buf0"], 1)
        self.assertEqual(plan.last_use["buf2"], 3)

    def test_no_cse_or_dce(self):
        graph, (a, b, _) = graph_inputs()
        x = graph.register_pointwise("mul", a, b, "x")
        graph.register_pointwise("mul", a, b, "unused_duplicate")
        graph.outputs = x
        scheduler = Scheduler(graph)
        self.assertEqual(len(scheduler.nodes[0].get_nodes()), 2)
        kernel = scheduler.codegen()[0]
        self.assertEqual(kernel.source.count("tl.load("), 4)
        self.assertEqual(kernel.source.count("tl.store("), 2)

    def test_topological_sort_and_handler_restoration(self):
        graph, (a, b, c) = graph_inputs()
        x = graph.register_pointwise("mul", b, c, "x")
        graph.outputs = graph.register_pointwise("add", a, x, "y")
        graph.operations.reverse()
        scheduler = Scheduler(graph)
        values = {"arg0": 2, "arg1": 3, "arg2": 5}
        with V.set_ops_handler(ScalarHandler(values)) as handler:
            with V.set_ops_handler(object()):
                pass
            self.assertIs(V.ops, handler)
            for member in scheduler.nodes[0].get_nodes():
                member.codegen("i")
        self.assertEqual(values[graph.outputs.name], 17)

    def test_shapes_and_unknown_dependencies_rejected(self):
        graph, (a, b, _) = graph_inputs()
        with self.assertRaises(NotImplementedError):
            graph.register_pointwise("add", a, InputBuffer("other", TensorSpec((3,))), "bad")
        with self.assertRaises(NotImplementedError):
            graph.register_pointwise("add", a, 3, "scalar")
        x = graph.register_pointwise("mul", a, b, "x")
        graph.inputs.pop(1)
        graph.outputs = x
        with self.assertRaisesRegex(ValueError, "missing producer"):
            Scheduler(graph)

    def test_nonidentity_index_rejected(self):
        graph, (a, _, _) = graph_inputs()
        op = Pointwise(a.layout.shape, lambda i: V.ops.load(a.name, i + "+1"), "shift")
        graph.outputs = ComputedBuffer("buf0", a.layout, op, "shift")
        graph.operations.append(graph.outputs)
        with self.assertRaisesRegex(NotImplementedError, "identity indexing"):
            Scheduler(graph)

    def test_zero_size_and_passthrough(self):
        graph, (a, b, _) = graph_inputs((0,))
        graph.outputs = graph.register_pointwise("add", a, b, "empty")
        with tempfile.TemporaryDirectory() as directory:
            result = compile_ir(graph, directory)
            self.assertNotIn("kernel_0[(", result.wrapper_path.read_text())
        graph, (a, _, _) = graph_inputs(())
        graph.outputs = [a, (a,)]
        with tempfile.TemporaryDirectory() as directory:
            result = compile_ir(graph, directory)
            self.assertEqual(result.kernels, [])
            self.assertEqual(result.memory_plan.slots, [])
            self.assertIn("return [arg0, (arg0,)]", result.wrapper_path.read_text())

    def test_generated_source_with_storage_reuse(self):
        class Pointer:
            def __init__(self, values):
                self.values = values

            def __add__(self, index):
                return self.values, index

        def load(pointer, mask, other):
            values, index = pointer
            return values[index] if mask else other

        def store(pointer, value, mask):
            if mask:
                values, index = pointer
                values[index] = value

        rng = random.Random(0)
        for _ in range(20):
            graph, inputs = graph_inputs((9,))
            buffers = list(inputs)
            for i in range(10):
                buffers.append(graph.register_pointwise(
                    rng.choice(["add", "mul"]), rng.choice(buffers), rng.choice(buffers), f"op{i}",
                ))
            graph.outputs = (buffers[-1], buffers[-3])
            values = {buffer.name: [rng.randrange(-1, 2) for _ in range(9)] for buffer in inputs}
            expected = {name: [] for name in graph.output_names}
            for lane in range(9):
                handler = ScalarHandler({name: items[lane] for name, items in values.items()})
                with V.set_ops_handler(handler):
                    for op in graph.operations:
                        op.codegen("i")
                for name in expected:
                    expected[name].append(handler.values[name])

            for fusion in (False, True):
                scheduler = Scheduler(graph, fusion=fusion, max_fused_nodes=3)
                plan = plan_memory(scheduler)
                slots = [Pointer([None] * 9) for _ in plan.slots]
                pointers = {name: Pointer(items[:]) for name, items in values.items()}
                pointers.update({name: slots[slot] for name, slot in plan.assignments.items()})
                for kernel in scheduler.codegen():
                    tree = ast.parse(kernel.source)
                    tree.body[0].decorator_list = []
                    # Scalar host interpretation checks source dataflow, not GPU numerics.
                    tl = SimpleNamespace(program_id=lambda axis: 0, load=load, store=store)
                    scope = {"tl": tl}
                    exec(compile(tree, "generated_kernel", "exec"), scope)
                    for lane in range(kernel.block):
                        tl.arange = lambda start, end: lane
                        scope[kernel.name](*(pointers[name] for name in kernel.args))
                for name in expected:
                    self.assertEqual(pointers[name].values, expected[name])


if __name__ == "__main__":
    unittest.main()
