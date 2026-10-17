from __future__ import annotations

from .dependencies import extract_read_writes, topological_sort
from .ir import ComputedBuffer


class SchedulerNode:
    def __init__(self, buffer):
        self.node = buffer
        self.read_writes = extract_read_writes(buffer)
        self.reads = {dep.name for dep in self.read_writes.reads}
        self.writes = {dep.name for dep in self.read_writes.writes}

    def get_nodes(self):
        return [self]

    def codegen(self, index):
        self.node.codegen(index)


class FusedSchedulerNode:
    def __init__(self, members):
        self.members = topological_sort(members)
        self.writes = set().union(*(node.writes for node in members))
        self.reads = set().union(*(node.reads for node in members)) - self.writes

    def get_nodes(self):
        return self.members


class Scheduler:
    def __init__(self, graph, *, fusion=True, max_fused_nodes=32):
        from .codegen.cuda_combined_scheduling import CUDACombinedScheduling

        if max_fused_nodes < 1:
            raise ValueError("max_fused_nodes must be positive")
        self.graph = graph
        self.max_fused_nodes = max_fused_nodes
        self.backend = CUDACombinedScheduling()
        self.nodes = [self.create_scheduler_node(op) for op in graph.operations]
        self.compute_dependencies()
        self.nodes = topological_sort(self.nodes)
        self.before_fusion = self.describe_nodes()
        self.fusion_log = []
        if fusion:
            self.fuse_nodes()

    def create_scheduler_node(self, op):
        if isinstance(op, ComputedBuffer):
            node = SchedulerNode(op)
            if any(dep.index != "i" for dep in node.read_writes.reads | node.read_writes.writes):
                raise NotImplementedError("v1 requires identity indexing; extend fusion legality before adding views")
            return node
        raise NotImplementedError(f"no scheduler node for {type(op).__name__}")

    def compute_dependencies(self):
        buffers = [*self.graph.inputs, *self.graph.operations]
        if len({buffer.name for buffer in buffers}) != len(buffers):
            raise ValueError("buffer names must be unique")
        self.users = {buffer.name: set() for buffer in buffers}
        for node in self.nodes:
            for name in node.reads:
                if name not in self.users:
                    raise ValueError(f"missing producer/input: {name}")
                if name in node.writes:
                    raise ValueError("in-place mutation is outside v1")
                self.users[name].add(node.node.name)
        if not self.graph.output_names <= self.users.keys():
            raise ValueError("output has no producer/input")

    def contracted_nodes(self, first, second):
        fused = FusedSchedulerNode(first.get_nodes() + second.get_nodes())
        return [fused if node is first else node for node in self.nodes if node is not second]

    def can_fuse(self, first, second):
        if len(first.get_nodes()) + len(second.get_nodes()) > self.max_fused_nodes:
            return False
        shared = (first.reads | first.writes) & (second.reads | second.writes)
        if not shared or not self.backend.can_fuse(first, second):
            return False
        try:
            topological_sort(self.contracted_nodes(first, second))
        except ValueError:
            return False
        return True

    def score_fusion(self, first, second):
        intermediates = (first.writes & second.reads) | (second.writes & first.reads)
        shared_reads = first.reads & second.reads
        buffers = self.graph.buffers
        return sum(buffers[name].layout.numel * 4 * 2 for name in intermediates), len(shared_reads)

    def get_possible_fusions(self):
        candidates = [
            (first, second)
            for i, first in enumerate(self.nodes)
            for second in self.nodes[i + 1:]
            if self.can_fuse(first, second)
        ]
        return sorted(candidates, key=lambda pair: self.score_fusion(*pair), reverse=True)

    def fuse_nodes(self):
        while True:
            candidates = self.get_possible_fusions()
            if not candidates:
                return
            first, second = candidates[0]
            self.fusion_log.append({
                "left": [n.node.name for n in first.get_nodes()],
                "right": [n.node.name for n in second.get_nodes()],
                "score": self.score_fusion(first, second),
            })
            self.nodes = topological_sort(self.contracted_nodes(first, second))

    def materialized_outputs(self, group):
        members = {node.node.name for node in group.get_nodes()}
        return [
            node.node.name for node in group.get_nodes()
            if node.node.name in self.graph.output_names
            or self.users[node.node.name] - members
            or not self.users[node.node.name]
        ]

    def describe_nodes(self):
        return [{
            "type": type(group).__name__,
            "members": [node.node.name for node in group.get_nodes()],
            "reads": sorted(group.reads),
            "writes": sorted(group.writes),
            "materialized": self.materialized_outputs(group),
        } for group in self.nodes]

    def codegen(self):
        return [
            self.backend.codegen_node(group, f"triton_pw_{i}", self.materialized_outputs(group))
            for i, group in enumerate(self.nodes)
        ]
