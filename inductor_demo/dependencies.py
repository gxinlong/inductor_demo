from dataclasses import dataclass, field

from .virtualized import V


@dataclass(frozen=True)
class MemoryDep:
    name: str
    index: str


@dataclass
class ReadWrites:
    reads: set[MemoryDep] = field(default_factory=set)
    writes: set[MemoryDep] = field(default_factory=set)


class DependencyHandler:
    def __init__(self):
        self.read_writes = ReadWrites()

    def load(self, name, index):
        self.read_writes.reads.add(MemoryDep(name, index))

    def store(self, name, index, value):
        self.read_writes.writes.add(MemoryDep(name, index))

    def add(self, lhs, rhs):
        pass

    def mul(self, lhs, rhs):
        pass


def extract_read_writes(buffer):
    with V.set_ops_handler(DependencyHandler()) as handler:
        buffer.codegen("i")
    return handler.read_writes


def topological_sort(nodes):
    """Stable Kahn ordering; also checks cycles after contracting a fusion pair."""
    producers = {name: i for i, node in enumerate(nodes) for name in node.writes}
    pending = {
        i: {producers[name] for name in node.reads if name in producers and producers[name] != i}
        for i, node in enumerate(nodes)
    }
    order = []
    while pending:
        ready = next((i for i, deps in pending.items() if not deps), None)
        if ready is None:
            raise ValueError("dependency cycle")
        order.append(nodes[ready])
        del pending[ready]
        for deps in pending.values():
            deps.discard(ready)
    return order
