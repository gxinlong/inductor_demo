from .triton import TritonScheduling


class CUDACombinedScheduling:
    def __init__(self):
        self.triton = TritonScheduling()

    def choose_node_backend(self, node):
        kinds = {member.node.kind for member in node.get_nodes()}
        if kinds == {"pointwise"}:
            return self.triton
        raise NotImplementedError(f"register a template scheduling path here: {kinds}")

    def can_fuse(self, first, second):
        backend = self.choose_node_backend(first)
        return backend is self.choose_node_backend(second) and backend.can_fuse(first, second)

    def codegen_node(self, node, name, outputs):
        return self.choose_node_backend(node).codegen_node(node, name, outputs)
