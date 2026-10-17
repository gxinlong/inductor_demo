from ..dependencies import topological_sort
from ..virtualized import V


class SIMDScheduling:
    def can_fuse(self, first, second):
        nodes = first.get_nodes() + second.get_nodes()
        return all(node.node.layout == nodes[0].node.layout for node in nodes)

    def generate_node_schedule(self, node):
        return topological_sort(node.get_nodes())

    def codegen_node_schedule_with_kernel(self, node_schedule, kernel):
        with V.set_ops_handler(kernel):
            for node in node_schedule:
                node.codegen("xindex")

    def codegen_node(self, node, name, outputs):
        node_schedule = self.generate_node_schedule(node)
        kernel = self.kernel_type(name, node_schedule, outputs)
        self.codegen_node_schedule_with_kernel(node_schedule, kernel)
        return kernel.codegen_kernel()
