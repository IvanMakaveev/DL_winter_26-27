import graphviz

from dl_lib.engine.value import Value


def trace(root: Value):
    result_nodes = set()
    result_edges = set()
    for child in root._prev:
        nodes, edges = trace(child)
        result_nodes = result_nodes.union(nodes)
        result_edges = result_edges.union(edges)
        result_edges.add((child, root))

    result_nodes.add(root)
    return result_nodes, result_edges


def draw_dot(root: Value):
    nodes, edges = trace(root)
    dot = graphviz.Digraph(graph_attr={"rankdir": "LR"})
    for node in nodes:
        dot.node(
            name=node.label,
            label="{ "
            + f"{node.label} | data: {node.data:.4f} | grad: {node.grad:.4f}"
            + " }",
            shape="record",
        )

        if node._op is not None:
            dot.node(name=node._op + node.label, label=node._op)

    for child, parent in edges:
        dot.edge(child.label, parent._op + parent.label)
        dot.edge(parent._op + parent.label, parent.label)

    return dot


def top_sort(root: Value) -> list:
    stack = []
    visited = set()

    def _topo_helper(node: Value) -> None:
        if node in visited:
            return
        visited.add(node)

        for child in node._prev:
            _topo_helper(child)
        stack.append(node)

    _topo_helper(root)
    return stack
