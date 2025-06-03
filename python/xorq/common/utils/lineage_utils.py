from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple
from attrs import evolve, field, frozen
from attrs.validators import instance_of

import xorq.expr.relations as rel
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
from xorq.vendor.ibis.expr.operations.core import Node


def _to_node(maybe_expr: Any) -> Node:
    """Normalize *anything* that quacks like an Ibis Expr into a Node."""
    while not isinstance(maybe_expr, Node):
        if hasattr(maybe_expr, "op"):
            maybe_expr = maybe_expr.op()
        else:
            raise TypeError(f"Cannot convert {type(maybe_expr)} into an ibis Node")
    return maybe_expr


def get_children(node: Node) -> List[Node]:
    """Get children with special handling for RemoteTable, FlightUDXF, etc."""
    children = []

    # Special handling for Field nodes to avoid including entire Project
    if isinstance(node, ops.Field):
        if (rel_node := node.rel) is not None:
            if isinstance(rel_node, ops.Project):
                # For Project, follow the specific expression for this field
                _, mapping = rel_node.args
                if node.name in mapping:
                    raw_expr = mapping[node.name]
                    children.append(_to_node(raw_expr))
                    return children
            # For non-Project relations, follow the relation
            children.append(_to_node(rel_node))
        return children

    if isinstance(node, rel.RemoteTable):
        remote_expr = node.remote_expr
        try:
            children.append(_to_node(remote_expr))
        except (AttributeError, TypeError):
            pass  # Skip if we can't convert to node
        return children

    if isinstance(node, rel.CachedNode):
        children.append(_to_node(node.parent))
        return children

    if isinstance(node, rel.FlightExpr):
        children.append(_to_node(node.input_expr))
        return children

    if isinstance(node, rel.FlightUDXF):
        children.append(_to_node(node.input_expr))
        return children

    if isinstance(node, udf.ExprScalarUDF):
        exprs = node.computed_kwargs_expr
        if isinstance(exprs, Node):
            children.append(exprs)
        elif exprs is not None:
            for item in exprs:
                try:
                    children.append(_to_node(item))
                except (TypeError, AttributeError):
                    pass  # Skip items that can't be converted to nodes
        return children

    if isinstance(node, rel.Read):
        return []

    # Default case: use __children__
    raw_children = getattr(node, "__children__", ())
    for child in raw_children:
        try:
            children.append(_to_node(child))
        except (TypeError, AttributeError):
            pass  # Skip items that can't be converted to nodes

    return children


@frozen
class GenericNode:
    """A simple node in a tree with no edge semantics."""
    op: Node = field(validator=instance_of(Node))
    children: Tuple["GenericNode", ...] = field(
        factory=tuple, validator=instance_of(tuple)
    )

    def map_children(self, fn: Callable[["GenericNode"], "GenericNode"]) -> "GenericNode":
        return evolve(self, children=tuple(fn(c) for c in self.children))

    def clone(self, **changes: Any) -> "GenericNode":
        return evolve(self, **changes)


def build_tree(node: Node) -> GenericNode:
    """Build a generic tree from any node by recursively traversing children."""
    raw_children = get_children(node)
    children = tuple(build_tree(child) for child in raw_children)
    return GenericNode(op=node, children=children)


def build_column_trees(expr: Any) -> Dict[str, GenericNode]:
    """Return mapping column-name → tree for all columns in the expression."""
    op = expr.op()

    # First try the standard way (works for Project, etc.)
    cols: Dict[str, Any] = getattr(op, "values", getattr(op, "fields", {}))

    if cols:
        return {
            name: build_tree(_to_node(col))
            for name, col in cols.items()
        }

    # For table expressions without values/fields (like RemoteTable),
    # create Field nodes for each column in the schema
    try:
        schema = expr.schema()
        return {
            name: build_tree(ops.Field(rel=op, name=name))
            for name in schema.names
        }
    except (AttributeError, TypeError):
        # Fallback: return empty dict if we can't determine columns
        return {}


def walk_tree(node: GenericNode, visitor: Callable[[Node], None]) -> None:
    """Visit every node in the tree with the given visitor function."""
    visitor(node.op)
    for child in node.children:
        walk_tree(child, visitor)


def collect_nodes(node: GenericNode, predicate: Callable[[Node], bool] = None) -> Tuple[Node, ...]:
    """Collect all nodes that match the predicate (or all nodes if no predicate)."""
    nodes = []

    def collector(op: Node) -> None:
        if predicate is None or predicate(op):
            nodes.append(op)

    walk_tree(node, collector)
    return tuple(nodes)


def _label(node: Node) -> str:
    """Get a human-readable label for a node."""
    name = getattr(node, "name", None)
    return f"{name} ({type(node).__name__})" if name else type(node).__name__


def get_node_names(node: GenericNode) -> Tuple[str, ...]:
    """Get names of all nodes in the tree."""
    return tuple(_label(op) for op in collect_nodes(node))


def flatten_tree(
    trees: Dict[str, GenericNode],
    *,
    blacklist: Tuple[str, ...] = ("RemoteTable", "DatabaseTable", "Cast(", "Project"),
) -> Dict[str, Dict[str, Tuple[str, ...]]]:
    """Collapse deep trees into `{col: {inputs, steps}}` for fast assertions."""

    def _walk(node: GenericNode, steps: set, inputs: set):
        label = _label(node.op)
        if label.endswith("(Field)"):
            inputs.add(label.split()[0])
        elif not any(b in label for b in blacklist):
            steps.add(label)

        for child in node.children:
            _walk(child, steps, inputs)

    out: Dict[str, Dict[str, Tuple[str, ...]]] = {}
    for col, tree in trees.items():
        s, i = set(), set()
        _walk(tree, s, i)
        out[col] = {"inputs": tuple(sorted(i)), "steps": tuple(sorted(s))}
    return out


def print_tree(node: GenericNode, *, indent: str = "", is_last: bool = True) -> None:
    connector = "└─" if is_last else "├─"
    print(f"{indent}{connector} {_label(node.op)}")

    new_indent = indent + ("   " if is_last else "│  ")
    for idx, child in enumerate(node.children):
        print_ascii_tree(child, indent=new_indent, is_last=idx == len(node.children) - 1)
