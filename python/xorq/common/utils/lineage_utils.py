from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from attrs import evolve, field, frozen
from attrs.validators import instance_of

import xorq.expr.relations as rel
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
from xorq.vendor.ibis.expr.operations.core import Node
from xorq.common.utils.graph_utils import (to_node, children_of)

__all__ = [
    "GenericNode",
    "build_tree",
    "build_column_trees",
    "collect_nodes",
    "get_node_names",
    "print_ascii_tree",
]


@frozen
class GenericNode:
    op: Node = field(validator=instance_of(Node))
    children: Tuple["GenericNode", ...] = field(factory=tuple, validator=instance_of(tuple))

    def map_children(self, fn: Callable[["GenericNode"], "GenericNode"]) -> "GenericNode":
        return evolve(self, children=tuple(fn(c) for c in self.children))

    def clone(self, **changes: Any) -> "GenericNode":
        return evolve(self, **changes)


def build_tree(node: Node) -> GenericNode:
    return GenericNode(op=node, children=tuple(build_tree(c) for c in children_of(node)))


def build_column_trees(expr: Any) -> Dict[str, GenericNode]:
    op = to_node(expr).op() if hasattr(expr, "op") else to_node(expr)

    # Project / Aggregation: look for ``values`` or ``fields`` dict attrs.
    cols: Dict[str, Any] = getattr(op, "values", getattr(op, "fields", {}))
    if cols:
        return {k: build_tree(to_node(v)) for k, v in cols.items()}

    # Fallback – build synthetic Field nodes for tables.
    schema = getattr(expr, "schema", lambda: None)()
    if schema is None:
        return {}
    return {
        name: build_tree(ops.Field(rel=op, name=name))
        for name in schema.names
    }


def collect_nodes(node: GenericNode, predicate: Optional[Callable[[Node], bool]] = None) -> Tuple[Node, ...]:
    def _collector(cur: GenericNode, acc: list[Node]):
        if predicate is None or predicate(cur.op):
            acc.append(cur.op)
        for child in cur.children:
            _collector(child, acc)

    out: list[Node] = []
    _collector(node, out)
    return tuple(out)


def _label(node: Node) -> str:
    name = getattr(node, "name", None)
    return f"{name} ({type(node).__name__})" if name else type(node).__name__


def get_node_names(node: GenericNode) -> Tuple[str, ...]:
    return tuple(_label(n) for n in collect_nodes(node))


def print_tree(node: GenericNode, *, indent: str = "", is_last: bool = True) -> None:
    connector = "└─" if is_last else "├─"
    print(f"{indent}{connector} {_label(node.op)}")
    new_indent = indent + ("   " if is_last else "│  ")
    for idx, child in enumerate(node.children):
        print_ascii_tree(child, indent=new_indent, is_last=idx == len(node.children) - 1)
