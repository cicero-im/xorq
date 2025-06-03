from __future__ import annotations

import functools
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Set, Tuple

from attrs import evolve, field, frozen
from attrs.validators import instance_of, optional

import xorq.expr.relations as rel
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.graph_utils import get_children
from xorq.expr.udf import ScalarUDF
from xorq.vendor.ibis.expr.operations.core import Node


class EdgeKind(Enum):
    PARENT = auto()
    UDF = auto()
    UDF_INPUT = auto()
    RELATION = auto()
    EXPR = auto()
    METRIC = auto()
    REMOTE_EXPR = auto()

    GROUP_BY = auto()
    ORDER_BY = auto()


@frozen
class Edge:
    """Relationship between this node and its parent."""

    kind: EdgeKind = field(validator=instance_of(EdgeKind))
    detail: Optional[str] = field(default=None, validator=optional(instance_of(str)))

    def __str__(self) -> str:
        return f"{self.kind.name.lower()}{f':{self.detail}' if self.detail else ''}"


def _label(node: Node) -> str:
    name = getattr(node, "name", None)
    return f"{name} ({type(node).__name__})" if name else type(node).__name__


def _to_node(maybe_expr: Any) -> Node:
    """Normalize *anything* that quacks like an Ibis Expr into a Node."""

    while not isinstance(maybe_expr, Node):
        if hasattr(maybe_expr, "op"):
            maybe_expr = maybe_expr.op()
        else:
            raise TypeError(f"Cannot convert {type(maybe_expr)} into an ibis Node")
    return maybe_expr


@frozen
class LineageNode:
    op: Node = field(validator=instance_of(Node))
    edge: Optional[Edge] = field(default=None, validator=optional(instance_of(Edge)))
    children: Tuple["LineageNode", ...] = field(
        factory=tuple, validator=instance_of(tuple)
    )

    def map_children(self, fn: "_ChildMapper") -> "LineageNode":
        return evolve(self, children=tuple(fn(c) for c in self.children))

    def clone(self, **changes: Any) -> "LineageNode":
        return evolve(self, **changes)

    def __attrs_post_init__(self):
        for c in self.children:
            if not isinstance(c, LineageNode):
                raise TypeError("children must be LineageNode instances, got " f"{type(c).__name__}")


_ChildMapper = Callable[[LineageNode], LineageNode]


@functools.singledispatch
def _build(node: Node, target_field: Optional[str]) -> LineageNode:  # noqa: D401
    """Default builder – recurses into children via graph_utils.get_children."""

    # Special‑case ScalarUDF so args are tagged with UDF_INPUT.
    if isinstance(node, ScalarUDF):
        udf_kids: Tuple[LineageNode, ...] = tuple(
            LineageNode(op=sub.op, edge=Edge(EdgeKind.UDF_INPUT), children=sub.children)
            for arg in node.args
            if isinstance(arg, Node) and (sub := _build(arg, target_field))
        )
        return LineageNode(op=node, edge=Edge(EdgeKind.UDF), children=udf_kids)

    raw_children = get_children(node)
    kids = tuple(_build(c, target_field) for c in raw_children if isinstance(c, Node))
    return LineageNode(op=node, edge=None, children=kids)


@_build.register
def _field(node: ops.Field, target_field: Optional[str]) -> LineageNode:  # noqa: D401
    children: Tuple[LineageNode, ...] = ()
    if (rel_node := node.rel) is not None:
        if isinstance(rel_node, ops.Project):
            _, mapping = rel_node.args
            raw_expr = mapping.get(node.name)
            expr_node = _to_node(raw_expr)
            edge_kind = EdgeKind.UDF if isinstance(expr_node, ScalarUDF) else EdgeKind.EXPR
            children = (
                LineageNode(op=expr_node, edge=Edge(edge_kind), children=(
                    _build(expr_node, target_field),
                )),
            )
        else:
            children = (
                LineageNode(op=rel_node, edge=Edge(EdgeKind.RELATION), children=(
                    _build(rel_node, target_field),
                )),
            )
    return LineageNode(op=node, edge=None, children=children)


@_build.register
def _aggregate(node: ops.Aggregate, target_field: Optional[str]) -> LineageNode:  # noqa: D401
    metric_kids = tuple(
        LineageNode(
            op=expr,
            edge=Edge(EdgeKind.METRIC, name),
            children=(_build(expr, target_field),),
        )
        for name, expr in node.metrics.items()
        if target_field is None or name == target_field
    )
    parent_child = LineageNode(op=node.parent, edge=Edge(EdgeKind.PARENT), children=(
        _build(node.parent, None),
    ))
    return LineageNode(op=node, edge=None, children=metric_kids + (parent_child,))


@_build.register
def _remote_table(node: rel.RemoteTable, target_field: Optional[str]) -> LineageNode:  # noqa: D401
    remote_op = node.remote_expr.op()
    child_remote = LineageNode(op=remote_op, edge=Edge(EdgeKind.REMOTE_EXPR), children=(
        _build(remote_op, target_field),
    ))
    return LineageNode(op=node, edge=None, children=(child_remote,))


@_build.register
def _project(node: ops.Project, target_field: Optional[str]) -> LineageNode:  # noqa: D401
    parent_op = node.parent
    return LineageNode(op=node, edge=None, children=(
        LineageNode(op=parent_op, edge=Edge(EdgeKind.PARENT), children=(
            _build(parent_op, target_field),
        )),
    ))


@_build.register
def _window(node: ops.WindowFunction, target_field: Optional[str]):  # noqa: D401
    kids: Tuple[LineageNode, ...] = ()

    def _add(kind: EdgeKind, obj: Any):
        nonlocal kids
        n = _to_node(obj)
        kids += (LineageNode(op=n, edge=Edge(kind), children=(
            _build(n, None),
        )),)

    _add(EdgeKind.EXPR, node.func)
    for g in getattr(node, "group_by", ()) or ():
        _add(EdgeKind.GROUP_BY, g)
    for o in getattr(node, "order_by", ()) or ():
        _add(EdgeKind.ORDER_BY, o)
    for attr in ("start", "end"):
        if (b := getattr(node, attr, None)) is not None:
            _add(EdgeKind.EXPR, b)

    return LineageNode(op=node, edge=None, children=kids)


def build_lineage_tree(node: Node, *, target_field: Optional[str] = None) -> LineageNode:
    """Build lineage tree for a *single* expression/field."""
    return _build(node, target_field)


def build_column_lineage(expr: Any) -> Dict[str, LineageNode]:
    """Return mapping column‑name → lineage tree."""
    op = expr.op()
    cols: Dict[str, Any] = getattr(op, "values", getattr(op, "fields", {}))
    return {
        name: build_lineage_tree(_to_node(col), target_field=name)
        for name, col in cols.items()
    }


_DefaultBlacklist: Tuple[str, ...] = (
    "RemoteTable",
    "DatabaseTable",
    "Cast(",
    "Project",
)


def flatten_lineage(
    lineage: Dict[str, LineageNode],
    *,
    blacklist: Tuple[str, ...] = _DefaultBlacklist,
) -> Dict[str, Dict[str, Tuple[str, ...]]]:
    """Collapse deep trees into `{col: {inputs, steps}}` for fast assertions."""

    def _walk(node: LineageNode, steps: Set[str], inputs: Set[str]):
        label = _label(node.op)
        if label.endswith("(Field)"):
            inputs.add(label.split()[0])
        elif not any(b in label for b in blacklist):
            steps.add(label)
        if node.edge is None or node.edge.kind in {
            EdgeKind.UDF,
            EdgeKind.UDF_INPUT,
            EdgeKind.RELATION,
            EdgeKind.EXPR,
            EdgeKind.METRIC,
            EdgeKind.PARENT,
        }:
            for child in node.children:
                _walk(child, steps, inputs)

    out: Dict[str, Dict[str, Tuple[str, ...]]] = {}
    for col, tree in lineage.items():
        s, i = set(), set()
        _walk(tree, s, i)
        out[col] = {"inputs": tuple(sorted(i)), "steps": tuple(sorted(s))}
    return out


def print_lineage_ascii(node: LineageNode, *, indent: str = "", is_last: bool = True) -> None:  # noqa: D401
    connector = "└─" if is_last else "├─"
    edge_str = f" [{node.edge}]" if node.edge else ""
    print(f"{indent}{connector} {_label(node.op)}{edge_str}")
    new_indent = indent + ("   " if is_last else "│  ")
    for idx, child in enumerate(node.children):
        print_lineage_ascii(child, indent=new_indent, is_last=idx == len(node.children) - 1)
