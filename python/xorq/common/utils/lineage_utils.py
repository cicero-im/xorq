from __future__ import annotations


import functools
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from attrs import field, frozen
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


@frozen
class Edge:
    kind: EdgeKind = field(validator=instance_of(EdgeKind))
    detail: Optional[str] = field(default=None, validator=optional(instance_of(str)))

    def __str__(self) -> str:
        suffix = f":{self.detail}" if self.detail is not None else ""
        return f"{self.kind.name.lower()}{suffix}"


def _label(node: Node) -> str:
    nm = getattr(node, "name", None)
    return f"{nm} ({type(node).__name__})" if nm else type(node).__name__


def _to_node(maybe_expr: Any) -> Node:
    while not isinstance(maybe_expr, Node):
        if hasattr(maybe_expr, "op"):
            maybe_expr = maybe_expr.op()
        else:
            raise TypeError(f"Cannot turn {type(maybe_expr)} into an ibis Node")
    return maybe_expr


@frozen
class LineageNode:
    op: Node = field(validator=instance_of(Node))
    edge: Optional[Edge] = field(validator=optional(instance_of(Edge)), default=None)
    children: Tuple["LineageNode", ...] = field(
        validator=instance_of(tuple), factory=tuple
    )


@functools.singledispatch
def _build(node: Node, target_field: Optional[str]) -> LineageNode:
    if isinstance(node, ScalarUDF):
        udf_children: List[LineageNode] = []
        for arg in node.args:
            if not isinstance(arg, Node):
                continue
            arg_subtree = _build(arg, target_field)
            udf_children.append(
                LineageNode(
                    op=arg_subtree.op,
                    edge=Edge(EdgeKind.UDF_INPUT),
                    children=arg_subtree.children,
                )
            )
        return LineageNode(op=node, edge=Edge(EdgeKind.UDF), children=tuple(udf_children))

    raw_children = get_children(node)
    children = tuple(
        _build(child, target_field) for child in raw_children if isinstance(child, Node)
    )
    return LineageNode(op=node, edge=None, children=children)

@_build.register
def _(node: ops.Field, target_field: Optional[str]) -> LineageNode:  # noqa: D401
    base_children: List[LineageNode] = []
    if node.rel is not None:
        rel_node = node.rel

        if isinstance(rel_node, (ops.Project,)) :
            _, mapping = rel_node.args
            raw_expr = mapping.get(node.name)

            expr_node = _to_node(raw_expr)

            edge_kind = EdgeKind.UDF if isinstance(expr_node, ScalarUDF) else EdgeKind.EXPR
            base_children.append(
                LineageNode(
                    op=expr_node,
                    edge=Edge(edge_kind),
                    children=(_build(expr_node, target_field),),
                )
            )
        else:
            base_children.append(
            LineageNode(
                op=rel_node,
                edge=Edge(EdgeKind.RELATION),
                children=(_build(rel_node, target_field),),
            )
        )
    return LineageNode(op=node, edge=None, children=tuple(base_children))


@_build.register
def _(node: ops.Aggregate, target_field: Optional[str]) -> LineageNode:
    metric_children = tuple(
        LineageNode(
            op=metric_expr,
            edge=Edge(EdgeKind.METRIC, metric_name),
            children=(_build(metric_expr, target_field),),
        )
        for metric_name, metric_expr in node.metrics.items()
        if target_field is None or metric_name == target_field
    )
    parent_child = LineageNode(
        op=node.parent,
        edge=Edge(EdgeKind.PARENT),
        children=(_build(node.parent, None),),
    )
    return LineageNode(op=node, edge=None, children=metric_children + (parent_child,))


@_build.register
def _(node: rel.RemoteTable, target_field: Optional[str]) -> LineageNode:
    remote_op = node.remote_expr.op()
    child_remote = LineageNode(
        op=remote_op,
        edge=Edge(EdgeKind.REMOTE_EXPR),
        children=(_build(remote_op, target_field),),
    )
    return LineageNode(op=node, edge=None, children=(child_remote,))


@_build.register
def _(node: ops.Project, target_field: Optional[str]) -> LineageNode:
    parent_op = node.parent
    return LineageNode(
        op=node,
        edge=None,
        children=(
            LineageNode(
                op=parent_op,
                edge=Edge(EdgeKind.PARENT),
                children=(_build(parent_op, target_field),),
            ),
        ),
    )


def build_lineage_tree(node: Node, target_field: Optional[str] = None) -> LineageNode:
    return _build(node, target_field)


def build_column_lineage_dict(expr: Any) -> Dict[str, LineageNode]:
    op = expr.op()
    columns = getattr(op, "values", getattr(op, "fields", {}))
    return {name: build_lineage_tree(_to_node(col), name) for name, col in columns.items()}


def flatten_lineage(
    lineage: Dict[str, LineageNode],
    blacklist: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    if blacklist is None:
        blacklist = (
            "RemoteTable",
            "DatabaseTable",
            "Cast(",
            "Project",
        )

    def walk(node: LineageNode, steps: Set[str], inputs: Set[str]) -> None:
        label = _label(node.op)
        if label.endswith("(Field)"):
            field_name, *_ = label.split()
            inputs.add(field_name)
        elif not any(bad in label for bad in blacklist):
            steps.add(label)

        for child in node.children:
            edge = child.edge
            if (
                edge is None
                or edge.kind
                in {
                    EdgeKind.UDF,
                    EdgeKind.UDF_INPUT,
                    EdgeKind.RELATION,
                    EdgeKind.EXPR,
                    EdgeKind.METRIC,
                    EdgeKind.PARENT,
                }
            ):
                walk(child, steps, inputs)

    flat: Dict[str, Any] = {}
    for col, tree in lineage.items():
        steps_set: Set[str] = set()
        inputs_set: Set[str] = set()
        walk(tree, steps_set, inputs_set)
        flat[col] = {
            "inputs": tuple(sorted(inputs_set)),
            "steps": tuple(sorted(steps_set)),
        }
    return flat


def print_lineage_ascii(
    lineage_node: LineageNode,
    indent: str = "",
    is_last: bool = True,
) -> None:
    connector = "└─" if is_last else "├─"
    edge_str = f" [{lineage_node.edge}]" if lineage_node.edge else ""
    print(f"{indent}{connector} {_label(lineage_node.op)}{edge_str}")

    new_indent = indent + ("   " if is_last else "│  ")
    for idx, child in enumerate(lineage_node.children):
        print_lineage_ascii(child, new_indent, idx == len(lineage_node.children) - 1)
