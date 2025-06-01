import functools
from typing import Any, Dict, Optional, Set, Tuple

from attrs import field, frozen
from attrs.validators import instance_of, optional

import xorq.expr.relations as rel
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.graph_utils import get_children
from xorq.expr.udf import ScalarUDF
from xorq.vendor.ibis.expr.operations.core import Node


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
    edge: Optional[str] = field(validator=optional(instance_of(str)), default=None)
    # How do we validate that the children contains instances of self?
    children: Tuple["LineageNode", ...] = field(
        validator=instance_of(tuple), factory=tuple
    )


@functools.singledispatch
def _build(node: Node, target_field: Optional[str]) -> LineageNode:
    if isinstance(node, ScalarUDF):
        children = tuple(
            LineageNode(op=arg, edge="udf_input", children=())
            for arg in node.args
            if isinstance(arg, Node)
        )
        return LineageNode(op=node, edge="udf", children=children)

    raw_children = get_children(node)
    children = tuple(
        _build(child, target_field) for child in raw_children if isinstance(child, Node)
    )
    return LineageNode(op=node, edge=None, children=children)


@_build.register
def _(node: ops.Field, target_field: Optional[str]) -> LineageNode:
    base_children = []
    if node.rel is not None:
        rel_node = node.rel
        child_relation = LineageNode(
            op=rel_node,
            edge="relation",
            children=(_build(rel_node, target_field),),
        )
        base_children.append(child_relation)

        other_cls = None
        if isinstance(rel_node, (ops.Project,) + ((other_cls,) if other_cls else ())):
            _, mapping = rel_node.args
            raw_expr = mapping.get(node.name)
            try:
                expr_node = _to_node(raw_expr)
            except TypeError:
                return LineageNode(op=node, edge=None, children=tuple(base_children))

            edge_label = "udf" if isinstance(expr_node, ScalarUDF) else "expr"
            child_expr = LineageNode(
                op=expr_node,
                edge=edge_label,
                children=(_build(expr_node, target_field),),
            )
            base_children.append(child_expr)

    return LineageNode(op=node, edge=None, children=tuple(base_children))


@_build.register
def _(node: ops.Aggregate, target_field: Optional[str]) -> LineageNode:
    metric_children = tuple(
        LineageNode(
            op=metric_expr,
            edge=f"metric:{metric_name}",
            children=(_build(metric_expr, target_field),),
        )
        for metric_name, metric_expr in node.metrics.items()
        if target_field is None or metric_name == target_field
    )
    parent_child = LineageNode(
        op=node.parent,
        edge="parent",
        children=(_build(node.parent, None),),
    )
    return LineageNode(op=node, edge=None, children=metric_children + (parent_child,))


@_build.register
def _(node: rel.RemoteTable, target_field: Optional[str]) -> LineageNode:
    remote_op = node.remote_expr.op()
    child_remote = LineageNode(
        op=remote_op,
        edge="remote_expr",
        children=(_build(remote_op, target_field),),
    )
    return LineageNode(op=node, edge=None, children=(child_remote,))


@_build.register
def _(node: ops.Project, target_field: Optional[str]) -> LineageNode:
    parent_op = node.parent
    child_parent = LineageNode(
        op=parent_op,
        edge="parent",
        children=(_build(parent_op, target_field),),
    )
    return LineageNode(op=node, edge=None, children=(child_parent,))


def build_lineage_tree(node: Node, target_field: Optional[str] = None) -> LineageNode:
    return _build(node, target_field)


def build_column_lineage_dict(expr: Any) -> Dict[str, LineageNode]:
    op = expr.op()
    columns = getattr(op, "values", getattr(op, "fields", {}))
    return {
        name: build_lineage_tree(_to_node(col), name) for name, col in columns.items()
    }


def flatten_lineage(
    lineage: Dict[str, LineageNode],
    blacklist: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    if blacklist is None:
        blacklist = ("RemoteTable", "DatabaseTable", "Cast(", "Project")

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
                or edge in {"udf", "udf_input", "relation", "expr", "metric", "parent"}
                or (edge and edge.startswith("metric:"))
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
