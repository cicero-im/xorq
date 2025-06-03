from typing import List, Iterable, Optional, Tuple, Any

import xorq.expr.relations as rel
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
from xorq.vendor.ibis.expr.operations.core import Node

def _filter_none(values: Iterable[Optional[Node]]) -> Tuple[Node, ...]:
    return tuple(v for v in values if v is not None)


def to_node(maybe_expr: Any) -> Node:
    while not isinstance(maybe_expr, Node):
        op_fn = getattr(maybe_expr, "op", None)
        if op_fn is None:
            raise TypeError(f"Cannot convert {type(maybe_expr).__name__} into an Ibis Node")
        maybe_expr = op_fn()
    return maybe_expr


def children_of(node: Node) -> Tuple[Node, ...]:
    def _as_node(value: Any) -> Optional[Node]:
        try:
            return to_node(value)
        except (TypeError, AttributeError):
            return None

    match node:
        case ops.Field():
            rel_node = node.rel
            if rel_node is None:
                return tuple()
            # If the relation is a Project, follow the *specific* expression that
            # produced this field instead of the whole Project.
            if isinstance(rel_node, ops.Project):
                _, mapping = rel_node.args
                expr = mapping.get(node.name)
                return _filter_none((_as_node(expr),))
            return _filter_none((_as_node(rel_node),))

        case rel.RemoteTable():
            return _filter_none((_as_node(node.remote_expr),))
        case rel.CachedNode():
            return (to_node(node.parent),)
        case rel.FlightExpr():
            return (to_node(node.input_expr),)
        case rel.FlightUDXF():
            return (to_node(node.input_expr),)

        case udf.ExprScalarUDF():
            exprs = node.computed_kwargs_expr
            if isinstance(exprs, Node):
                return (exprs,)
            if exprs is not None:
                return _filter_none(map(_as_node, exprs))
            return tuple()

        case rel.Read():
            return tuple()  # leaf

        case _:
            raw_children = getattr(node, "__children__", ())
            return _filter_none(map(_as_node, raw_children))

opaque_ops = (
    rel.Read,
    rel.CachedNode,
    rel.RemoteTable,
    rel.FlightUDXF,
    rel.FlightExpr,
    udf.ExprScalarUDF,
)


def walk_nodes(node_types, expr):
    def process_node(op):
        match op:
            case rel.RemoteTable():
                if isinstance(op, node_types):
                    yield op
                yield from walk_nodes(
                    node_types,
                    op.remote_expr,
                )
            case rel.CachedNode():
                if isinstance(op, node_types):
                    yield op
                yield from walk_nodes(
                    node_types,
                    op.parent,
                )
            case rel.FlightExpr():
                if isinstance(op, node_types):
                    yield op
                yield from walk_nodes(node_types, op.input_expr)
            case rel.FlightUDXF():
                if isinstance(op, node_types):
                    yield op
                yield from walk_nodes(node_types, op.input_expr)
            case udf.ExprScalarUDF():
                if isinstance(op, node_types):
                    yield op
                yield from walk_nodes(
                    node_types,
                    op.computed_kwargs_expr,
                )
            case rel.Read():
                if isinstance(op, node_types):
                    yield op
            case _:
                if isinstance(op, opaque_ops):
                    raise ValueError(f"unhandled opaque op {type(op)}")
                yield from op.find(opaque_ops + tuple(node_types))

    def inner(rest, seen):
        if not rest:
            return seen
        op = rest.pop()
        seen.add(op)
        new = process_node(op)
        rest.update(set(new).difference(seen))
        return inner(rest, seen)

    initial_op = expr.op() if hasattr(expr, "op") else expr
    rest = process_node(initial_op)
    nodes = inner(set(rest), set())
    return tuple(node for node in nodes if isinstance(node, node_types))


def find_all_sources(expr):
    import xorq.vendor.ibis.expr.operations as ops

    node_types = (
        ops.DatabaseTable,
        ops.SQLQueryResult,
        rel.CachedNode,
        rel.Read,
        rel.RemoteTable,
        rel.FlightUDXF,
        rel.FlightExpr,
        # ExprScalarUDF has an expr we need to get to
        # FlightOperator has a dynamically generated connection: it should be passed a Profile instead
    )
    nodes = walk_nodes(node_types, expr)
    sources = tuple(
        source
        for (source, _) in set((node.source, node.source._profile) for node in nodes)
    )
    return sources
