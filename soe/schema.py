"""Normalize a Pydantic model (or JSON Schema dict) into a small node tree.

The constrained decoder and the repair/coercion layer both walk this tree, so
they agree on exactly which subset of JSON Schema is supported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


class UnsupportedSchema(ValueError):
    pass


@dataclass
class Node:
    kind: str  # object | array | string | integer | number | boolean | enum
    properties: list[tuple[str, "Node"]] = field(default_factory=list)
    required: set[str] = field(default_factory=set)
    items: "Node | None" = None
    enum: list[Any] = field(default_factory=list)
    min_items: int = 0
    max_items: int | None = None
    max_length: int | None = None
    minimum: float | None = None
    maximum: float | None = None


def _resolve(s: dict, defs: dict) -> dict:
    while "$ref" in s:
        name = s["$ref"].split("/")[-1]
        s = defs[name]
    if "allOf" in s and len(s["allOf"]) == 1:
        return _resolve(s["allOf"][0], defs)
    return s


def _build(s: dict, defs: dict) -> Node:
    s = _resolve(s, defs)
    if "enum" in s:
        return Node("enum", enum=list(s["enum"]))
    if "const" in s:
        return Node("enum", enum=[s["const"]])
    if "anyOf" in s or "oneOf" in s:
        raise UnsupportedSchema("anyOf/oneOf (incl. Optional) is not supported by the decoder")
    t = s.get("type")
    if t == "object":
        props = s.get("properties", {})
        return Node(
            "object",
            properties=[(k, _build(v, defs)) for k, v in props.items()],
            required=set(s.get("required", [])),
        )
    if t == "array":
        if "items" not in s:
            raise UnsupportedSchema("array without items")
        return Node(
            "array",
            items=_build(s["items"], defs),
            min_items=s.get("minItems", 0),
            max_items=s.get("maxItems"),
        )
    if t == "string":
        return Node("string", max_length=s.get("maxLength"))
    if t in ("integer", "number"):
        lo = s.get("minimum", s.get("exclusiveMinimum"))
        hi = s.get("maximum", s.get("exclusiveMaximum"))
        return Node(t, minimum=lo, maximum=hi)
    if t == "boolean":
        return Node("boolean")
    raise UnsupportedSchema(f"unsupported schema fragment: {s}")


def json_schema_of(model: type[BaseModel] | dict) -> dict:
    if isinstance(model, dict):
        return model
    return model.model_json_schema()


def to_node(model: type[BaseModel] | dict) -> Node:
    s = json_schema_of(model)
    return _build(s, s.get("$defs", {}))
