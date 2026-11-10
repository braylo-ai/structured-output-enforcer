"""Validate-and-repair fallback for models whose logits we cannot touch (API models).

Pipeline, each stage cheap and deterministic:
  1. extract   - pull the JSON object out of prose / markdown fences
  2. syntax    - fix common JSON syntax slips (trailing commas, single quotes,
                 Python literals, unquoted keys, unclosed brackets, comments)
  3. coerce    - bend values toward the schema ("5" -> 5, "Yes" -> true,
                 scalar -> [scalar], enum case-folding, drop unknown keys)
  4. validate  - Pydantic is the final judge
If all fail, the Enforcer re-prompts the model with the validation errors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from .schema import Node, to_node

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def extract_json_text(text: str) -> str | None:
    """Return the most plausible JSON object substring, or None."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    quote = None
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == quote:
                in_str = False
            continue
        if c in ('"', "'"):
            in_str, quote = True, c
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]  # unbalanced: let syntax repair close it


def _tokenize_outside_strings(s: str):
    """Yield (is_string, chunk) segments; normalizes single-quoted strings to double."""
    i, n = 0, len(s)
    buf = []
    while i < n:
        c = s[i]
        if c in ('"', "'"):
            q = c
            j = i + 1
            out = []
            while j < n:
                d = s[j]
                if d == "\\" and j + 1 < n:
                    out.append(s[j : j + 2])
                    j += 2
                    continue
                if d == q:
                    break
                if d == '"' and q == "'":
                    out.append('\\"')
                else:
                    out.append(d)
                j += 1
            if buf:
                yield False, "".join(buf)
                buf = []
            yield True, '"' + "".join(out) + '"'  # unclosed strings get closed here
            i = j + 1
        else:
            buf.append(c)
            i += 1
    if buf:
        yield False, "".join(buf)


def repair_syntax(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    parts = []
    for is_str, chunk in _tokenize_outside_strings(s):
        if not is_str:
            chunk = re.sub(r"//[^\n]*", "", chunk)
            chunk = re.sub(r"/\*.*?\*/", "", chunk, flags=re.S)
            chunk = re.sub(r"\bTrue\b", "true", chunk)
            chunk = re.sub(r"\bFalse\b", "false", chunk)
            chunk = re.sub(r"\bNone\b", "null", chunk)
            # unquoted keys:  {foo: 1}  ->  {"foo": 1}
            chunk = re.sub(
                r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', chunk
            )
        parts.append((is_str, chunk))
    out = "".join(c for _, c in parts)
    # trailing commas (outside strings is good enough after normalization)
    out = re.sub(r",(\s*[}\]])", r"\1", out)
    # close unbalanced brackets
    stack = []
    for is_str, chunk in parts:
        if is_str:
            continue
        for c in chunk:
            if c in "{[":
                stack.append("}" if c == "{" else "]")
            elif c in "}]" and stack:
                stack.pop()
    out = out.rstrip().rstrip(",")
    return out + "".join(reversed(stack))


_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


def coerce(value: Any, node: Node) -> Any:
    k = node.kind
    if k == "object":
        if not isinstance(value, dict):
            return value
        keys = {key.lower(): key for key in value}
        out = {}
        for name, sub in node.properties:
            src = name if name in value else keys.get(name.lower())
            if src is not None:
                out[name] = coerce(value[src], sub)
        return out
    if k == "array":
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        items = [coerce(v, node.items) for v in value]
        if node.max_items is not None:
            items = items[: node.max_items]
        return items
    if k == "string":
        if isinstance(value, (int, float, bool)):
            value = json.dumps(value) if isinstance(value, bool) else str(value)
        if isinstance(value, str) and node.max_length is not None:
            value = value[: node.max_length]
        return value
    if k in ("integer", "number"):
        if isinstance(value, str):
            cleaned = re.sub(r"[^0-9.\-]", "", value)
            try:
                value = float(cleaned)
            except ValueError:
                return value
        if isinstance(value, float) and k == "integer" and value.is_integer():
            value = int(value)
        return value
    if k == "boolean":
        if isinstance(value, str):
            low = value.strip().lower()
            if low in _TRUE:
                return True
            if low in _FALSE:
                return False
        return value
    if k == "enum":
        if isinstance(value, str):
            for opt in node.enum:
                if isinstance(opt, str) and opt.lower() == value.strip().lower():
                    return opt
        return value
    return value


@dataclass
class RepairResult:
    ok: bool
    value: BaseModel | None
    stage: (
        str  # which stage first produced a valid object: raw|extract|syntax|coerce|fail
    )
    errors: list[str] = field(default_factory=list)
    data: Any = None


def _try_validate(model: type[BaseModel], data: Any):
    try:
        return model.model_validate(data), None
    except ValidationError as e:
        return None, [
            f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
        ]


def repair(text: str, model: type[BaseModel]) -> RepairResult:
    node = to_node(model)
    errors: list[str] = []
    # stage 0: raw
    try:
        data = json.loads(text)
        v, errs = _try_validate(model, data)
        if v:
            return RepairResult(True, v, "raw", data=data)
        errors = errs
    except json.JSONDecodeError as e:
        errors = [f"json: {e.msg}"]
    # stage 1: extract
    sub = extract_json_text(text)
    if sub is None:
        return RepairResult(False, None, "fail", ["no JSON object found in output"])
    data = None
    try:
        data = json.loads(sub)
        v, errs = _try_validate(model, data)
        if v:
            return RepairResult(True, v, "extract", data=data)
        errors = errs
    except json.JSONDecodeError:
        pass
    # stage 2: syntax
    if data is None:
        try:
            data = json.loads(repair_syntax(sub))
            v, errs = _try_validate(model, data)
            if v:
                return RepairResult(True, v, "syntax", data=data)
            errors = errs
        except json.JSONDecodeError as e:
            return RepairResult(False, None, "fail", [f"json after repair: {e.msg}"])
    # stage 3: coerce
    data = coerce(data, node)
    v, errs = _try_validate(model, data)
    if v:
        return RepairResult(True, v, "coerce", data=data)
    return RepairResult(False, None, "fail", errs or errors, data=data)
