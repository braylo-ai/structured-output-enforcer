"""Enforcer: one call site, two strategies.

* Local HF model (logits available)  -> constrained decoding, valid by construction.
* Any text-in/text-out model (API)   -> generate, validate-and-repair, then re-prompt
                                         with the validation errors up to `max_retries`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel

from .repair import repair
from .schema import json_schema_of

SYSTEM = (
    "You extract structured data. Reply with a single JSON object that matches this JSON Schema. "
    "No prose, no markdown.\nSchema:\n{schema}"
)
RETRY = (
    "Your previous reply was not valid for the schema.\nErrors:\n{errors}\n"
    "Reply again with only the corrected JSON object."
)


MINIMAL = "Extract the data from the user's text as a JSON object with the fields: {fields}."


def build_messages(model: type[BaseModel], text: str, prompt: str = "schema") -> list[dict]:
    if prompt == "minimal":
        fields = ", ".join(json_schema_of(model).get("properties", {}))
        return [{"role": "system", "content": MINIMAL.format(fields=fields)}, {"role": "user", "content": text}]
    schema = json.dumps(json_schema_of(model), separators=(",", ":"))
    return [
        {"role": "system", "content": SYSTEM.format(schema=schema)},
        {"role": "user", "content": text},
    ]


@dataclass
class EnforceResult:
    ok: bool
    value: BaseModel | None
    raw: str
    strategy: str
    stage: str  # constrained | raw | extract | syntax | coerce | retry-N | fail
    attempts: int = 1
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0


class Enforcer:
    def __init__(
        self,
        generate_text: Callable[[list[dict]], str] | None = None,
        constrained=None,
        max_retries: int = 2,
    ):
        """generate_text: messages -> text (any API/local model).
        constrained: object with .generate_constrained(messages, model) -> text (optional).
        """
        if generate_text is None and constrained is None:
            raise ValueError("need a text generator or a constrained backend")
        self.generate_text = generate_text
        self.constrained = constrained
        self.max_retries = max_retries

    def run(
        self, model: type[BaseModel], text: str, strategy: str = "auto"
    ) -> EnforceResult:
        t0 = time.perf_counter()
        msgs = build_messages(model, text)
        if strategy == "auto":
            strategy = "constrained" if self.constrained is not None else "repair"
        if strategy == "constrained":
            raw = self.constrained.generate_constrained(msgs, model)
            r = repair(
                raw, model
            )  # validation only: constrained output should pass at stage "raw"
            stage = "constrained" if r.ok and r.stage == "raw" else r.stage
            return EnforceResult(
                r.ok,
                r.value,
                raw,
                strategy,
                stage,
                1,
                r.errors,
                time.perf_counter() - t0,
            )
        if strategy == "none":
            raw = self.generate_text(msgs)
            r = repair(raw, model)
            ok = r.ok and r.stage == "raw"
            return EnforceResult(
                ok,
                r.value if ok else None,
                raw,
                strategy,
                "raw" if ok else "fail",
                1,
                r.errors,
                time.perf_counter() - t0,
            )
        # repair (+ retries)
        raw = self.generate_text(msgs)
        r = repair(raw, model)
        attempts = 1
        while not r.ok and attempts <= self.max_retries:
            msgs = msgs + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": RETRY.format(errors="\n".join(r.errors[:8])),
                },
            ]
            raw = self.generate_text(msgs)
            r = repair(raw, model)
            attempts += 1
            if r.ok:
                return EnforceResult(
                    True,
                    r.value,
                    raw,
                    strategy,
                    f"retry-{attempts - 1}",
                    attempts,
                    [],
                    time.perf_counter() - t0,
                )
        return EnforceResult(
            r.ok,
            r.value,
            raw,
            strategy,
            r.stage,
            attempts,
            r.errors,
            time.perf_counter() - t0,
        )
