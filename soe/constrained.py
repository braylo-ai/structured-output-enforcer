"""Schema-constrained decoding for local Hugging Face causal LMs.

The decoder walks the schema tree and, at every step, either
  * forces structural text the schema fixes (braces, keys, colons, commas), or
  * lets the model pick a token, with logits masked to the tokens that keep
    the output inside the schema (string-body tokens, digits, enum tries, ...),
  * or "peeks": compares the model's best logit across groups of tokens
    (e.g. "another array item" vs "close the array") without consuming one.

Every emitted prefix is a valid prefix of a schema-valid JSON document, so the
final text always parses and matches the supported schema subset by
construction. Works on CPU; uses a KV cache so forced tokens cost one batched
forward pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from .schema import Node, to_node

_TERM_CHARS = ",}]"


@dataclass
class DecodeStats:
    model_tokens: int = 0  # tokens chosen by the model under a mask
    forced_tokens: int = 0  # tokens inserted by the schema
    forward_passes: int = 0


class VocabIndex:
    """Precomputed token-id groups over the tokenizer vocabulary."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        n = len(tokenizer)
        special = set(tokenizer.all_special_ids)
        self.text: list[str] = [""] * n
        body, end_ids, digits = [], [], []
        starts: dict[str, list[int]] = {c: [] for c in _TERM_CHARS + '"{.-'}
        for i in range(n):
            if i in special:
                continue
            t = tokenizer.decode([i])
            self.text[i] = t
            if not t or "�" in t:
                continue
            if t[0] in starts:
                starts[t[0]].append(i)
            if t.isdigit() and t.isascii():
                digits.append(i)
            if all(c not in '"\\' and ord(c) >= 0x20 for c in t):
                body.append(i)
            elif t.endswith('"') and all(
                c not in '"\\' and ord(c) >= 0x20 for c in t[:-1]
            ):
                end_ids.append(i)
        self.body = torch.tensor(body)
        self.string_end = torch.tensor(end_ids)
        self.digits = torch.tensor(digits)
        self.starts = {c: torch.tensor(v, dtype=torch.long) for c, v in starts.items()}
        self.term = torch.cat([self.starts[c] for c in _TERM_CHARS])
        self.body_len = torch.tensor([len(self.text[i]) for i in body])
        self.string_end_set = set(end_ids)
        self.end_len = torch.tensor([len(self.text[i]) - 1 for i in end_ids])
        self.minus = torch.tensor([i for i in range(n) if self.text[i] == "-"], dtype=torch.long)
        self.dot = torch.tensor([i for i in range(n) if self.text[i] == "."], dtype=torch.long)


class ConstrainedJSONDecoder:
    def __init__(
        self,
        model,
        tokenizer,
        max_string_tokens: int = 40,
        vocab: VocabIndex | None = None,
    ):
        self.model = model
        self.tok = tokenizer
        self.v = vocab or VocabIndex(tokenizer)
        self.max_string_tokens = max_string_tokens

    # ---------------- low-level state machine ----------------
    def _reset(self, prompt_ids: list[int]):
        self.cache = None
        self.pending = list(prompt_ids)
        self.out: list[str] = []  # exact emitted text chunks
        self.stats = DecodeStats()
        self._last_logits = None

    @torch.no_grad()
    def _logits(self) -> torch.Tensor:
        ids = torch.tensor([self.pending], device=self.model.device)
        res = self.model(input_ids=ids, past_key_values=self.cache, use_cache=True)
        self.cache = res.past_key_values
        self.pending = []
        self.stats.forward_passes += 1
        self._last_logits = res.logits[0, -1].float().cpu()
        return self._last_logits

    def _next_logits(self) -> torch.Tensor:
        return self._logits() if self.pending else self._last_logits

    def _force(self, text: str):
        ids = self.tok.encode(text, add_special_tokens=False)
        self.pending.extend(ids)
        self.out.append(text)
        self.stats.forced_tokens += len(ids)

    def _force_ids(self, ids: list[int]):
        self.pending.extend(ids)
        self.out.extend(self.v.text[i] for i in ids)
        self.stats.forced_tokens += len(ids)

    def _pick(self, allowed: torch.Tensor) -> int:
        logits = self._next_logits()
        tid = int(allowed[int(torch.argmax(logits[allowed]))])
        self.pending.append(tid)
        self.out.append(self.v.text[tid])
        self.stats.model_tokens += 1
        return tid

    def _peek(self, groups: dict[str, torch.Tensor]) -> str:
        """Return the group whose best token the model prefers; consumes nothing."""
        logits = self._next_logits()
        best, best_v = None, -float("inf")
        for name, ids in groups.items():
            if len(ids) == 0:
                continue
            val = float(logits[ids].max())
            if val > best_v:
                best, best_v = name, val
        return best

    def _trie_choose(self, options: list[str], stop: torch.Tensor | None = None) -> str:
        """Greedy constrained choice among fixed strings (enums, booleans, small int ranges).
        `stop` breaks ties when one option is a token-prefix of another (e.g. "1" vs "10")."""
        seqs = {o: self.tok.encode(o, add_special_tokens=False) for o in options}
        pos, alive = 0, list(options)
        while True:
            done = [o for o in alive if len(seqs[o]) == pos]
            cont = [o for o in alive if len(seqs[o]) > pos]
            if not cont:
                return done[0]
            if len(cont) == 1 and not done:
                self._force_ids(seqs[cont[0]][pos:])
                return cont[0]
            allowed = torch.tensor(sorted({seqs[o][pos] for o in cont}))
            if done and stop is not None:
                if self._peek({"stop": stop, "more": allowed}) == "stop":
                    return done[0]
            tid = self._pick(allowed)
            alive = [o for o in cont if seqs[o][pos] == tid]
            pos += 1

    # ---------------- schema walk ----------------
    def _value(self, node: Node):
        k = node.kind
        if k == "object":
            self._force("{")
            for i, (name, sub) in enumerate(node.properties):
                self._force(("" if i == 0 else ", ") + json.dumps(name) + ": ")
                self._value(sub)
            self._force("}")
        elif k == "array":
            self._force("[")
            n = 0
            while True:
                if node.max_items is not None and n >= node.max_items:
                    break
                if n >= node.min_items:
                    nxt = self._start_ids(node.items) if n == 0 else self.v.starts[","]
                    if self._peek({"end": self.v.starts["]"], "item": nxt}) == "end":
                        break
                if n > 0:
                    self._force(", ")
                self._value(node.items)
                n += 1
            self._force("]")
        elif k == "string":
            self._string(node)
        elif k == "enum":
            self._trie_choose([json.dumps(o) for o in node.enum])
        elif k == "boolean":
            self._trie_choose(["true", "false"])
        elif k in ("integer", "number"):
            self._number(node)
        else:  # pragma: no cover
            raise ValueError(k)

    def _start_ids(self, node: Node) -> torch.Tensor:
        if node.kind in ("string", "enum"):
            return self.v.starts['"']
        if node.kind == "object":
            return self.v.starts["{"]
        return torch.cat([self.v.digits, self.v.starts["-"], self.v.starts['"']])

    def _string(self, node: Node):
        self._force('"')
        chars, limit = 0, node.max_length
        for _ in range(self.max_string_tokens):
            body, end = self.v.body, self.v.string_end
            if limit is not None:
                remaining = limit - chars
                if remaining <= 0:
                    break
                body = body[self.v.body_len <= remaining]
                end = end[self.v.end_len <= remaining]
            allowed = torch.cat([body, end])
            # the model often wants a merged close token like `",` or `"}` - treat those as "close here"
            if self._peek({"close": self.v.starts['"'], "cont": allowed}) == "close":
                break
            tid = self._pick(allowed)
            if tid in self.v.string_end_set:
                return
            chars += len(self.v.text[tid])
        self._force('"')  # length/token budget reached: close the string

    def _number(self, node: Node):
        lo, hi = node.minimum, node.maximum
        if node.kind == "integer" and lo is not None and hi is not None and hi - lo <= 200:
            self._trie_choose([str(i) for i in range(int(lo), int(hi) + 1)], stop=self.v.term)
            return
        if lo is None or lo < 0:
            if self._peek({"neg": self.v.minus, "pos": self.v.digits}) == "neg":
                self._force("-")
        self._pick(self.v.digits)
        seen_dot = False
        for _ in range(15):
            groups = {"digit": self.v.digits, "term": self.v.term}
            if node.kind == "number" and not seen_dot:
                groups["dot"] = self.v.dot
            choice = self._peek(groups)
            if choice == "term":
                return
            if choice == "dot":
                self._force(".")
                seen_dot = True
            self._pick(self.v.digits)

    # ---------------- public API ----------------
    def generate(self, prompt_ids: list[int], schema) -> tuple[str, DecodeStats]:
        node = schema if isinstance(schema, Node) else to_node(schema)
        self._reset(prompt_ids)
        self._value(node)
        return "".join(self.out), self.stats
