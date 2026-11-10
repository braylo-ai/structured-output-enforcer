"""Schema-valid rate with vs without enforcement on data/testset.jsonl.

Conditions (same model, same prompt, greedy decoding):
  none          raw model output must json.loads + validate as-is
  repair        + extract / syntax repair / coercion (no extra model calls)
  repair+retry  + up to 2 re-prompts with the validation errors
  constrained   schema-constrained decoding (local model only)

Writes results/<tag>/rows.jsonl and results/<tag>/summary.json.
Usage: python -m eval.run_eval --model Qwen/Qwen2.5-0.5B-Instruct [--limit N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from soe.enforcer import RETRY, build_messages
from soe.hf_backend import HFBackend
from soe.repair import repair

from .schemas import SCHEMAS

ROOT = Path(__file__).resolve().parents[1]


def leaves(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(x, list):
        yield prefix + ".__len__", len(x)
        for i, v in enumerate(x):
            yield from leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def norm(v):
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def field_accuracy(pred: dict | None, gold: dict) -> float:
    g = dict(leaves(gold))
    if pred is None:
        return 0.0
    p = dict(leaves(pred))
    return sum(1 for k, v in g.items() if k in p and norm(p[k]) == norm(v)) / len(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--prompt", choices=["schema", "minimal"], default="schema",
                    help="schema: JSON Schema in the system prompt; minimal: only field names")
    args = ap.parse_args()
    tag = args.model.split("/")[-1] + ("" if args.prompt == "schema" else "-minimal-prompt")
    outdir = ROOT / "results" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(l)
        for l in (ROOT / "data" / "testset.jsonl").read_text().splitlines()
        if l.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    be = HFBackend(args.model)
    out_rows = []
    t_start = time.time()
    for i, row in enumerate(rows):
        model = SCHEMAS[row["schema"]]
        msgs = build_messages(model, row["text"], args.prompt)
        rec = {"id": row["id"], "schema": row["schema"]}
        # unconstrained
        t0 = time.perf_counter()
        raw = be.generate_text(msgs)
        rec["sec_unconstrained"] = round(time.perf_counter() - t0, 3)
        r = repair(raw, model)
        rec["raw"] = raw
        rec["none_valid"] = r.ok and r.stage == "raw"
        rec["repair_valid"] = r.ok
        rec["repair_stage"] = r.stage
        final, attempts, cur_raw, cur = r, 1, raw, r
        m2 = msgs
        while not cur.ok and attempts <= args.retries:
            m2 = m2 + [
                {"role": "assistant", "content": cur_raw},
                {
                    "role": "user",
                    "content": RETRY.format(errors="\n".join(cur.errors[:8])),
                },
            ]
            cur_raw = be.generate_text(m2)
            cur = repair(cur_raw, model)
            attempts += 1
        rec["retry_valid"] = cur.ok
        rec["retry_attempts"] = attempts
        rec["errors_after_retry"] = cur.errors[:5]
        rec["acc_none"] = field_accuracy(
            json.loads(raw) if rec["none_valid"] else None, row["gold"]
        )
        rec["acc_repair"] = field_accuracy(
            r.value.model_dump() if r.ok else None, row["gold"]
        )
        rec["acc_retry"] = field_accuracy(
            cur.value.model_dump() if cur.ok else None, row["gold"]
        )
        # constrained
        t0 = time.perf_counter()
        craw = be.generate_constrained(msgs, model)
        rec["sec_constrained"] = round(time.perf_counter() - t0, 3)
        cr = repair(craw, model)
        rec["constrained_raw"] = craw
        rec["constrained_valid"] = cr.ok and cr.stage == "raw"
        rec["constrained_errors"] = cr.errors[:5]
        rec["acc_constrained"] = field_accuracy(
            cr.value.model_dump() if cr.ok else None, row["gold"]
        )
        out_rows.append(rec)
        print(
            f"[{i + 1}/{len(rows)}] {row['id']} none={rec['none_valid']} repair={rec['repair_valid']} "
            f"retry={rec['retry_valid']} constrained={rec['constrained_valid']}",
            flush=True,
        )
    n = len(out_rows)
    summ = {
        "model": args.model, "prompt": args.prompt,
        "n": n,
        "decoding": "greedy",
        "device": "cpu",
        "wall_seconds": round(time.time() - t_start, 1),
    }
    for c in ("none", "repair", "retry", "constrained"):
        summ[f"valid_{c}"] = sum(r[f"{c}_valid"] for r in out_rows)
        summ[f"valid_rate_{c}"] = round(summ[f"valid_{c}"] / n, 4)
        summ[f"field_acc_{c}"] = round(sum(r[f"acc_{c}"] for r in out_rows) / n, 4)
    summ["repair_stage_counts"] = {}
    for r in out_rows:
        summ["repair_stage_counts"][r["repair_stage"]] = (
            summ["repair_stage_counts"].get(r["repair_stage"], 0) + 1
        )
    summ["per_schema"] = {}
    for s in SCHEMAS:
        rs = [r for r in out_rows if r["schema"] == s]
        if rs:
            summ["per_schema"][s] = {
                c: f"{sum(r[f'{c}_valid'] for r in rs)}/{len(rs)}"
                for c in ("none", "repair", "retry", "constrained")
            }
    summ["mean_sec_unconstrained"] = round(
        sum(r["sec_unconstrained"] for r in out_rows) / n, 3
    )
    summ["mean_sec_constrained"] = round(
        sum(r["sec_constrained"] for r in out_rows) / n, 3
    )
    (outdir / "rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in out_rows) + "\n"
    )
    (outdir / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
