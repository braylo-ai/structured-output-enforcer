"""Constrained decoding: output must validate for ANY logits.

A random-logits fake model stress-tests validity-by-construction; a tiny real
model (SmolLM2-135M-Instruct, cached locally) checks the KV-cache path.
"""

from types import SimpleNamespace
from typing import Literal

import pytest
import torch
from pydantic import BaseModel, Field

from eval.schemas import SCHEMAS
from soe.constrained import ConstrainedJSONDecoder, VocabIndex

TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"
tf = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tok():
    try:
        return tf.AutoTokenizer.from_pretrained(TINY)
    except Exception:  # pragma: no cover
        pytest.skip("tiny model not cached")


@pytest.fixture(scope="module")
def vocab(tok):
    return VocabIndex(tok)


class RandomLM:
    """Stands in for a causal LM: returns random logits, keeps no cache."""

    def __init__(self, vocab_size, seed):
        self.g = torch.Generator().manual_seed(seed)
        self.n = vocab_size
        self.device = "cpu"

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        return SimpleNamespace(
            logits=torch.randn(1, input_ids.shape[1], self.n, generator=self.g),
            past_key_values=None,
        )


@pytest.mark.parametrize("schema_name", list(SCHEMAS))
@pytest.mark.parametrize("seed", range(4))
def test_random_logits_always_valid(tok, vocab, schema_name, seed):
    model = SCHEMAS[schema_name]
    dec = ConstrainedJSONDecoder(
        RandomLM(len(tok), seed), tok, max_string_tokens=12, vocab=vocab
    )
    text, stats = dec.generate(tok.encode("x"), model)
    model.model_validate_json(text)  # raises if invalid
    assert stats.model_tokens > 0 and stats.forced_tokens > 0


def test_bounds_and_lengths_enforced(tok, vocab):
    class M(BaseModel):
        n: int = Field(ge=3, le=7)
        s: str = Field(max_length=5)
        tags: list[Literal["a", "bb", "ccc"]] = Field(min_length=2, max_length=3)
        x: float

    for seed in range(6):
        dec = ConstrainedJSONDecoder(RandomLM(len(tok), seed), tok, vocab=vocab)
        text, _ = dec.generate(tok.encode("x"), M)
        v = M.model_validate_json(text)
        assert 3 <= v.n <= 7 and len(v.s) <= 5 and 2 <= len(v.tags) <= 3


def test_real_tiny_model_with_kv_cache(tok, vocab):
    model = tf.AutoModelForCausalLM.from_pretrained(TINY).eval()
    dec = ConstrainedJSONDecoder(model, tok, vocab=vocab)
    msgs = [
        {
            "role": "user",
            "content": "Review of the Nimbus lamp: 4/5, bright, cheap. Recommend.",
        }
    ]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
    text, stats = dec.generate(list(ids), SCHEMAS["ProductReview"])
    SCHEMAS["ProductReview"].model_validate_json(text)
    assert stats.forward_passes >= 1
