"""Local Hugging Face backend: plain greedy generation and constrained generation."""

from __future__ import annotations

import torch

from .constrained import ConstrainedJSONDecoder


class HFBackend:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cpu",
        max_new_tokens: int = 384,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
            .to(device)
            .eval()
        )
        self.max_new_tokens = max_new_tokens
        self._decoder = None

    def _prompt_ids(self, messages: list[dict]) -> list[int]:
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tok.encode(text, add_special_tokens=False)

    @torch.no_grad()
    def generate_text(self, messages: list[dict]) -> str:
        ids = torch.tensor([self._prompt_ids(messages)], device=self.model.device)
        out = self.model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.eos_token_id,
        )
        return self.tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)

    def generate_constrained(self, messages: list[dict], schema) -> str:
        if self._decoder is None:
            self._decoder = ConstrainedJSONDecoder(self.model, self.tok)
        text, self.last_stats = self._decoder.generate(
            self._prompt_ids(messages), schema
        )
        return text
