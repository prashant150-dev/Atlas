"""Real Hugging Face base model (bf16, CPU) — the quality generation path.

Ported and hardened from ``projects/v2_design/integration/atlas_engine_full.py``.
Loads a local checkpoint lazily on first ``generate`` so importing the engine is
cheap and offline. Applies a chat template when the model is a chat model.
"""

from __future__ import annotations

from pathlib import Path


class HFModel:
    """A local Hugging Face causal-LM wrapped behind the ``BaseModel`` interface."""

    def __init__(self, model_path: str, chat: bool = True, max_new_tokens: int = 80) -> None:
        """Configure (but do not yet load) a local model.

        Args:
            model_path: Local directory containing the checkpoint + tokenizer.
            chat: Whether to wrap prompts in the tokenizer's chat template.
            max_new_tokens: Default generation cap.
        """

        if not isinstance(model_path, str) or not model_path:
            raise ValueError("model_path must be a non-empty string")
        if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        self.model_path = model_path
        self.chat = bool(chat)
        self.max_new_tokens = int(max_new_tokens)
        self._tok = None
        self._lm = None

    def _load(self) -> None:
        """Load the tokenizer + model into memory (idempotent)."""

        if self._lm is not None:
            return
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"model not found at {self.model_path!r}. Download it into models/ first."
            )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        try:
            self._lm = AutoModelForCausalLM.from_pretrained(
                self.model_path, local_files_only=True, dtype=torch.bfloat16
            ).eval()
        except TypeError:  # older transformers: dtype kwarg not yet renamed
            self._lm = AutoModelForCausalLM.from_pretrained(
                self.model_path, local_files_only=True, torch_dtype=torch.bfloat16
            ).eval()

    def generate(self, prompt: str, max_new_tokens: int | None = None,
                 sample: bool = False, temperature: float = 0.8,
                 seed: int | None = None) -> str:
        """Generate text for ``prompt``.

        Greedy by default (deterministic). With ``sample=True`` it draws a diverse
        completion (temperature + nucleus), seeded per call so self-consistency gets
        genuinely different chains.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        self._load()
        import torch

        cap = int(max_new_tokens or self.max_new_tokens)
        if self.chat:
            messages = [{"role": "user", "content": prompt}]
            text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        enc = self._tok(text, return_tensors="pt")
        ids = enc.input_ids
        kwargs = {
            "attention_mask": enc.attention_mask,  # reliable results when pad==eos
            "max_new_tokens": cap,
            "pad_token_id": self._tok.eos_token_id,
        }
        if sample:
            if seed is not None:
                torch.manual_seed(int(seed))
            kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        with torch.inference_mode():
            out = self._lm.generate(ids, **kwargs)
        return self._tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def _self_test() -> None:
    """Validate construction + guards without loading weights (offline-safe)."""

    model = HFModel("models/qwen2.5-1.5b", chat=True)
    if model._lm is not None:
        raise RuntimeError("model should load lazily, not at construction")
    for bad in (lambda: HFModel("", chat=True), lambda: HFModel("x", max_new_tokens=0)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise RuntimeError("expected ValueError for invalid args")
    try:
        model.generate("")
    except ValueError:
        pass
    else:
        raise RuntimeError("empty prompt should raise ValueError")
    print("HFModel self-test")
    print(f"  lazy: not loaded at construction; path={model.model_path}")
    print("  status: ok (weights not loaded — offline)")


if __name__ == "__main__":
    _self_test()
