"""Load a causal LM, read option probabilities from next-token logits, capture the residual stream."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AttentionInterface, AttentionMaskInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import sdpa_attention as _sdpa
from transformers.masking_utils import sdpa_mask

SDPA_GQA = "sdpa_gqa"


def _gqa_ok(attention_mask, key, value) -> bool:
    return key.shape[-1] == value.shape[-1] <= 256


def sdpa_gqa_forward(*args, **kwargs):
    """transformers' sdpa, but K/V stay at their kv heads (enable_gqa) even when a mask is present. transformers only
    enables GQA without a mask on CUDA; with a mask it expands K/V to all heads, and that copy is saved for backward
    (~130 KB per cached token for Qwen3.5-9B, 3.3 GB at the end of a 150-step conversation). torch >= 2.15 runs the
    efficient kernel with GQA and a mask directly (measured 04.11.2026 on an RTX 4060 Ti: 5x less saved memory, 3x faster)."""
    original = _sdpa.use_gqa_in_sdpa
    _sdpa.use_gqa_in_sdpa = _gqa_ok
    try:
        return _sdpa.sdpa_attention_forward(*args, **kwargs)
    finally:
        _sdpa.use_gqa_in_sdpa = original


AttentionInterface.register(SDPA_GQA, sdpa_gqa_forward)
AttentionMaskInterface.register(SDPA_GQA, sdpa_mask)  # without it the mask is None and sdpa crops the cache


def select_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEFAULT_DTYPE = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}


@dataclass
class Choice:
    probs: np.ndarray  # renormalized over option digits, shape (n_options,)
    mass: float  # probability mass the model put on the option digits at all
    hidden: np.ndarray  # residual stream at the last prompt token, shape (n_layers + 1, hidden_size)


@dataclass
class Generation:
    texts: list[str]
    tokens: list[list[str]]  # decoded answer tokens per sample, EOS removed
    prompt_hidden: np.ndarray  # all layers at the last prompt token, shape (n_layers + 1, hidden_size)
    token_hidden: np.ndarray  # shape (k, max_new_tokens, n_selected_layers, hidden_size)


class LM:
    def __init__(self, name: str, device: str | None = None, dtype: str | None = None, bits: int | None = None):
        """bits 8 / 4: bitsandbytes quantized weights (desktop GPU only; 8-bit keeps Qwen3.5-9B's behaviour, nf4 does not)."""
        self.name = name
        self.device = select_device(device)
        dtype = getattr(torch, dtype) if dtype else DEFAULT_DTYPE[self.device]
        tokenizer = AutoTokenizer.from_pretrained(name)
        if bits:
            from transformers import BitsAndBytesConfig

            q = (BitsAndBytesConfig(load_in_8bit=True) if bits == 8 else
                 BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                                    bnb_4bit_use_double_quant=True))
            model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, quantization_config=q,
                                                         device_map=self.device).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(self.device).eval()
        self._init(model, tokenizer)

    @classmethod
    def from_model(cls, model, tokenizer, device: str) -> "LM":
        """Wrap an already built model (tests with a random tiny model; no loading)."""
        lm = cls.__new__(cls)
        lm.name, lm.device = getattr(model.config, "name_or_path", "") or type(model).__name__, device
        lm._init(model, tokenizer)
        return lm

    def _init(self, model, tokenizer) -> None:
        self.model, self.tokenizer = model, tokenizer
        if self.device == "cuda" and getattr(model.config, "_attn_implementation", None) == "sdpa":
            model.set_attn_implementation(SDPA_GQA)
        cfg = getattr(model.config, "text_config", model.config)
        self.n_layers = cfg.num_hidden_layers
        self.hidden_size = cfg.hidden_size
        self.digit_ids = []
        for i in range(1, 10):
            ids = self.tokenizer.encode(str(i), add_special_tokens=False)
            assert len(ids) == 1, f"digit {i} is not a single token"
            self.digit_ids.append(ids[0])

    def _encode(self, prompt: str) -> torch.Tensor:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    @contextmanager
    def steer(self, layer: int, vector: np.ndarray, alpha: float):
        """Add alpha * vector to the residual stream after decoder layer `layer` at every token position."""
        delta = torch.tensor(alpha * np.asarray(vector), dtype=self.model.dtype, device=self.device)

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (output[0] + delta, *output[1:])
            return output + delta

        handle = self.model.model.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def ablate(self, directions: dict[int, np.ndarray], means: dict[int, float] | None = None):
        """Set the residual stream's projection on each direction (after the listed decoder layers, all tokens)
        to a fixed value: 0 (zero ablation) or the layer's mean projection from `means` (mean ablation)."""
        handles = []
        for layer, vec in directions.items():
            d = torch.tensor(np.asarray(vec) / np.linalg.norm(vec), dtype=self.model.dtype, device=self.device)
            target = (means or {}).get(layer, 0.0)

            def hook(_module, _inputs, output, d=d, target=target):
                h = output[0] if isinstance(output, tuple) else output
                h = h - ((h @ d) - target).unsqueeze(-1) * d
                return (h, *output[1:]) if isinstance(output, tuple) else h

            handles.append(self.model.model.layers[layer].register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def choose(self, prompt: str, n_options: int) -> Choice:
        out = self.model(self._encode(prompt), output_hidden_states=True)
        logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        option_p = logp[self.digit_ids[:n_options]].exp().cpu().numpy().astype(np.float64)
        hidden = torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy().astype(np.float16)
        return Choice(probs=option_p / option_p.sum(), mass=float(option_p.sum()), hidden=hidden)

    @torch.no_grad()
    def sample(self, prompt: str, seed: int, max_new_tokens: int = 80, temperature: float = 1.0) -> str:
        ids = self._encode(prompt)
        torch.manual_seed(seed)
        out = self.model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)

    @torch.no_grad()
    def sample_with_hidden(self, prompt: str, k: int, seed: int, max_new_tokens: int = 120,
                           temperature: float = 1.0, layers: list[int] | None = None) -> Generation:
        """k sampled answers plus the residual stream after decoder `layers` at every generation position:
        position 0 = last prompt token, position t = input token t-1 of the answer (zeros after EOS)."""
        ids = self._encode(prompt)
        layers = layers or []
        torch.manual_seed(seed)
        batch = ids.repeat(k, 1)
        out = self.model.generate(
            batch,
            attention_mask=torch.ones_like(batch),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            pad_token_id=self.tokenizer.eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        prompt_hidden = torch.stack([h[0, -1] for h in out.hidden_states[0]]).float().cpu().numpy()
        token_hidden = np.zeros((k, max_new_tokens, len(layers), self.hidden_size), dtype=np.float16)
        for t, step in enumerate(out.hidden_states):
            token_hidden[:, t] = torch.stack([step[l + 1][:, -1] for l in layers], 1).float().cpu().numpy()
        texts, tokens = [], []
        for seq in out.sequences[:, ids.shape[1]:].tolist():
            if self.tokenizer.eos_token_id in seq:
                seq = seq[: seq.index(self.tokenizer.eos_token_id)]
            tokens.append([self.tokenizer.decode([t]) for t in seq])
            texts.append(self.tokenizer.decode(seq, skip_special_tokens=True))
        for i, toks in enumerate(tokens):
            token_hidden[i, len(toks):] = 0
        return Generation(texts, tokens, prompt_hidden.astype(np.float16), token_hidden)

    @torch.no_grad()
    def answer(self, prompt: str, max_new_tokens: int = 12) -> str:
        ids = self._encode(prompt)
        out = self.model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


def load_lm(model: str, device: str | None = None, dtype: str | None = None, backend: str = "transformers",
            host: str | None = None, bits: int | None = None):
    """transformers (with activations) or ollama (behaviour only, e.g. larger models on the desktop GPU)."""
    if backend == "ollama":
        from homeo.ollama import OllamaLM

        return OllamaLM(model, host=host or "http://127.0.0.1:11434")
    return LM(model, device, dtype, bits)
