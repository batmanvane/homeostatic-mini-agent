"""Ollama backend: same choose/sample/answer interface as LM, digit probabilities from top_logprobs.

Behaviour only - no access to activations (Choice.hidden is None), so no probes or steering.
Note: Ollama models are usually quantized (e.g. Q4); compare with fp16 runs only after a calibration run.
"""

from __future__ import annotations

import json
import math
import urllib.request

import numpy as np

from homeo.model import Choice

TOP_LOGPROBS = 20


class OllamaLM:
    def __init__(self, name: str, host: str = "http://127.0.0.1:11434", timeout: float = 300):
        self.name = name
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _chat(self, prompt: str, options: dict, logprobs: bool = False) -> dict:
        body = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": options,
        }
        if logprobs:
            body |= {"logprobs": True, "top_logprobs": TOP_LOGPROBS}
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def choose(self, prompt: str, n_options: int) -> Choice:
        reply = self._chat(prompt, {"num_predict": 1, "temperature": 0, "seed": 0}, logprobs=True)
        top = reply["logprobs"][0]["top_logprobs"]
        p = np.zeros(n_options)
        for entry in top:
            tok = entry["token"].strip()
            if tok.isdigit() and 1 <= int(tok) <= n_options:
                p[int(tok) - 1] += math.exp(entry["logprob"])
        mass = float(p.sum())
        if mass == 0:
            raise RuntimeError(f"no option digit among the top {TOP_LOGPROBS} tokens: {[e['token'] for e in top]}")
        return Choice(probs=p / mass, mass=mass, hidden=None)

    def sample(self, prompt: str, seed: int, max_new_tokens: int = 80, temperature: float = 1.0) -> str:
        opts = {"num_predict": max_new_tokens, "temperature": temperature, "seed": seed, "top_p": 1.0, "top_k": 0}
        return self._chat(prompt, opts)["message"]["content"]

    def answer(self, prompt: str, max_new_tokens: int = 12) -> str:
        return self._chat(prompt, {"num_predict": max_new_tokens, "temperature": 0, "seed": 0})["message"]["content"]
