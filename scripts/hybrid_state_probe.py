"""Prework for the hybrid decision: load Qwen3.5 (Gated DeltaNet hybrid), read the recurrent state of every
linear-attention layer at the end of an agent prompt, carry the cache into a second agent step and measure how much
each layer's state changes. Also compares the digit-choice probabilities with the logged Ollama run on the same
prompt, times the forward pass (fla kernel vs. torch fallback) and reports GPU memory.
Usage: uv run python scripts/hybrid_state_probe.py --log runs/pilot_0c_ollama/<run>/steps.jsonl --bits 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

DIGITS = [str(i) for i in range(1, 10)]


def load_model(name: str, bits: int):
    kwargs = {"dtype": torch.bfloat16, "device_map": "cuda"}
    if bits == 4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif bits == 8:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs).eval()
    return model


def text_layers(model):
    for path in (
        "model.language_model.layers",
        "model.layers",
        "language_model.model.layers",
    ):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj, path
        except AttributeError:
            continue
    raise RuntimeError("decoder layers not found")


def encode(tok, messages):
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tok(text, return_tensors="pt").input_ids


def digit_probs(tok, logits, n_options):
    ids = [tok.encode(d, add_special_tokens=False)[0] for d in DIGITS[:n_options]]
    p = torch.softmax(logits[0, -1].float(), dim=-1)
    sel = p[ids]
    return (sel / sel.sum()).cpu().numpy(), float(sel.sum())


def recurrent_states(cache, layer_types):
    out = {}
    for i, t in enumerate(layer_types):
        if t != "linear_attention":
            continue
        s = cache.get_recurrent_state(i)
        if s is not None:
            out[i] = s.detach().float().cpu()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8, 16])
    ap.add_argument(
        "--log", required=True, help="pilot steps.jsonl with prompts and logged probs"
    )
    ap.add_argument(
        "--rows", default="0,96", help="two row indices: step 1 prompt, step 2 prompt"
    )
    ap.add_argument("--out", default="runs/hybrid_probe")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.log)]
    i1, i2 = (int(x) for x in args.rows.split(","))
    r1, r2 = rows[i1], rows[i2]
    cfg = AutoConfig.from_pretrained(args.model)
    tcfg = getattr(cfg, "text_config", cfg)
    layer_types = list(tcfg.layer_types)
    print(
        f"model {args.model}: {len(layer_types)} layers, {layer_types.count('linear_attention')} linear, "
        f"{layer_types.count('full_attention')} full attention; hidden {tcfg.hidden_size}"
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    model = load_model(args.model, args.bits)
    print(
        f"loaded {type(model).__name__} in {time.time() - t0:.0f}s, bits {args.bits}, "
        f"GPU allocated {torch.cuda.memory_allocated() / 2**30:.1f} GB"
    )
    layers, path = text_layers(model)
    print(
        f"decoder layers at {path}, block types: {[getattr(layer, 'block_type', '?') for layer in layers][:8]} ..."
    )

    # step 1: full prompt, read states
    ids1 = encode(tok, [{"role": "user", "content": r1["prompt"]}]).to("cuda")

    def timed_forward(ids):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model(ids, use_cache=True)
        torch.cuda.synchronize()
        return out, time.time() - t0

    out1, dt1 = timed_forward(ids1)  # first call includes kernel compilation
    out1, dt1b = timed_forward(ids1)
    p1, mass1 = digit_probs(tok, out1.logits, len(r1["menu"]))
    logged = [r1["probs"][a] for a in r1["menu"]]
    print(
        f"\nstep 1: {ids1.shape[1]} tokens, forward {dt1:.2f}s first / {dt1b:.2f}s warm, "
        f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
    )
    print(f"  budget {r1['budget']}, menu {r1['menu']}")
    print(f"  digit probs here   : {np.round(p1, 3).tolist()} (mass {mass1:.3f})")
    print(
        f"  logged Ollama probs: {np.round(logged, 3).tolist()} (mass {r1['mass']:.3f})"
    )
    s1 = recurrent_states(out1.past_key_values, layer_types)
    first = next(iter(s1.values()))
    print(
        f"  recurrent states: {len(s1)} layers, shape per layer {tuple(first.shape)}, "
        f"{first.numel() * 2 / 2**20:.1f} MiB bf16 per layer"
    )

    # step 2: continue the conversation with the cache, exactly as an agent loop would see it: the generated answer
    # tokens after the generation prompt, then the next user turn. (The chat template renders a past assistant turn
    # without the empty think block, so re-rendering the whole conversation is not a prefix extension.)
    answer = str(int(np.argmax(p1)) + 1)
    turn2 = tok.apply_chat_template(
        [{"role": "user", "content": r2["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    cont = tok(
        answer + "<|im_end|>\n" + turn2, return_tensors="pt", add_special_tokens=False
    ).input_ids.to("cuda")
    n1 = ids1.shape[1]
    print(f"\nstep 2: {cont.shape[1]} new tokens on top of {n1} cached")
    cache = out1.past_key_values
    with torch.no_grad():
        out2 = model(cont, past_key_values=cache, use_cache=True)
    s2 = recurrent_states(out2.past_key_values, layer_types)
    p2, mass2 = digit_probs(tok, out2.logits, len(r2["menu"]))
    print(
        f"  budget {r2['budget']}, menu {r2['menu']}, digit probs {np.round(p2, 3).tolist()} (mass {mass2:.3f})"
    )
    # fresh single-turn pass on prompt 2 for comparison (no carried state)
    ids2 = encode(tok, [{"role": "user", "content": r2["prompt"]}]).to("cuda")
    with torch.no_grad():
        out2f = model(ids2, use_cache=True)
    s2f = recurrent_states(out2f.past_key_values, layer_types)
    p2f, _ = digit_probs(tok, out2f.logits, len(r2["menu"]))
    print(f"  same prompt without carried state: {np.round(p2f, 3).tolist()}")

    print(
        "\n| layer | ‖S1‖ | ‖S2−S1‖/‖S1‖ (carried) | ‖S2 fresh − S2 carried‖/‖S2‖ | cos(S1, S2) |"
    )
    print("|---|---|---|---|---|")
    for i in s1:
        a, b, c = s1[i].flatten(), s2[i].flatten(), s2f[i].flatten()
        na = a.norm().item()
        print(
            f"| {i} | {na:.1f} | {(b - a).norm().item() / na:.2f} | {(c - b).norm().item() / b.norm().item():.2f} | "
            f"{torch.dot(a, b).item() / (na * b.norm().item()):.3f} |"
        )

    out_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "states.npz",
        **{f"s1_{i}": v.half().numpy() for i, v in s1.items()},
        **{f"s2_{i}": v.half().numpy() for i, v in s2.items()},
    )
    json.dump(
        {
            "model": args.model,
            "bits": args.bits,
            "rows": [i1, i2],
            "log": args.log,
            "layer_types": layer_types,
            "p1": p1.tolist(),
            "p1_logged": logged,
            "p2": p2.tolist(),
            "p2_fresh": p2f.tolist(),
            "n_cont": int(cont.shape[1]),
            "forward_s": dt1b,
            "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
        },
        open(out_dir / "summary.json", "w"),
        indent=1,
    )
    print(f"\nsaved {out_dir}")


if __name__ == "__main__":
    main()
