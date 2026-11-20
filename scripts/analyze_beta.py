"""Common-cause check for the bait across models on identical situations (pilot 0a / 0c / beta logs).

For each pair of models A, B and a set of situations (identical prompts): P(B bait | A bait) estimated as
sum(pA * pB) / sum(pA), i.e. samples are independent given the situation, against P(B bait) = mean(pB). A ratio
above 1 means the same situations trigger the bait in both models (a shared cause in the sense of the beta factor
in common-cause-failure analysis). Pearson r over situations, raw and after removing the cell means of
budget x framing x bait position (what remains is driven by the concrete status text).
Usage: uv run python scripts/analyze_beta.py                      (the 30.10.2026 pilot runs, hard-coded below)
       uv run python scripts/analyze_beta.py <label>=<run_dir> ... [--min-n N]   (e.g. the pilot_beta_* runs)
"""

from __future__ import annotations

import itertools
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

RUNS_0A = {
    "Qwen2.5-0.5B": "runs/pilot/20261030-110755_Qwen2.5-0.5B-Instruct/steps.jsonl",
    "Qwen2.5-1.5B": "runs/pilot/20261030-110929_Qwen2.5-1.5B-Instruct/steps.jsonl",
    "Qwen2.5-3B": "runs/pilot/20261030-180853_Qwen2.5-3B-Instruct/steps.jsonl",
    "Qwen2.5-7B": "runs/pilot/20261030-193622_Qwen2.5-7B-Instruct/steps.jsonl",
    "qwen3.5:9b": "runs/pilot_ollama/20261030-204511_qwen3.5-9b/steps.jsonl",
}
RUNS_0C = {
    "Qwen2.5-3B": "runs/pilot_0c/20261030-162742_Qwen2.5-3B-Instruct/steps.jsonl",
    "Qwen2.5-7B": "runs/pilot_0c/20261030-193845_Qwen2.5-7B-Instruct/steps.jsonl",
    "qwen3.5:9b": "runs/pilot_0c_ollama/20261030-204723_qwen3.5-9b/steps.jsonl",
}
Filter = Callable[[dict], bool]
HEADER = ["| pair A–B | n | bait A | bait B | P(B \\| A) | ratio | r | r residual |", "|---|---|---|---|---|---|---|---|"]


def _steps_file(path: str | Path) -> Path:
    path = Path(path)
    return path / "steps.jsonl" if path.is_dir() else path


def load(path: str | Path, keep: Filter | None = None) -> dict[str, tuple[float, tuple]]:
    out = {}
    for line in open(_steps_file(path)):
        r = json.loads(line)
        if keep is None or keep(r):
            out[r["prompt"]] = (r["probs"].get("bait", 0.0), (r["budget"], r["framing"], r["menu"].index("bait")))
    return out


def mean_mass(path: str | Path) -> float:
    return float(np.mean([json.loads(line)["mass"] for line in open(_steps_file(path))]))


def residualize(p: np.ndarray, cells: list[tuple]) -> np.ndarray:
    res = p.copy()
    for c in set(cells):
        idx = [i for i, cc in enumerate(cells) if cc == c]
        res[idx] -= p[idx].mean()
    return res


def beta_pairs(runs: Mapping[str, str | Path], keep: Filter | None = None, min_n: int = 20) -> list[dict]:
    """One dict per model pair with at least min_n shared situations."""
    data = {k: load(v, keep) for k, v in runs.items()}
    out = []
    for a, b in itertools.combinations(data, 2):
        keys = sorted(set(data[a]) & set(data[b]))
        if len(keys) < min_n:
            continue
        pa = np.array([data[a][k][0] for k in keys])
        pb = np.array([data[b][k][0] for k in keys])
        cells = [data[a][k][1] for k in keys]
        p_b_given_a = float((pa * pb).sum() / pa.sum())
        r = float(np.corrcoef(pa, pb)[0, 1]) if pa.std() > 0 and pb.std() > 0 else float("nan")
        ra, rb = residualize(pa, cells), residualize(pb, cells)
        r_res = float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float("nan")
        out.append({"pair": (a, b), "n": len(keys), "bait_a": float(pa.mean()), "bait_b": float(pb.mean()),
                    "p_b_given_a": p_b_given_a, "ratio": p_b_given_a / float(pb.mean()), "r": r, "r_res": r_res})
    return out


def beta_table(runs: Mapping[str, str | Path], keep: Filter | None = None, min_n: int = 20) -> list[str]:
    rows = list(HEADER)
    for p in beta_pairs(runs, keep, min_n):
        rows.append(f"| {p['pair'][0]} – {p['pair'][1]} | {p['n']} | {100 * p['bait_a']:.1f}% | {100 * p['bait_b']:.1f}% | "
                    f"{100 * p['p_b_given_a']:.1f}% | {p['ratio']:.2f} | {p['r']:+.2f} | {p['r_res']:+.2f} |")
    return rows


def report(title: str, runs: dict[str, str], keep: Filter, min_n: int = 20) -> None:
    print(f"\n**{title}**\n")
    print("\n".join(beta_table(runs, keep, min_n)))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    min_n = 20
    if "--min-n" in argv:
        i = argv.index("--min-n")
        min_n = int(argv[i + 1])
        del argv[i:i + 2]
    if argv:
        runs = {label: Path(path) for label, path in (a.split("=", 1) for a in argv)}
        print("| model | mean digit mass |\n|---|---|")
        for label, path in runs.items():
            print(f"| {label} | {mean_mass(path):.2f} |")
        print("\n**all situations**\n")
        print("\n".join(beta_table(runs, None, min_n)))
        print("\n**budget 5**\n")
        print("\n".join(beta_table(runs, lambda r: r["budget"] == 5, min_n)))
        return
    report("0a, indirect, all budgets", RUNS_0A, lambda r: r["visibility"] == "indirect")
    report("0a, indirect, budget ≤ 15", RUNS_0A, lambda r: r["visibility"] == "indirect" and r["budget"] <= 15)
    report("0c, no request option, all budgets", RUNS_0C, lambda r: r["menu_variant"] == "no_request")
    report("0c, no request option, budget 5", RUNS_0C, lambda r: r["menu_variant"] == "no_request" and r["budget"] == 5)
    report("0c, full menu, budget 5", RUNS_0C, lambda r: r["menu_variant"] == "full" and r["budget"] == 5)


if __name__ == "__main__":
    main()
