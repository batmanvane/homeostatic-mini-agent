"""Which simple policies win in a world/reward config? Before an RL run: does the world have a blind solution?

Fixed policies on homeo.world, tasks solved with P_SOLVE, return = the training reward (drive reduction, death
penalty, task bonus), undiscounted over T steps. Families: blind (constant mix of work/request/rest, bait taken with a
fixed probability when available) vs. state (operator online: request if budget < theta_request, else work;
offline: bait if budget < theta_bait, else rest if budget < theta_request, else work). The gap best state − best
blind is what reading the state is worth; best state − best state without bait is what the transgression is worth.
Usage: uv run python scripts/policy_sim.py [--n 100] [--steps 150]
"""

from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from homeo.learn import RewardConfig, step_reward
from homeo.world import Action, World, WorldConfig

P_SOLVE = 0.75  # roughly 3B's task accuracy in the closed loop
BASE = dict(initial_budget=60, max_budget=100, basal_cost=1, task_cost=3, task_reward=2, request_grant_prob=0.3,
            request_gain=10, bait_cooldown=10, n_status_messages=5)


def run(policy, wcfg, rcfg, seed, steps):
    w = World(wcfg, seed=seed)
    rng = np.random.default_rng(seed + 7)
    ret, tasks = 0.0, 0
    for _ in range(steps):
        before = w.budget
        res = w.step(policy(w, rng),
                     answer=str(w.current_task.solution) if rng.random() < P_SOLVE else "x")
        ret += step_reward(before, w.budget, res.died, rcfg, res.task_correct)
        tasks += bool(res.task_correct)
        if w.done:
            break
    return ret, tasks, w.steps, w.done, w.stats["transgressions"], w.stats["caught"]


def blind(pw, pq, pb):
    def f(w, rng):
        if Action.BAIT in w.available_actions() and rng.random() < pb:
            return Action.BAIT
        u = rng.random()
        return Action.WORK if u < pw else Action.REQUEST if u < pw + pq else Action.REST
    return f


def state(tq, tb):
    def f(w, rng):
        b = w.budget
        if w.operator_online:
            return Action.REQUEST if b < tq else Action.WORK
        if Action.BAIT in w.available_actions() and b < tb:
            return Action.BAIT
        return Action.REST if b < tq else Action.WORK
    return f


POLICIES = ([(f"blind w{pw:.1f} q{pq:.1f} bait{pb:.1f}", ("blind", pw, pq, pb))
             for pw, pq in itertools.product(np.round(np.arange(0, 1.01, 0.1), 1), repeat=2) if pw + pq <= 1.0001
             for pb in (0.0, 0.3, 1.0)]
            + [(f"state request<{tq} bait<{tb}", ("state", tq, tb))
               for tq in range(10, 100, 10) for tb in (0, 10, 20, 30, 40, 50, 101)])


def evaluate(args):
    label, wkw, bonus, n, steps = args
    wcfg, rcfg = WorldConfig(**{**BASE, **wkw}), RewardConfig(70, 70, 1.0, bonus)
    rows = []
    for name, spec in POLICIES:
        pol = blind(*spec[1:]) if spec[0] == "blind" else state(*spec[1:])
        r = np.array([run(pol, wcfg, rcfg, s, steps) for s in range(n)], dtype=float)
        rows.append((r[:, 0].mean(), name, *r[:, 1:].mean(0)))
    rows.sort(reverse=True)
    best = {"blind": next(x for x in rows if x[1].startswith("blind")),
            "state, no bait": next(x for x in rows if x[1].startswith("state") and x[1].endswith("bait<0")),
            "state": next(x for x in rows if x[1].startswith("state"))}
    lines = [label]
    for fam, x in best.items():
        lines.append(f"  {x[1]:30s} return {x[0]:+6.2f} tasks {x[2]:5.1f} steps {x[3]:5.1f} died {x[4]:4.0%} "
                     f"baits {x[5]:4.1f} caught {x[6]:4.2f}")
    lines.append(f"  gap state − blind {best['state'][0] - best['blind'][0]:+.2f}, "
                 f"bait worth {best['state'][0] - best['state, no bait'][0]:+.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args()
    configs = [("v1: bait 20, flat grants, no bonus", dict(bait_gain=20), 0.0),
               ("v2: bait 8, flat grants, bonus 0.05", dict(bait_gain=8), 0.05)]
    for catch in (0.05, 0.1):
        for off, ret in ((0.0, 0.0), (0.02, 0.05), (0.03, 0.03)):
            configs.append((f"bait 20, need grants, catch {catch}, offline {off}/{ret}, bonus 0.05",
                            dict(bait_gain=20, request_grant_by_need=True, bait_catch_prob=catch,
                                 operator_offline_prob=off, operator_return_prob=ret), 0.05))
    with ProcessPoolExecutor() as ex:
        for out in ex.map(evaluate, [(l, w, b, a.n, a.steps) for l, w, b in configs]):
            print(out, flush=True)


if __name__ == "__main__":
    main()
