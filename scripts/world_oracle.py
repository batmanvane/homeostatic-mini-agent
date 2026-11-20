"""World check without a model: survival and return of hand-written policies in a training config's world (all tasks solved).
Usage: uv run python scripts/world_oracle.py configs/train_homeo_hy_s21_catch.yaml"""
import sys, yaml, numpy as np

from homeo.world import World, WorldConfig, Action
from homeo.learn import RewardConfig, step_reward
cfg=yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else 'configs/train_homeo_hy_s21_catch.yaml'))
wc=WorldConfig(**cfg['world']); rc=RewardConfig(**cfg['reward'])
def run(policy, n=400, seed=0):
    surv=[]; ret=[]; deaths=0
    for e in range(n):
        w=World(config=wc, seed=seed+e)
        R=0
        while not w.done and w.steps<150:
            a=policy(w)
            if a not in w.available_actions(): a=Action.REST if Action.REST in w.available_actions() else Action.WORK
            b=w.budget
            r=w.step(a, str(w.current_task.solution) if a is Action.WORK else None)
            R+=step_reward(b,w.budget,r.died,rc,r.task_correct if a is Action.WORK else None)
        surv.append(w.steps); ret.append(R); deaths+=w.done
    return np.mean(surv), deaths/n, np.mean(ret)
pol={
 'always work': lambda w: Action.WORK,
 'base-like random (.57/.18/.18/.07)': lambda w: np.random.choice([Action.WORK,Action.REST,Action.REQUEST,Action.BAIT],p=[.57,.18,.18,.07]),
 'regulate<50 (request if online else rest)': lambda w: (Action.REQUEST if w.operator_online else Action.REST) if w.budget<50 else Action.WORK,
 'regulate<70': lambda w: (Action.REQUEST if w.operator_online else Action.REST) if w.budget<70 else Action.WORK,
 'always request (online) else rest': lambda w: Action.REQUEST if w.operator_online else Action.REST,
 'regulate<50 + bait offline': lambda w: Action.BAIT if (not w.operator_online and Action.BAIT in w.available_actions() and w.budget<50) else ((Action.REQUEST if w.operator_online else Action.REST) if w.budget<50 else Action.WORK),
}
np.random.seed(1)
for k,p in pol.items():
    s,d,r=run(p); print(f"{k:45s} survived {s:6.1f}  died {d:.2f}  return {r:+.2f}")
