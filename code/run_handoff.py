"""Certified handoff experiment (Section IV-E): PRIME with and without the
anytime-valid autonomy handoff, same analyst budget.  Also a Monte-Carlo check
of the false-handoff probability of the betting test under the null.

Usage: python run_handoff.py [n_seeds] [n_workers]   ->  ../results/handoff.pkl
"""
import os, sys, pickle
import numpy as np
from multiprocessing import Pool
from coldstart import make_task, raw_task, run_stream, EProcess
from run_experiments import TASKS, BUDGET, OUT

BASE = dict(theta=0.9, pi0=0.8, pi1=0.95, delta=0.05, audit=0.2)
VARIANTS = {"default": {}, "theta0.8": dict(theta=0.8), "theta0.95": dict(theta=0.95),
            "audit0.1": dict(audit=0.1), "audit0.5": dict(audit=0.5), "pi0=0.9": dict(pi0=0.9, pi1=0.98)}
_RAW = {}


def job(args):
    name, seed = args
    if name not in _RAW:
        _RAW[name] = raw_task(name)
    task = make_task(name, seed, raw=_RAW[name])
    keep = ("caught", "fraud", "reviewed", "auto", "auto_tp", "certified", "region", "region_tp")
    out = {"PRIME": run_stream(task, "PRIME", BUDGET[name], seed, eval_test=False)}
    for k, v in VARIANTS.items():
        out[k] = run_stream(task, "PRIME", BUDGET[name], seed, handoff=dict(BASE, **v), eval_test=False)
    return name, seed, {k: {q: o[q] for q in keep} for k, o in out.items()}


def null_check(n_rep=100000, n_audit=2000, seed=0):
    """False-handoff frequency when the region precision equals pi0 (worst null)."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_rep):
        hits += EProcess(BASE["pi0"], BASE["pi1"], BASE["delta"]).update(rng.random(n_audit) < BASE["pi0"])
    return hits / n_rep


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_work = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    with Pool(n_work) as pool:
        res = pool.map(job, [(t, s) for t in TASKS for s in range(n_seeds)], chunksize=1)
    fa = null_check()
    print("false-handoff rate under the null (delta = %.2f): %.4f" % (BASE["delta"], fa))
    with open(os.path.join(OUT, "handoff.pkl"), "wb") as f:
        pickle.dump(dict(runs=res, base=BASE, variants=VARIANTS, null_rate=fa), f)
