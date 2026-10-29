"""Runs every experiment of the paper and stores raw per-seed, per-round results.

Usage:  python run_experiments.py [n_seeds] [n_workers]
Output: ../results/<task>_seed<k>.pkl  (one file per task and seed; resumable)
"""
import os, sys, pickle, time
import numpy as np
from multiprocessing import Pool
from coldstart import make_task, raw_task, run_stream

TASKS = ["ULB", "IEEE-C", "IEEE-H", "IEEE-R", "IEEE-S"]
BUDGET = {"ULB": 0.005, "IEEE-C": 0.05, "IEEE-H": 0.05, "IEEE-R": 0.05, "IEEE-S": 0.05}
SWEEP = [0.2, 0.4, 1.0, 2.0]                      # multiples of the default budget
MAIN = ["Random", "Anomaly", "Transfer", "PriorSum", "AL-Unc", "Boot-Sup", "PRIME", "Oracle"]
ABL = ["PRIME-noA", "PRIME-noS", "PRIME-noX", "PRIME-UCB", "PRIME-TS"]
SENS = [("lam_x", 0.1), ("lam_x", 0.3), ("lam_x", 3.0), ("lam_x", 10.0),
        ("lam_p", 0.1), ("lam_p", 10.0), ("base_rate", 0.001), ("base_rate", 0.1)]
HANDOFF = dict(theta=0.9, pi0=0.8, pi1=0.95, delta=0.05, audit=0.2)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
_RAW = {}


def job(args):
    name, seed = args
    fn = os.path.join(OUT, "%s_seed%02d.pkl" % (name, seed))
    if os.path.exists(fn):
        return fn
    if name not in _RAW:
        _RAW[name] = raw_task(name)
    task = make_task(name, seed, raw=_RAW[name])
    b0 = BUDGET[name]
    res = dict(name=name, seed=seed, budget=b0, n_round=len(task["rounds"][0]),
               n_stream=len(task["y"]), n_fraud=float(task["y"].sum()),
               n_test=len(task["yt"]), dim=task["Z"].shape[1], runs={})
    # best achievable with the same budget (perfect ranking)
    res["ceiling"] = {m: np.array([min(max(1, int(round(m * b0 * len(i)))), task["y"][i].sum())
                                   for i in task["rounds"]]) for m in SWEEP}
    skip = lambda p: (not task["has_src"]) and p in ("Transfer", "PriorSum", "PRIME-noA", "PRIME-noS")
    for p in MAIN + ABL:
        if not skip(p):
            res["runs"][(p, 1.0, None)] = run_stream(task, p, b0, seed)
    for m in SWEEP:
        if m == 1.0:
            continue
        for p in ["Anomaly", "Transfer", "AL-Unc", "Boot-Sup", "PRIME"]:
            if not skip(p):
                res["runs"][(p, m, None)] = run_stream(task, p, m * b0, seed)
    for key, val in SENS:
        res["runs"][("PRIME", 1.0, (key, val))] = run_stream(task, "PRIME", b0, seed, {key: val})
    res["runs"][("PRIME+H", 1.0, None)] = run_stream(task, "PRIME", b0, seed, handoff=HANDOFF)
    with open(fn + ".tmp", "wb") as f:
        pickle.dump(res, f)
    os.replace(fn + ".tmp", fn)
    return fn


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_work = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    os.makedirs(OUT, exist_ok=True)
    jobs = [(t, s) for t in TASKS for s in range(n_seeds)]
    t0 = time.time()
    with Pool(n_work) as pool:
        for k, fn in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            print("%3d/%d %s  %.0fs" % (k + 1, len(jobs), os.path.basename(fn), time.time() - t0), flush=True)
