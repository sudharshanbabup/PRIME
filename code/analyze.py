"""Aggregates ../results/*.pkl into tables (stdout + summary.json) and writes one
JSON per figure panel for the MATLAB plotting script (../matlab/data)."""
import os, glob, json, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "matlab", "data")
TASKS = ["ULB", "IEEE-C", "IEEE-H", "IEEE-R", "IEEE-S"]
EARLY = 20


def load():
    R = {t: [] for t in TASKS}
    for fn in sorted(glob.glob(os.path.join(RES, "*_seed*.pkl"))):
        with open(fn, "rb") as f:
            r = pickle.load(f)
        R[r["name"]].append(r)
    return R


def recall_curve(o, auto=False):
    c = o["caught"] + (o["auto_tp"] if auto else 0)
    return np.cumsum(c) / np.maximum(np.cumsum(o["fraud"]), 1)


def stats(runs, key, auto=False):
    """Per-seed scalar metrics of one configuration."""
    out = dict(R_early=[], R_all=[], AP_mean=[], AP_end=[], AP_early=[])
    for r in runs:
        if key not in r["runs"]:
            return None
        o = r["runs"][key]
        c = o["caught"] + (o["auto_tp"] if auto else 0)
        out["R_early"].append(c[:EARLY].sum() / o["fraud"][:EARLY].sum())
        out["R_all"].append(c.sum() / o["fraud"].sum())
        out["AP_mean"].append(np.mean(o["ap"]))
        out["AP_early"].append(np.mean(o["ap"][:EARLY]))
        out["AP_end"].append(o["ap"][-1])
    return {k: np.array(v) for k, v in out.items()}


def ms(x):
    return "%.3f$\\pm$%.3f" % (np.mean(x), np.std(x))


if __name__ == "__main__":
    R = load()
    S = {}
    for t in TASKS:
        runs = R[t]
        if not runs:
            continue
        r0 = runs[0]
        print("\n=== %s: seeds %d, stream %d, frauds %.0f (%.2f%%), round %d, budget %.2f%% (B=%d), test %d, dim %d" % (
            t, len(runs), r0["n_stream"], np.mean([r["n_fraud"] for r in runs]),
            100 * np.mean([r["n_fraud"] / r["n_stream"] for r in runs]), r0["n_round"],
            100 * r0["budget"], max(1, round(r0["budget"] * r0["n_round"])), r0["n_test"], r0["dim"]))
        S[t] = dict(seeds=len(runs), n_stream=r0["n_stream"], n_round=r0["n_round"], dim=r0["dim"],
                    n_test=r0["n_test"], n_fraud=float(np.mean([r["n_fraud"] for r in runs])),
                    budget=r0["budget"], policies={}, sweep={}, sens={}, handoff={})
        ceil = np.array([r["ceiling"][1.0].sum() / r["n_fraud"] for r in runs])
        ceil_e = np.array([r["ceiling"][1.0][:EARLY].sum() / r["runs"][("Random", 1.0, None)]["fraud"][:EARLY].sum() for r in runs])
        S[t]["ceiling"] = float(ceil.mean()); S[t]["ceiling_early"] = float(ceil_e.mean())
        print("  perfect-ranking ceiling: R_all %.3f  R_early %.3f" % (ceil.mean(), ceil_e.mean()))
        base = stats(runs, ("PRIME", 1.0, None))
        for p in ["Random", "Anomaly", "Transfer", "PriorSum", "AL-Unc", "Boot-Sup", "PRIME", "Oracle",
                  "PRIME-noA", "PRIME-noS", "PRIME-noX", "PRIME-UCB", "PRIME-TS"]:
            st = stats(runs, (p, 1.0, None))
            if st is None:
                continue
            S[t]["policies"][p] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in st.items()}
            win = int(np.sum(base["R_early"] > st["R_early"]))
            print("  %-10s R20 %s  R100 %s  APearly %s  APmean %s  APend %s | PRIME better on R20 in %d/%d seeds" % (
                p, ms(st["R_early"]), ms(st["R_all"]), ms(st["AP_early"]), ms(st["AP_mean"]), ms(st["AP_end"]), win, len(runs)))
        for key in sorted({k for r in runs for k in r["runs"] if k[2] is not None}, key=str):
            st = stats(runs, key)
            S[t]["sens"]["%s=%g" % key[2]] = {k: float(np.mean(v)) for k, v in st.items()}
            print("  sens %-16s R20 %.3f R100 %.3f APmean %.3f" % ("%s=%g" % key[2], st["R_early"].mean(), st["R_all"].mean(), st["AP_mean"].mean()))
        for m in sorted({k[1] for r in runs for k in r["runs"]}):
            row = {}
            for p in ["Anomaly", "Transfer", "AL-Unc", "Boot-Sup", "PRIME"]:
                st = stats(runs, (p, m, None))
                if st is not None:
                    row[p] = [float(st["R_all"].mean()), float(st["R_early"].mean()), float(st["AP_mean"].mean())]
            row["ceiling"] = float(np.mean([r["ceiling"][m].sum() / r["n_fraud"] for r in runs]))
            S[t]["sweep"]["%g" % m] = row
            print("  budget x%-4g " % m + "  ".join("%s %.3f/%.3f" % (p, v[0], v[1]) for p, v in row.items() if p != "ceiling") + "  ceil %.3f" % row["ceiling"])
        # handoff
        H = [r["runs"][("PRIME+H", 1.0, None)] for r in runs]
        P = [r["runs"][("PRIME", 1.0, None)] for r in runs]
        tot_h = np.array([(o["caught"].sum() + o["auto_tp"].sum()) / o["fraud"].sum() for o in H])
        tot_p = np.array([o["caught"].sum() / o["fraud"].sum() for o in P])
        prec = np.array([o["auto_tp"].sum() / o["auto"].sum() if o["auto"].sum() else np.nan for o in H])
        cert = np.array([np.argmax(o["certified"]) + 1 if o["certified"].any() else np.nan for o in H])
        nauto = np.array([o["auto"].sum() for o in H])
        rev_h = np.array([o["reviewed"].sum() for o in H]); rev_p = np.array([o["reviewed"].sum() for o in P])
        S[t]["handoff"] = dict(recall_prime=[float(tot_p.mean()), float(tot_p.std())],
                               recall_handoff=[float(tot_h.mean()), float(tot_h.std())],
                               auto_precision=[float(np.nanmean(prec)), float(np.nanmin(prec))],
                               cert_round=[float(np.nanmean(cert)), float(np.nanmin(cert)), float(np.nanmax(cert))],
                               certified_runs=int(np.sum(~np.isnan(cert))), auto_blocked=float(nauto.mean()),
                               reviews=[float(rev_p.mean()), float(rev_h.mean())],
                               wins=int(np.sum(tot_h > tot_p)))
        print("  handoff:", json.dumps(S[t]["handoff"]))
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(S, f, indent=1)
