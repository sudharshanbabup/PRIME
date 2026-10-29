"""Prints the numbers quoted in the paper (tables, paired tests, handoff)."""
import os, pickle, numpy as np
from scipy.stats import wilcoxon
from analyze import load, stats, RES, TASKS

R = load()
print("TABLE II  (mean over seeds)   R20 / R100")
for p in ["Random", "Anomaly", "Transfer", "AL-Unc", "Boot-Sup", "PRIME", "Oracle",
          "PRIME-noA", "PRIME-noS", "PRIME-noX", "PRIME-UCB", "PRIME-TS"]:
    row = []
    for t in TASKS:
        st = stats(R[t], (p, 1.0, None))
        row.append("  --  /  -- " if st is None else "%.3f/%.3f" % (st["R_early"].mean(), st["R_all"].mean()))
    print("%-10s " % p + "  ".join(row))
print("\nPRIME vs Boot-Sup, paired over seeds (Wilcoxon signed-rank, two-sided)")
for t in TASKS:
    a, b = stats(R[t], ("PRIME", 1.0, None)), stats(R[t], ("Boot-Sup", 1.0, None))
    for k in ("R_early", "R_all", "AP_early"):
        print("  %-7s %-8s PRIME %.3f  Boot %.3f  rel %+5.1f%%  wins %2d/%d  p=%.2g" % (
            t, k, a[k].mean(), b[k].mean(), 100 * (a[k].mean() / b[k].mean() - 1),
            np.sum(a[k] > b[k]), len(a[k]), wilcoxon(a[k], b[k]).pvalue))
print("\nAP mean over rounds / AP at round 100")
for p in ["AL-Unc", "Boot-Sup", "PRIME", "Oracle"]:
    print("%-10s " % p + "  ".join("%.3f/%.3f" % (stats(R[t], (p, 1.0, None))["AP_mean"].mean(),
                                                 stats(R[t], (p, 1.0, None))["AP_end"].mean()) for t in TASKS))
print("\nsensitivity: range of R20 and R100 of PRIME over the 8 hyper-parameter settings")
for t in TASKS:
    keys = sorted({k for r in R[t] for k in r["runs"] if k[2] is not None}, key=str)
    v = np.array([[stats(R[t], k)[m].mean() for m in ("R_early", "R_all")] for k in keys])
    print("  %-7s R20 [%.3f, %.3f]  R100 [%.3f, %.3f]" % (t, v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()))
hp = os.path.join(RES, "handoff.pkl")
if os.path.exists(hp):
    H = pickle.load(open(hp, "rb"))
    print("\nHANDOFF  null false-handoff rate %.4f" % H["null_rate"])
    for var in H["variants"]:
        print(" variant", var, {**H["base"], **H["variants"][var]})
        for t in TASKS:
            rr = [o for n, s, o in H["runs"] if n == t]
            b = np.array([o["PRIME"]["caught"].sum() / o["PRIME"]["fraud"].sum() for o in rr])
            h = np.array([(o[var]["caught"].sum() + o[var]["auto_tp"].sum()) / o[var]["fraud"].sum() for o in rr])
            au = np.array([o[var]["auto"].sum() for o in rr]); tp = np.array([o[var]["auto_tp"].sum() for o in rr])
            cert = np.array([np.argmax(o[var]["certified"]) + 1 if o[var]["certified"].any() else np.nan for o in rr])
            ok = au > 0
            print("   %-7s recall %.3f -> %.3f (%+5.1f%%) wins %2d/%d | certified %2d/%d, round mean %s | auto-blocked %.1f, pooled precision %s, min per-run %s, false blocks/run %.2f" % (
                t, b.mean(), h.mean(), 100 * (h.mean() / b.mean() - 1), np.sum(h > b), len(b), np.sum(~np.isnan(cert)), len(b),
                "%.1f" % np.nanmean(cert) if np.any(~np.isnan(cert)) else "-", au.mean(),
                "%.3f" % (tp.sum() / au.sum()) if au.sum() else "-", "%.3f" % np.min(tp[ok] / au[ok]) if ok.any() else "-", (au - tp).mean()))
