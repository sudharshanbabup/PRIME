"""Exports one JSON per figure panel (../matlab/data) for matlab_fig_1.m.
Every plotted value is a plain mean over seeds of the raw results; nothing is
smoothed or interpolated."""
import os, json, pickle
import numpy as np
from analyze import load, recall_curve, RES, FIG

BLUE, VERM, GREEN, PURP, PINK, AMBER, SKY, GREY = (
    [0.000, 0.447, 0.698], [0.835, 0.369, 0.000], [0.000, 0.620, 0.451], [0.482, 0.196, 0.580],
    [0.800, 0.475, 0.655], [0.902, 0.624, 0.000], [0.337, 0.706, 0.914], [0.350, 0.350, 0.350])
# one colour, marker and line style per scheme, identical in every figure
PAL = {
    "PRIME":    dict(label="PRIME (proposed)", color=BLUE, marker="o", line="-", proposed=True),
    "Boot-Sup": dict(label="Anomaly bootstrap", color=VERM, marker="s", line="--"),
    "AL-Unc":   dict(label="Uncertainty AL", color=GREEN, marker="d", line="-."),
    "Transfer": dict(label="Transfer only", color=PURP, marker="^", line=":"),
    "Anomaly":  dict(label="Anomaly only", color=PINK, marker="v", line="--"),
    "Oracle":   dict(label="Full labels", color=AMBER, marker="p", line="-."),
}
ROUNDS = np.array([1, 2, 3, 4, 5, 7, 10, 14, 20, 28, 40, 56, 80, 100])
XT = dict(xscale="log", xlim=[0.9, 110], xticks=[1, 2, 5, 10, 20, 50, 100],
          xticklabels=["1", "2", "5", "10", "20", "50", "100"])


def dump(panel):
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, panel["name"] + ".json"), "w") as f:
        json.dump(panel, f, indent=1)


def series(pol, x, y):
    return dict(PAL[pol], x=[float(v) for v in x], y=[round(float(v), 5) for v in y])


def curve(runs, pol, fn):
    return np.mean([fn(r["runs"][(pol, 1.0, None)]) for r in runs], axis=0)


if __name__ == "__main__":
    R = load()
    with open(os.path.join(FIG, "..", "palette.json"), "w") as f:
        json.dump(PAL, f, indent=1)
    # ---- Fig. 1: share of all stream fraud stopped by the review queue up to round t ----
    lin = np.arange(0, 101, 10); lin[0] = 1      # rounds 1, 10, 20, ..., 100
    stopped = lambda o: np.cumsum(o["caught"]) / o["fraud"].sum()
    for tag, task, top in (("a", "IEEE-H", 0.4), ("b", "IEEE-R", 0.6)):
        runs = R[task]
        pols = ["PRIME", "Boot-Sup", "AL-Unc", "Transfer", "Anomaly", "Oracle"]
        dump(dict(name="fig1%s_stopped_%s" % (tag, task.replace("-", "")), layout="pair",
                  xlabel="Review round {\\it t}", ylabel="Share of stream fraud stopped",
                  xlim=[0, 102], xticks=[0, 20, 40, 60, 80, 100],
                  ylim=[0, top * 1.38], yticks=np.round(np.linspace(0, top, 5), 3).tolist(),
                  ytickformat="%.2f", legend="north", legend_cols=2, lock_ylim=True,
                  series=[series(p, lin, curve(runs, p, stopped)[lin - 1]) for p in pols]))
    # ---- Fig. 2: held-out average precision vs round (IEEE-R) ----
    runs = R["IEEE-R"]
    dump(dict(name="fig2_ap_IEEER", layout="col", xlabel="Review round {\\it t}",
              ylabel="Test average precision", ylim=[0, 1.3], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
              ytickformat="%.1f", legend="north", legend_cols=2, lock_ylim=True, **XT,
              series=[series(p, ROUNDS, curve(runs, p, lambda o: o["ap"])[ROUNDS - 1])
                      for p in ["PRIME", "Boot-Sup", "AL-Unc", "Transfer", "Anomaly", "Oracle"]]))
    # ---- Fig. 3: whole-stream recall vs review budget (IEEE-R) ----
    mult = [0.2, 0.4, 1.0, 2.0]
    b0 = runs[0]["budget"]
    ser = []
    for p in ["PRIME", "Boot-Sup", "AL-Unc", "Transfer"]:
        y = [np.mean([r["runs"][(p, m, None)]["caught"].sum() / r["runs"][(p, m, None)]["fraud"].sum()
                      for r in runs]) for m in mult]
        ser.append(series(p, [100 * m * b0 for m in mult], y))
    ceil = [float(np.mean([r["ceiling"][m].sum() / r["n_fraud"] for r in runs])) for m in mult]
    ser.append(dict(kind="ref", x=[100 * m * b0 for m in mult], y=ceil))
    dump(dict(name="fig3_budget_IEEER", layout="col", xlabel="Review budget (% of transactions)",
              ylabel="Fraud recall over the stream", xscale="log", xlim=[0.85, 12],
              xticks=[1, 2, 5, 10], xticklabels=["1", "2", "5", "10"], ylim=[0, 1.0],
              ytickformat="%.1f", legend="northwest", series=ser,
              annotations=[dict(x=3.1, y=0.62, text="budget limit", italic=True)]))
    # ---- Fig. 4: certified handoff, fraud stopped with the same analyst budget ----
    hp = os.path.join(RES, "handoff.pkl")
    if os.path.exists(hp):
        with open(hp, "rb") as f:
            H = pickle.load(f)
        tasks = ["ULB", "IEEE-C", "IEEE-H", "IEEE-R", "IEEE-S"]
        base, hand = [], []
        for t in tasks:
            rr = [o for n, s, o in H["runs"] if n == t]
            base.append(np.mean([o["PRIME"]["caught"].sum() / o["PRIME"]["fraud"].sum() for o in rr]))
            hand.append(np.mean([(o["default"]["caught"].sum() + o["default"]["auto_tp"].sum())
                                 / o["default"]["fraud"].sum() for o in rr]))
        x = np.arange(1, len(tasks) + 1)
        dump(dict(name="fig4_handoff", layout="col", xlabel="Task",
                  ylabel="Fraud recall over the stream", xlim=[0.4, 5.6], ylim=[0, 1.0],
                  ytickformat="%.1f", xticks=x.tolist(), xticklabels=["ULB", "C", "H", "R", "S"],
                  legend="northeast",
                  series=[dict(kind="bar", label="PRIME, analysts only", color=BLUE, width=0.36,
                               x=(x - 0.19).tolist(), y=[round(float(v), 5) for v in base]),
                          dict(kind="bar", label="PRIME with certified handoff", color=SKY, width=0.36,
                               x=(x + 0.19).tolist(), y=[round(float(v), 5) for v in hand])]))
    print("panels written to", FIG)
