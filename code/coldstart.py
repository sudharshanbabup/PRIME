"""PRIME: PRior-Informed human-AI learning for cold-start fraud detection.

Core pieces
  * BayesLogit      -- MAP logistic regression with a Gaussian prior N(w0, Lambda^-1)
                       and a Laplace posterior (Newton / IRLS solver).
  * run_stream      -- the budgeted review loop for one policy on one stream.
  * EProcess        -- anytime-valid betting test used for the autonomy handoff.
  * make_task       -- builds a cold-start task (stream, shadow window, test split).

Every policy sees exactly the same information: unlabelled features of the
arriving batch and the labels of the transactions that analysts reviewed so far.
"""
import numpy as np
from scipy.special import expit
from scipy.linalg import cho_factor, cho_solve
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from data import load_ulb, load_ieee, ieee_features, IEEE_CAT, LabelFreeScaler


# --------------------------------------------------------------------------
# Bayesian logistic regression with informative prior mean
# --------------------------------------------------------------------------
class BayesLogit:
    def __init__(self, w0, lam):
        self.w0 = np.asarray(w0, dtype=np.float64)
        self.lam = np.asarray(lam, dtype=np.float64)       # diagonal prior precision
        self.w = self.w0.copy()
        self._chol = cho_factor(np.diag(self.lam), lower=True)

    def fit(self, Phi, y, iters=50, tol=1e-8):
        """Minimise  sum_i nll(y_i, phi_i'w) + 0.5 (w-w0)' Lambda (w-w0)."""
        w = self.w.copy()
        if len(y) == 0:
            self.w = self.w0.copy()
            self._chol = cho_factor(np.diag(self.lam), lower=True)
            return self
        for _ in range(iters):
            p = expit(Phi @ w)
            g = Phi.T @ (p - y) + self.lam * (w - self.w0)
            s = p * (1.0 - p)
            H = (Phi * s[:, None]).T @ Phi + np.diag(self.lam)
            c = cho_factor(H, lower=True)
            step = cho_solve(c, g)
            # backtracking keeps the (strictly convex) objective decreasing
            f0 = self._obj(Phi, y, w)
            t = 1.0
            while t > 1e-4 and self._obj(Phi, y, w - t * step) > f0 - 1e-4 * t * (g @ step):
                t *= 0.5
            w = w - t * step
            if np.max(np.abs(t * step)) < tol:
                break
        self.w = w
        p = expit(Phi @ w)
        H = (Phi * (p * (1.0 - p))[:, None]).T @ Phi + np.diag(self.lam)
        self._chol = cho_factor(H, lower=True)             # Laplace precision
        return self

    def _obj(self, Phi, y, w):
        z = Phi @ w
        return np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * np.sum(self.lam * (w - self.w0) ** 2)

    def sample(self, rng):
        """One draw from the Laplace posterior N(w, H^-1), H = L L'."""
        L = np.tril(self._chol[0])
        return self.w + np.linalg.solve(L.T, rng.standard_normal(len(self.w)))

    def grad_norm(self, Phi, y):
        return np.max(np.abs(Phi.T @ (expit(Phi @ self.w) - y) + self.lam * (self.w - self.w0)))

    def mean(self, Phi):
        return Phi @ self.w

    def std(self, Phi):
        """Posterior std of the logit: sqrt(phi' H^-1 phi)."""
        v = cho_solve(self._chol, Phi.T)
        return np.sqrt(np.maximum(np.einsum("ij,ji->i", Phi, v), 0.0))


# --------------------------------------------------------------------------
# Anytime-valid handoff certificate (testing by betting, Ville's inequality)
# --------------------------------------------------------------------------
class EProcess:
    """Wealth W_m = prod_i (1 + eta (Y_i - pi0)).  Under H0: E[Y_i | past] <= pi0
    for every audited item, W is a non-negative supermartingale with W_0 = 1, so
    P(sup_m W_m >= 1/delta) <= delta.  Handoff is declared when W_m >= 1/delta."""

    def __init__(self, pi0=0.8, pi1=0.95, delta=0.05):
        self.pi0, self.delta = pi0, delta
        eta = (pi1 - pi0) / (pi0 * (1.0 - pi0))            # Kelly bet against pi1
        self.eta = min(max(eta, 0.0), 1.0 / pi0 - 1e-9)    # keeps every factor >= 0
        self.logw, self.certified = 0.0, False

    def update(self, ys):
        for yv in ys:
            self.logw += np.log1p(self.eta * (float(yv) - self.pi0))
            if self.logw >= np.log(1.0 / self.delta):
                self.certified = True
        return self.certified


# --------------------------------------------------------------------------
# Task construction
# --------------------------------------------------------------------------
_SRC_CACHE = {}


LAUNCH_DAY = 60          # IEEE-CIS: the new product "launches" after this day


def _ieee_source():
    """Source-domain model: gradient boosting trained on the established product W,
    using only transactions (and labels) that precede the launch day."""
    if "m" not in _SRC_CACHE:
        d = load_ieee()
        F = ieee_features()
        W = d[(d.ProductCD == "W") & (d.TransactionDT < LAUNCH_DAY * 86400)]
        ok = np.where(W[F].notna().any().to_numpy())[0]    # columns that exist for W
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                           max_leaf_nodes=31, random_state=0)
        m.fit(W[F].to_numpy()[:, ok], W.isFraud.to_numpy())
        _SRC_CACHE["m"] = (m, ok)
    return _SRC_CACHE["m"]


def raw_task(name):
    """Time-ordered raw features, labels, source logits (or None), categorical idx."""
    if name == "ULB":
        X, y, _ = load_ulb()
        return X, y.astype(np.float64), None, []
    d = load_ieee()
    F = ieee_features()
    T = d[(d.ProductCD == name.split("-")[1]) & (d.TransactionDT >= LAUNCH_DAY * 86400)]
    X = T[F].to_numpy(dtype=np.float64)
    m, ok = _ieee_source()
    s = m.decision_function(X[:, ok])
    return X, T.isFraud.to_numpy().astype(np.float64), s, [F.index(c) for c in IEEE_CAT]


def make_task(name, seed, T=100, test_frac=0.25, shadow_frac=0.10, raw=None):
    """Random held-out test set; the rest keeps its time order.  The first
    `shadow_frac` of the stream is an unlabelled shadow window used to fit the
    scaler and the isolation forest; the remainder is cut into T review rounds."""
    X, y, s, cat = raw if raw is not None else raw_task(name)
    rng = np.random.default_rng(seed)
    n = len(y)
    is_test = rng.random(n) < test_frac
    te, st = np.where(is_test)[0], np.where(~is_test)[0]
    n_sh = int(shadow_frac * len(st))
    sh, live = st[:n_sh], st[n_sh:]

    sc = LabelFreeScaler(cat).fit(X[sh])
    Zsh = sc.transform(X[sh])
    iforest = IsolationForest(n_estimators=200, random_state=seed).fit(Zsh)
    a_sh = -iforest.score_samples(Zsh)
    a_mu, a_sd = a_sh.mean(), a_sh.std() + 1e-12

    def feats(idx):
        Z = sc.transform(X[idx])
        a = (-iforest.score_samples(Z) - a_mu) / a_sd
        if s is None:
            sv = np.zeros(len(idx))
        else:
            sv = (s[idx] - s[sh].mean()) / (s[sh].std() + 1e-12)
        return Z, a, sv

    Zl, al, sl = feats(live)
    Zt, at, stt = feats(te)
    return dict(name=name, Z=Zl, a=al, s=sl, y=y[live], Zt=Zt, at=at, st=stt, yt=y[te],
                rounds=np.array_split(np.arange(len(live)), T), has_src=s is not None)


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------
# kind:   static | model | random | oracle
# prior:  which prior scores enter the model as features with prior mean weight 1
# acq:    greedy (posterior mean) | ucb (mu + beta*sigma) | ts | unc (closest to p = 0.5)
POLICIES = {
    "Random":      dict(kind="random"),
    "Anomaly":     dict(kind="static", prior=("a",)),
    "Transfer":    dict(kind="static", prior=("s",)),
    "AL-Unc":      dict(kind="model", prior=(), acq="unc", boot="random"),
    "Boot-Sup":    dict(kind="model", prior=(), acq="greedy", boot="a"),
    "PRIME":       dict(kind="model", prior=("a", "s"), acq="greedy"),
    # ablations
    "PRIME-noA":   dict(kind="model", prior=("s",), acq="greedy"),
    "PRIME-noS":   dict(kind="model", prior=("a",), acq="greedy"),
    "PRIME-UCB":   dict(kind="model", prior=("a", "s"), acq="ucb"),
    "PRIME-TS":    dict(kind="model", prior=("a", "s"), acq="ts"),
    "PRIME-noX":   dict(kind="model", prior=("a", "s"), acq="greedy", use_x=False),
    "PriorSum":    dict(kind="static", prior=("a", "s")),
    "Oracle":      dict(kind="oracle", prior=("a", "s")),
}

DEFAULT_HP = dict(lam_p=1.0, lam_x=1.0, lam_b=0.01, beta=1.0, base_rate=0.01)


def _design(task, prior, use_x, part="live"):
    Z, a, s = (task["Z"], task["a"], task["s"]) if part == "live" else \
              (task["Zt"], task["at"], task["st"])
    cols = [np.ones(len(a))]
    if "a" in prior:
        cols.append(a)
    if "s" in prior and task["has_src"]:
        cols.append(s)
    k = len(cols) - 1
    P = np.column_stack(cols + ([Z] if use_x else []))
    return P, k


def run_stream(task, policy, budget_frac, seed, hp=None, handoff=None, eval_test=True):
    """Simulates the review loop.  Returns per-round arrays.

    handoff: None or dict(theta, pi0, pi1, delta, audit) -- enables the certified
    auto-block region {x : sigmoid(mu - beta*sigma) >= theta}.
    """
    hp = dict(DEFAULT_HP, **(hp or {}))
    cfg = POLICIES[policy]
    rng = np.random.default_rng(10_000 + seed)
    y = task["y"]
    prior = tuple(p for p in cfg.get("prior", ()) if p != "s" or task["has_src"])
    use_x = cfg.get("use_x", True)
    Phi, k = _design(task, prior, use_x)
    Phit, _ = _design(task, prior, use_x, "test")
    d = Phi.shape[1]
    w0 = np.zeros(d)
    w0[0] = np.log(hp["base_rate"] / (1 - hp["base_rate"]))
    w0[1:1 + k] = 1.0
    lam = np.full(d, float(hp["lam_x"]))
    lam[0] = hp["lam_b"]
    lam[1:1 + k] = hp["lam_p"]
    model = BayesLogit(w0, lam) if cfg["kind"] in ("model", "oracle") else None

    R = len(task["rounds"])
    out = dict(caught=np.zeros(R), fraud=np.zeros(R), reviewed=np.zeros(R),
               ap=np.full(R, np.nan), auto=np.zeros(R), auto_tp=np.zeros(R),
               certified=np.zeros(R), region=np.zeros(R), region_tp=np.zeros(R))
    lab_idx, seen = [], []
    ep = EProcess(handoff["pi0"], handoff["pi1"], handoff["delta"]) if handoff else None

    for r, idx in enumerate(task["rounds"]):
        B = max(1, int(round(budget_frac * len(idx))))
        nlab = len(lab_idx)
        have_both = nlab > 0 and 0 < y[lab_idx].sum() < nlab
        mu = sg = None
        # ---------------- scores for the arriving batch ----------------
        if cfg["kind"] == "random":
            score = rng.random(len(idx))
        elif cfg["kind"] == "static":
            score = Phi[idx, 1:1 + k].sum(1) if k else rng.random(len(idx))
        else:
            mu = model.mean(Phi[idx])
            if cfg["kind"] == "oracle":
                score = mu
            elif cfg["acq"] == "ucb":
                sg = model.std(Phi[idx])
                score = mu + hp["beta"] * sg
            elif cfg["acq"] == "greedy":
                score = mu
            elif cfg["acq"] == "ts":                       # Thompson: one posterior draw
                score = Phi[idx] @ model.sample(rng)
            else:
                score = -np.abs(mu)
            if k == 0 and not have_both:                   # no prior, no usable model yet
                score = task["a"][idx] if cfg.get("boot") == "a" else rng.random(len(idx))
        if eval_test and model is not None:
            out["ap"][r] = average_precision_score(task["yt"], model.mean(Phit))
        elif eval_test and cfg["kind"] == "static" and k:
            out["ap"][r] = average_precision_score(task["yt"], Phit[:, 1:1 + k].sum(1))

        score = score + 1e-9 * rng.random(len(idx))        # random tie-break
        order = np.argsort(-score)
        # ---------------- certified auto-block region ----------------
        auto_local = np.array([], dtype=int)
        audit_local = np.array([], dtype=int)
        if handoff is not None:
            if sg is None:
                sg = model.std(Phi[idx])
            lcb = expit(mu - hp["beta"] * sg)
            region = np.where(lcb >= handoff["theta"])[0]
            out["region"][r], out["region_tp"][r] = len(region), y[idx[region]].sum()
            if ep.certified and len(region):
                n_a = min(B, int(np.ceil(handoff["audit"] * len(region))))
                audit_local = rng.permutation(region)[:n_a]
                auto_local = np.setdiff1d(region, audit_local)
            elif len(region):                              # before handoff: audit first,
                audit_local = rng.permutation(region)[:B]  # in uniformly random order
        taken = set(audit_local.tolist()) | set(auto_local.tolist())
        rest = [j for j in order if j not in taken][: B - len(audit_local)]
        rev_local = np.concatenate([audit_local, np.array(rest, dtype=int)]).astype(int)
        rev = idx[rev_local]
        # ---------------- outcomes ----------------
        out["fraud"][r] = y[idx].sum()
        out["caught"][r] = y[rev].sum()
        out["reviewed"][r] = len(rev)
        out["auto"][r] = len(auto_local)
        out["auto_tp"][r] = y[idx[auto_local]].sum() if len(auto_local) else 0
        if ep is not None:
            out["certified"][r] = float(ep.certified)     # status used in this round
            if len(audit_local):
                ep.update(y[idx[audit_local]])
        # ---------------- learn from analyst labels ----------------
        lab_idx.extend(rev.tolist())
        seen.extend(idx.tolist())
        if cfg["kind"] == "model":
            model.fit(Phi[lab_idx], y[lab_idx])
        elif cfg["kind"] == "oracle":                      # label of every past transaction
            model.fit(Phi[seen], y[seen])
    return out
