"""Data loading and label-free preprocessing for the cold-start experiments.

Two public, real-world datasets are used:
  * ULB credit-card fraud (284,807 European card transactions, 492 frauds).
  * IEEE-CIS / Vesta e-commerce fraud (590,540 transactions, five products).
Nothing in this file touches a label of the target (cold-start) domain.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "..", "cache")

IEEE_NUM = (["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2",
             "dist1", "dist2"] + ["C%d" % i for i in range(1, 15)] +
            ["D%d" % i for i in range(1, 16)])
IEEE_CAT = ["card4", "card6", "P_emaildomain", "R_emaildomain"] + ["M%d" % i for i in range(1, 10)]
# a spread of the Vesta engineered columns (one or two per missingness block)
IEEE_V = ["V%d" % i for i in (12, 13, 36, 45, 54, 62, 70, 75, 83, 87, 94, 100, 127, 130,
                              143, 156, 165, 187, 201, 220, 258, 264, 283, 294, 308, 317)]


def load_ulb():
    """Returns time-ordered features X (V1..V28, log amount), labels y, amounts."""
    d = pd.read_csv(os.path.join(DATA, "creditcard.csv")).sort_values("Time", kind="stable")
    X = d[["V%d" % i for i in range(1, 29)]].to_numpy(dtype=np.float64)
    X = np.column_stack([X, np.log1p(d["Amount"].to_numpy())])
    return X, d["Class"].to_numpy().astype(np.int8), d["Amount"].to_numpy()


def load_ieee():
    """Returns a time-ordered frame with numeric features (NaN kept), label, product."""
    pq = os.path.join(CACHE, "ieee.pkl")
    if os.path.exists(pq):
        return pd.read_pickle(pq)
    cols = ["isFraud", "ProductCD", "TransactionDT"] + IEEE_NUM + IEEE_CAT + IEEE_V
    d = pd.read_csv(os.path.join(DATA, "train_transaction.csv"), usecols=cols)
    d = d.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    for c in IEEE_CAT:  # label-free integer codes; NaN stays NaN
        codes = d[c].astype("category").cat.codes.astype(np.float64)
        codes[codes < 0] = np.nan
        d[c] = codes
    for c in ["TransactionAmt", "dist1", "dist2"] + ["C%d" % i for i in range(1, 15)]:
        d[c] = np.log1p(d[c].clip(lower=0))
    os.makedirs(CACHE, exist_ok=True)
    d.to_pickle(pq)
    return d


def ieee_features():
    return IEEE_NUM + IEEE_CAT + IEEE_V


class LabelFreeScaler:
    """Median imputation, missing indicators, one-hot of small categoricals,
    standardisation and clipping. Statistics come from unlabelled rows only."""

    def __init__(self, cat_idx=(), max_levels=6, clip=5.0):
        self.cat_idx, self.max_levels, self.clip = list(cat_idx), max_levels, clip

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.num_idx = [j for j in range(X.shape[1]) if j not in self.cat_idx]
        with np.errstate(all="ignore"):
            self.med = np.nan_to_num(np.nanmedian(X[:, self.num_idx], axis=0))
        miss = np.isnan(X).mean(0)
        self.ind_idx = [j for j in self.num_idx if 0.02 < miss[j] < 0.98]
        self.levels = {}
        for j in self.cat_idx:
            v = X[:, j]
            v = v[~np.isnan(v)]
            if v.size:
                u, c = np.unique(v, return_counts=True)
                self.levels[j] = u[np.argsort(-c)][: self.max_levels]
            else:
                self.levels[j] = np.array([])
        Z = self._raw(X)
        self.mu, self.sd = Z.mean(0), Z.std(0)
        self.keep = self.sd > 1e-8
        return self

    def _raw(self, X):
        X = np.asarray(X, dtype=np.float64)
        A = X[:, self.num_idx].copy()
        m = np.isnan(A)
        A[m] = np.take(self.med, np.where(m)[1])
        parts = [A]
        if self.ind_idx:
            parts.append(np.isnan(X[:, self.ind_idx]).astype(np.float64))
        for j in self.cat_idx:
            for lv in self.levels[j]:
                parts.append((X[:, j] == lv).astype(np.float64)[:, None])
            parts.append(np.isnan(X[:, j]).astype(np.float64)[:, None])
        return np.column_stack(parts)

    def transform(self, X):
        Z = self._raw(X)[:, self.keep]
        Z = (Z - self.mu[self.keep]) / self.sd[self.keep]
        return np.clip(Z, -self.clip, self.clip)
