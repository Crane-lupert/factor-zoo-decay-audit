"""Inter-rater agreement metrics for mechanism classification.

Cohen's kappa for two raters with discrete labels:
    kappa = (p_o - p_e) / (1 - p_e)
where p_o is observed agreement and p_e is chance agreement assuming
independent marginal distributions.

Fleiss' kappa for >2 raters generalizes p_e using overall label proportions.
We use Cohen's pairwise + overall majority-vs-rater Cohen.

CLAUDE.md Day-2 advance gate: oracle kappa > 0.7.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def cohen_kappa(rater_a: list[str], rater_b: list[str]) -> tuple[float, int]:
    """Cohen's kappa on overlapping non-null labels. Returns (kappa, n_overlap)."""
    pairs = [(a, b) for a, b in zip(rater_a, rater_b) if a and b and a != "PARSE_FAIL" and b != "PARSE_FAIL"]
    n = len(pairs)
    if n == 0:
        return float("nan"), 0
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    obs = np.zeros((k, k), dtype=float)
    for a, b in pairs:
        obs[idx[a], idx[b]] += 1
    obs /= n
    p_o = float(obs.trace())
    pa = obs.sum(axis=1)
    pb = obs.sum(axis=0)
    p_e = float((pa * pb).sum())
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else float("nan"), n
    return (p_o - p_e) / (1.0 - p_e), n


def pairwise_kappa_table(per_model: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """per_model: long-form DataFrame with columns [acronym, model, label].
    Returns a square DataFrame indexed by model with pairwise kappa."""
    pivot = per_model.pivot_table(index="acronym", columns="model", values=label_col, aggfunc="first")
    models = list(pivot.columns)
    out = pd.DataFrame(index=models, columns=models, dtype=float)
    for m1 in models:
        for m2 in models:
            if m1 == m2:
                out.loc[m1, m2] = 1.0
            else:
                k, _ = cohen_kappa(pivot[m1].tolist(), pivot[m2].tolist())
                out.loc[m1, m2] = k
    return out


def majority_vote(per_model: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Per-acronym majority label across models. Returns DataFrame with
    acronym, ensemble_label, votes_for, total_votes, unanimous (bool)."""
    rows = []
    for acr, grp in per_model.groupby("acronym"):
        labs = [x for x in grp[label_col].tolist() if x != "PARSE_FAIL"]
        if not labs:
            rows.append({"acronym": acr, "ensemble_label": "PARSE_FAIL", "votes_for": 0,
                         "total_votes": len(grp), "unanimous": False})
            continue
        c = Counter(labs)
        top, votes = c.most_common(1)[0]
        rows.append({
            "acronym": acr, "ensemble_label": top, "votes_for": votes,
            "total_votes": len(labs), "unanimous": votes == len(labs),
        })
    return pd.DataFrame(rows)


def oracle_kappa(majority: pd.DataFrame, oracle: pd.DataFrame) -> tuple[float, int, pd.DataFrame]:
    """Cohen's kappa between ensemble majority and hand oracle. Returns
    (kappa, n_overlap, joined_table_with_match_column)."""
    j = oracle.merge(
        majority[["acronym", "ensemble_label", "votes_for", "total_votes"]],
        left_on="Acronym", right_on="acronym", how="inner",
    )
    j["match"] = j["oracle_label"] == j["ensemble_label"]
    k, n = cohen_kappa(j["oracle_label"].tolist(), j["ensemble_label"].tolist())
    return k, n, j


def label_distribution(per_model: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Counts of each label per model -- sanity check on whether a model is
    biased toward one bucket."""
    return (
        per_model.groupby(["model", label_col])
        .size()
        .unstack(fill_value=0)
        .assign(total=lambda d: d.sum(axis=1))
    )
