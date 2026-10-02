import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

def _normalize_dict(d):
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return {k: 0.5 for k in d}
    return {k: float((v - lo) / (hi - lo)) for k, v in d.items()}

def user_cf_scores(ratings, user_id, metric="cosine"):
    pivot = ratings.pivot_table(index="student_id", columns="resource_id", values="rating", aggfunc="mean")
    if user_id not in pivot.index:
        return {}

    target = pivot.loc[user_id]
    seen = set(target.dropna().index)

    if metric == "pearson":
        sims = pivot.T.corr(method="pearson", min_periods=2).loc[user_id].fillna(0.0)
    else:
        filled = pivot.fillna(0.0)
        sim_matrix = cosine_similarity(filled.values)
        sims = pd.Series(sim_matrix[pivot.index.get_loc(user_id)], index=pivot.index)

    sims = sims.drop(index=user_id, errors="ignore")
    sims = sims[sims > 0].sort_values(ascending=False).head(25)
    scores = {}

    for item in pivot.columns:
        if item in seen:
            continue
        vals, weights = [], []
        for neighbor, sim in sims.items():
            r = pivot.at[neighbor, item]
            if pd.notna(r):
                vals.append(float(r))
                weights.append(float(sim))
        if weights and sum(weights) > 0:
            scores[item] = float(np.average(vals, weights=weights) / 5.0)
    return scores

def item_cf_scores(ratings, user_id):
    pivot = ratings.pivot_table(index="student_id", columns="resource_id", values="rating", aggfunc="mean")
    if user_id not in pivot.index:
        return {}

    filled = pivot.fillna(0.0)
    item_sim = cosine_similarity(filled.T.values)
    item_ids = list(filled.columns)
    sim_df = pd.DataFrame(item_sim, index=item_ids, columns=item_ids)

    target = pivot.loc[user_id]
    liked = target[target >= 4.0].index.tolist()
    seen = set(target.dropna().index)
    if not liked:
        return {}

    scores = {}
    for item in item_ids:
        if item in seen:
            continue
        scores[item] = float(sim_df.loc[item, liked].mean())
    return _normalize_dict(scores)

def svd_scores(ratings, user_id):
    pivot = ratings.pivot_table(index="student_id", columns="resource_id", values="rating", aggfunc="mean").fillna(0.0)
    if user_id not in pivot.index or min(pivot.shape) < 3:
        return {}

    n_components = max(2, min(8, pivot.shape[0] - 1, pivot.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_latent = svd.fit_transform(pivot.values)
    reconstructed = user_latent @ svd.components_

    row_idx = pivot.index.get_loc(user_id)
    raw = dict(zip(pivot.columns, reconstructed[row_idx]))
    seen = set(ratings.loc[ratings["student_id"] == user_id, "resource_id"])
    raw = {k: v for k, v in raw.items() if k not in seen}
    return _normalize_dict(raw)
