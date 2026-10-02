import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

def precision_recall_ndcg_at_k(recommended_ids, relevant_ids, k=5):
    recs = list(recommended_ids)[:k]
    relevant = set(relevant_ids)
    if not recs:
        return 0.0, 0.0, 0.0

    hits = [1 if rid in relevant else 0 for rid in recs]
    precision = sum(hits) / k
    recall = sum(hits) / len(relevant) if relevant else 0.0

    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal_hits = [1] * min(len(relevant), k)
    idcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0
    return precision, recall, ndcg

def svd_rmse(ratings, test_fraction=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    mask = rng.random(len(ratings)) >= test_fraction
    train = ratings.loc[mask].copy()
    test = ratings.loc[~mask].copy()

    pivot = train.pivot_table(index="student_id", columns="resource_id", values="rating", aggfunc="mean").fillna(0.0)
    if min(pivot.shape) < 3:
        return None

    n_components = max(2, min(8, pivot.shape[0]-1, pivot.shape[1]-1))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    latent = svd.fit_transform(pivot.values)
    reconstructed = latent @ svd.components_

    preds, actuals = [], []
    for _, row in test.iterrows():
        u, i = row["student_id"], row["resource_id"]
        if u in pivot.index and i in pivot.columns:
            pred = reconstructed[pivot.index.get_loc(u), pivot.columns.get_loc(i)]
            pred = float(np.clip(pred, 1.0, 5.0))
            preds.append(pred)
            actuals.append(float(row["rating"]))
    if not preds:
        return None
    return float(np.sqrt(mean_squared_error(actuals, preds)))
