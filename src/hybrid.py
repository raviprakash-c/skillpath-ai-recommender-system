from .config import ROLE_SKILLS
from .data_loader import parse_set
from .content_recommender import content_scores
from .collaborative_recommender import user_cf_scores, item_cf_scores, svd_scores

def _skill_gap_scores(resources, skills, target_role):
    required = set(ROLE_SKILLS.get(target_role, []))
    known = set(skills)
    missing = required - known

    out = {}
    for _, row in resources.iterrows():
        rskills = parse_set(row["skills"])
        if missing:
            score = len(rskills & missing) / len(missing)
        elif required:
            score = len(rskills & required) / len(required)
        else:
            score = 0.0
        out[row["resource_id"]] = score
    return out, sorted(missing)

def _popularity_scores(resources):
    maxp = max(float(resources["popularity"].max()), 1.0)
    return {r["resource_id"]: float(r["popularity"]) / maxp for _, r in resources.iterrows()}

def recommend(resources, ratings, skills, interests, target_role, user_id=None,
              user_metric="cosine", top_k=10):
    c = content_scores(resources, skills, interests, target_role)
    gap, missing = _skill_gap_scores(resources, skills, target_role)
    pop = _popularity_scores(resources)

    ucf = user_cf_scores(ratings, user_id, user_metric) if user_id else {}
    icf = item_cf_scores(ratings, user_id) if user_id else {}
    svd = svd_scores(ratings, user_id) if user_id else {}

    if user_id and (ucf or icf or svd):
        weights = {
            "content": 0.28,
            "user_cf": 0.18,
            "item_cf": 0.12,
            "svd": 0.15,
            "skill_gap": 0.20,
            "popularity": 0.07,
        }
    else:
        weights = {
            "content": 0.48,
            "user_cf": 0.0,
            "item_cf": 0.0,
            "svd": 0.0,
            "skill_gap": 0.37,
            "popularity": 0.15,
        }

    seen = set()
    if user_id:
        seen = set(ratings.loc[ratings["student_id"] == user_id, "resource_id"])

    rows = []
    for _, row in resources.iterrows():
        rid = row["resource_id"]
        if rid in seen:
            continue

        components = {
            "content": c.get(rid, 0.0),
            "user_cf": ucf.get(rid, 0.0),
            "item_cf": icf.get(rid, 0.0),
            "svd": svd.get(rid, 0.0),
            "skill_gap": gap.get(rid, 0.0),
            "popularity": pop.get(rid, 0.0),
        }
        final = sum(weights[k] * components[k] for k in weights)
        rows.append({
            **row.to_dict(),
            **{f"{k}_score": round(v, 4) for k, v in components.items()},
            "final_score": round(final, 4),
        })

    rows.sort(key=lambda x: x["final_score"], reverse=True)
    return rows[:top_k], missing, weights

def explain(rec, missing_skills):
    rskills = parse_set(rec["skills"])
    covered = sorted(rskills & set(missing_skills))
    reasons = []
    if covered:
        reasons.append("covers missing skill(s): " + ", ".join(covered[:3]))
    if rec["content_score"] >= 0.15:
        reasons.append("matches your profile and target role")
    if rec["user_cf_score"] >= 0.60:
        reasons.append("students with similar ratings preferred it")
    if rec["item_cf_score"] >= 0.60:
        reasons.append("similar to resources you rated highly")
    if rec["svd_score"] >= 0.60:
        reasons.append("latent-factor model predicts a strong preference")
    if not reasons:
        reasons.append("has useful relevance and popularity for your selected path")
    return "Recommended because it " + "; ".join(reasons) + "."
