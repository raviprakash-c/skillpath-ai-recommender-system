import streamlit as st
import pandas as pd

from src.config import ROLE_SKILLS, ALL_SKILLS
from src.data_loader import load_data, parse_set
from src.hybrid import recommend, explain
from src.evaluation import precision_recall_ndcg_at_k, svd_rmse

st.set_page_config(page_title="SkillPath AI", page_icon="🎯", layout="wide")

@st.cache_data
def get_data():
    return load_data()

students, resources, ratings = get_data()

st.title("🎯 SkillPath AI")
st.caption("Explainable Hybrid Career, Skill-Gap & Learning Resource Recommender")

page = st.sidebar.radio("Navigate", ["Recommend", "How It Works", "Evaluation"])

if page == "Recommend":
    mode = st.radio("Profile mode", ["Existing student", "New student (cold start)"], horizontal=True)

    user_id = None
    if mode == "Existing student":
        user_id = st.selectbox("Choose student", students["student_id"].tolist())
        profile = students.loc[students["student_id"] == user_id].iloc[0]
        default_role = profile["target_role"]
        skills = parse_set(profile["skills"])
        interests = parse_set(profile["interests"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Student", user_id)
        c2.metric("Target role", default_role)
        c3.metric("Ratings available", int((ratings["student_id"] == user_id).sum()))

        st.write("**Known skills:**", ", ".join(sorted(skills)))
        st.write("**Interests:**", ", ".join(sorted(interests)))
        target_role = st.selectbox("Target role", list(ROLE_SKILLS.keys()),
                                   index=list(ROLE_SKILLS.keys()).index(default_role))
    else:
        target_role = st.selectbox("Target role", list(ROLE_SKILLS.keys()))
        skills = set(st.multiselect("Skills you already know", ALL_SKILLS))
        interests = set(st.multiselect(
            "Interests",
            ["Data", "Backend", "Cloud", "AI", "Web", "Cybersecurity", "DevOps", "Analytics"]
        ))

    metric = st.selectbox("User similarity method", ["cosine", "pearson"],
                          help="Used only when an existing student has rating history.")
    top_k = st.slider("Number of recommendations", 5, 15, 8)

    if st.button("Generate recommendations", type="primary"):
        recs, missing, weights = recommend(
            resources, ratings, skills, interests, target_role,
            user_id=user_id, user_metric=metric, top_k=top_k
        )

        st.subheader("Skill-gap analysis")
        required = ROLE_SKILLS[target_role]
        st.write("**Required skills:**", ", ".join(required))
        st.write("**Missing skills:**", ", ".join(missing) if missing else "No major gap detected.")

        st.subheader("Recommended next steps")
        for rank, rec in enumerate(recs, start=1):
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"### {rank}. {rec['title']}")
                    st.write(f"**Type:** {rec['type']}  |  **Domain:** {rec['domain']}  |  **Difficulty:** {rec['difficulty']}")
                    st.write(f"**Skills:** {rec['skills'].replace('|', ', ')}")
                    st.write(explain(rec, missing))
                with right:
                    st.metric("Hybrid score", f"{rec['final_score']*100:.1f}%")

                with st.expander("See recommender scores"):
                    detail = pd.DataFrame({
                        "Component": ["Content", "User-CF", "Item-CF", "SVD", "Skill gap", "Popularity"],
                        "Score": [
                            rec["content_score"], rec["user_cf_score"], rec["item_cf_score"],
                            rec["svd_score"], rec["skill_gap_score"], rec["popularity_score"]
                        ],
                    })
                    st.dataframe(detail, hide_index=True, use_container_width=True)

        with st.expander("Hybrid weights used"):
            st.json(weights)

elif page == "How It Works":
    st.header("Recommender system pipeline")
    st.markdown("""
1. **Profile modelling:** skills, interests and target role.
2. **TF-IDF + Cosine Similarity:** matches profile text with resource text.
3. **Jaccard Similarity:** compares skill sets.
4. **User-based Collaborative Filtering:** finds students with similar rating behaviour.
5. **Pearson or Cosine User Similarity:** selectable for comparison.
6. **Item-based Collaborative Filtering:** finds resources similar to items the student liked.
7. **Matrix Factorization (Truncated SVD):** learns latent student-resource preferences.
8. **Knowledge / Skill-gap score:** promotes resources that teach missing target-role skills.
9. **Popularity fallback:** helps with cold-start users.
10. **Hybrid ranking:** weighted combination of all available scores.
11. **Explainability:** tells the user why each result was recommended.
12. **Top-K evaluation:** Precision@K, Recall@K and NDCG@K can be used to evaluate ranking quality.
""")

    st.subheader("Hybrid formula")
    st.latex(r"FinalScore = \sum_m w_m \times Score_m")
    st.info("For cold-start users, collaborative weights become zero and the app relies more on content, skill-gap and popularity.")

elif page == "Evaluation":
    st.header("Evaluation demo")

    rmse = svd_rmse(ratings)
    if rmse is not None:
        st.metric("SVD RMSE (demo split)", f"{rmse:.3f}")

    st.markdown("""
**Offline evaluation plan for your report**

- Treat ratings **4 or 5** as relevant.
- Hold out one or more relevant resources per student.
- Generate Top-K recommendations using only training ratings.
- Compute **Precision@K**, **Recall@K**, and **NDCG@K**.
- Compute **RMSE** for rating prediction.
- Compare:
  - Content-only
  - User-CF
  - Item-CF
  - SVD
  - Hybrid model
""")

    example_recs = ["R01", "R02", "R03", "R04", "R05"]
    example_rel = {"R02", "R05", "R09"}
    p, r, n = precision_recall_ndcg_at_k(example_recs, example_rel, 5)
    st.write("**Metric function example:**")
    st.write({"Precision@5": round(p, 3), "Recall@5": round(r, 3), "NDCG@5": round(n, 3)})
