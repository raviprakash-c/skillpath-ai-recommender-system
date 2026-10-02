# SkillPath AI

SkillPath AI is an explainable hybrid recommender system for personalized career paths, skill-gap analysis, and learning-resource recommendations.

## Recommender concepts demonstrated

- **Content-based filtering:** TF-IDF and cosine similarity over resource/profile text.
- **Jaccard similarity:** compares user skill sets with resource skills.
- **User-based collaborative filtering:** recommends from students with similar rating behavior.
- **User similarity variants:** cosine similarity and Pearson correlation.
- **Item-based collaborative filtering:** uses similarity between learning resources.
- **Matrix factorization:** Truncated SVD learns latent student-resource preference patterns.
- **Knowledge-based recommendation:** promotes resources that cover missing skills for a target career role.
- **Popularity-based recommendation:** provides a fallback signal for cold-start users.
- **Hybrid recommendation:** combines multiple recommender signals using weighted ranking.
- **Explainability:** reports why each resource was recommended.
- **Evaluation:** Precision@K, Recall@K, NDCG@K and SVD RMSE utilities.

## Application flow

```text
Student profile
    |
    +--> Content model -----------+
    |    TF-IDF + Cosine          |
    |    Jaccard                  |
    |                             |
    +--> Collaborative models ----+--> Hybrid weighted ranking --> Top-K resources
    |    User-CF                  |
    |    Item-CF                  |
    |    SVD                      |
    |                             |
    +--> Skill-gap model ---------+
    |                             |
    +--> Popularity / cold start -+
                                  |
                                  +--> Explanation
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The repository already contains the generated demo CSV data. To regenerate it deterministically:

```bash
python generate_data.py
```

Run the application:

```bash
streamlit run app.py
```

## Main modules

- `app.py` — Streamlit interface, recommendation workflow, cold-start mode and evaluation demo.
- `src/config.py` — career roles and required skills.
- `src/data_loader.py` — CSV loading and set parsing.
- `src/content_recommender.py` — TF-IDF, cosine similarity and Jaccard similarity.
- `src/collaborative_recommender.py` — user-CF, item-CF and Truncated SVD.
- `src/hybrid.py` — skill-gap scoring, popularity, hybrid ranking and explanations.
- `src/evaluation.py` — ranking metrics and SVD RMSE.
- `generate_data.py` — reproducible synthetic student, resource and rating data.

## Demo

1. Select an existing student to use rating history.
2. Inspect the student's known skills and target career role.
3. Generate Top-K learning recommendations.
4. Open **See recommender scores** to inspect each component's contribution.
5. Switch between cosine and Pearson user similarity.
6. Select **New student (cold start)** to see the non-collaborative fallback strategy.
7. Use **How It Works** to inspect the recommender pipeline and hybrid formula.
8. Use **Evaluation** to demonstrate Precision@K, Recall@K, NDCG and SVD RMSE.

## Project purpose

The application demonstrates how several recommender-system approaches can be combined in one practical educational/career application instead of treating each algorithm as an isolated example. The hybrid layer combines behavioral, content, career-skill and popularity signals so the same application can handle both existing users and users without rating history.
