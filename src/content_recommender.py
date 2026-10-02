import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .data_loader import parse_set

def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0

def content_scores(resources, skills, interests, target_role):
    df = resources.copy()
    resource_text = (
        df["title"].fillna("") + " " +
        df["domain"].fillna("") + " " +
        df["skills"].fillna("").str.replace("|", " ", regex=False) + " " +
        df["description"].fillna("")
    )
    query = " ".join(list(skills) + list(interests) + [target_role])

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(resource_text.tolist() + [query])
    tfidf_scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()

    user_terms = set(skills) | set(interests) | {target_role}
    jac = np.array([
        jaccard(user_terms, parse_set(row["skills"]) | {row["domain"]})
        for _, row in df.iterrows()
    ])

    combined = 0.75 * tfidf_scores + 0.25 * jac
    return dict(zip(df["resource_id"], combined))
