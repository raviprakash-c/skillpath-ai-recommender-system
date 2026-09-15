import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import parse_set



def jaccard_similarity(set_a, set_b):
    set_a = set(set_a)
    set_b = set(set_b)

    union = set_a | set_b

    if not union:
        return 0.0

    intersection = set_a & set_b

    return len(intersection) / len(union)



def calculate_content_scores(
    resources,
    user_skills,
    user_interests,
    target_role
):

    df = resources.copy()

    resource_text = (
        df["title"].fillna("")
        + " "
        + df["domain"].fillna("")
        + " "
        + df["skills"].fillna("").str.replace(
            "|",
            " ",
            regex=False
        )
        + " "
        + df["description"].fillna("")
    )

    user_profile = " ".join(
        list(user_skills)
        + list(user_interests)
        + [target_role]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        resource_text.tolist()
        + [user_profile]
    )

    cosine_scores = cosine_similarity(
        tfidf_matrix[-1],
        tfidf_matrix[:-1]
    ).flatten()

    jaccard_scores = []

    user_terms = (
        set(user_skills)
        | set(user_interests)
        | {target_role}
    )

    for _, row in df.iterrows():

        resource_terms = (
            parse_set(row["skills"])
            | {row["domain"]}
        )

        score = jaccard_similarity(
            user_terms,
            resource_terms
        )

        jaccard_scores.append(score)

    jaccard_scores = np.array(
        jaccard_scores
    )

    final_scores = (
        0.75 * cosine_scores
        + 0.25 * jaccard_scores
    )

    return dict(
        zip(
            df["resource_id"],
            final_scores
        )
    )