from src.config import ROLE_SKILLS
from src.data_loader import parse_set


def get_missing_skills(
    known_skills,
    target_role
):

    required_skills = set(
        ROLE_SKILLS.get(
            target_role,
            []
        )
    )

    known_skills = set(
        known_skills
    )

    missing_skills = (
        required_skills
        - known_skills
    )

    return sorted(
        missing_skills
    )


def calculate_skill_gap_scores(
    resources,
    known_skills,
    target_role
):

    required_skills = set(
        ROLE_SKILLS.get(
            target_role,
            []
        )
    )

    known_skills = set(
        known_skills
    )

    missing_skills = (
        required_skills
        - known_skills
    )

    scores = {}

    for _, resource in resources.iterrows():

        resource_skills = parse_set(
            resource["skills"]
        )

        if missing_skills:

            score = (
                len(
                    resource_skills
                    & missing_skills
                )
                / len(missing_skills)
            )

        else:

            score = 0.0

        scores[
            resource["resource_id"]
        ] = score

    return scores