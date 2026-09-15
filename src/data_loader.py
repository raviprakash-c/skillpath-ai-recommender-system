from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_students():
    return pd.read_csv(DATA_DIR / "students.csv")


def load_resources():
    return pd.read_csv(DATA_DIR / "resources.csv")


def load_ratings():
    return pd.read_csv(DATA_DIR / "ratings.csv")


def load_data():
    students = load_students()
    resources = load_resources()
    ratings = load_ratings()

    return students, resources, ratings


def parse_set(value):
    if pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    }