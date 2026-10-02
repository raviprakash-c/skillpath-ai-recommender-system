from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def load_data():
    students = pd.read_csv(DATA_DIR / "students.csv")
    resources = pd.read_csv(DATA_DIR / "resources.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    return students, resources, ratings

def parse_set(value):
    if pd.isna(value) or not str(value).strip():
        return set()
    return {x.strip() for x in str(value).split("|") if x.strip()}
