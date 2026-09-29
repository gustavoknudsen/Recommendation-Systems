from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud"
URLS = {
    "ratings": f"{BASE_URL}/IBMSkillsNetwork-ML0321EN-Coursera/labs/v2/module_3/ratings.csv",
    "course_genre": f"{BASE_URL}/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv",
    "course_processed": f"{BASE_URL}/IBM-ML321EN-SkillsNetwork/labs/datasets/course_processed.csv",
}

RATING_MIN = 3
RATING_MAX = 5

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load(name):
    """Load a dataset by name, downloading it to data/ on first use."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        pd.read_csv(URLS[name]).to_csv(path, index=False)
    return pd.read_csv(path)


def load_ratings():
    return load("ratings")


def load_course_genres():
    return load("course_genre")


def load_course_text():
    return load("course_processed")


class Index:
    """Fixed integer ids for users and courses, shared by every model."""

    def __init__(self, ratings):
        self.users = np.sort(ratings["user"].unique())
        self.items = np.sort(ratings["item"].unique())
        self.user_to_idx = pd.Series(np.arange(len(self.users)), index=self.users)
        self.item_to_idx = pd.Series(np.arange(len(self.items)), index=self.items)

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_items(self):
        return len(self.items)

    def encode(self, df):
        """Return (user_idx, item_idx) arrays for a ratings-like DataFrame."""
        return (
            self.user_to_idx.loc[df["user"]].to_numpy(),
            self.item_to_idx.loc[df["item"]].to_numpy(),
        )
