import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

SEED = 123


def rating_split(ratings, test_size=0.2, seed=SEED):
    """Random split over interactions, used for rating prediction (RMSE)."""
    return train_test_split(ratings, test_size=test_size, random_state=seed)


def leave_one_out(ratings, seed=SEED, min_interactions=2):
    """Hold out one random enrolment per user who has at least `min_interactions`.

    Users with fewer enrolments stay entirely in train and are not evaluated.
    """
    counts = ratings.groupby("user")["user"].transform("size")
    eligible = ratings[counts >= min_interactions]
    rng = np.random.default_rng(seed)
    shuffled = eligible.iloc[rng.permutation(len(eligible))]
    test = shuffled.drop_duplicates("user").sort_values("user")
    train = ratings.drop(test.index)
    return train, test


def interaction_matrix(df, index):
    """Binary user x item enrolment matrix (CSR) over the full index."""
    users, items = index.encode(df)
    data = np.ones(len(df), dtype=np.float32)
    return sp.csr_matrix((data, (users, items)), shape=(index.n_users, index.n_items))
