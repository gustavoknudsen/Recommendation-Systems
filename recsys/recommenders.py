import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


def popularity_scores(train_matrix, n_rows):
    popularity = np.asarray(train_matrix.sum(axis=0)).ravel()
    return np.tile(popularity, (n_rows, 1))


def item_cosine_similarity(train_matrix):
    co = (train_matrix.T @ train_matrix).toarray().astype(float)
    norms = np.sqrt(np.diag(co))
    sim = co / (np.outer(norms, norms) + 1e-9)
    np.fill_diagonal(sim, 0.0)
    return sim


def item_knn_scores(train_matrix, user_rows, similarity, k=None):
    """Sum of similarities between each candidate item and the user's enrolled items.

    If k is set, each item only keeps its k most similar neighbours.
    """
    sim = similarity.copy()
    if k is not None and k < sim.shape[1]:
        cutoff = -np.sort(-sim, axis=1)[:, k - 1 : k]
        sim[sim < cutoff] = 0.0
    return np.asarray(train_matrix[user_rows] @ sim)


def profile_scores(train_matrix, user_rows, item_features):
    """Content score: user profile (sum of enrolled items' features) dot each item's features."""
    profiles = np.asarray(train_matrix[user_rows] @ item_features)
    return profiles @ item_features.T


def similarity_scores(train_matrix, user_rows, item_similarity):
    """Content score from an item-item similarity matrix built on course content."""
    return np.asarray(train_matrix[user_rows] @ item_similarity)


def minmax_rows(scores):
    """Scale each row to [0, 1] so scores from different models can be blended."""
    low = scores.min(axis=1, keepdims=True)
    high = scores.max(axis=1, keepdims=True)
    return (scores - low) / np.where(high > low, high - low, 1.0)


def user_knn_scores(train_matrix, user_rows, k=50, chunk=1000):
    """Sum of the enrolments of each user's k most similar users (cosine)."""
    norms = np.sqrt(np.asarray(train_matrix.multiply(train_matrix).sum(axis=1)).ravel()) + 1e-9
    normed = sp.diags(1.0 / norms) @ train_matrix
    scores = np.zeros((len(user_rows), train_matrix.shape[1]))
    for start in range(0, len(user_rows), chunk):
        rows = user_rows[start : start + chunk]
        sim = (normed[rows] @ normed.T).toarray().astype(np.float32)
        sim[np.arange(len(rows)), rows] = 0.0
        if k < sim.shape[1]:
            drop = np.argpartition(-sim, k, axis=1)[:, k:]
            np.put_along_axis(sim, drop, 0.0, axis=1)
        scores[start : start + len(rows)] = (train_matrix.T @ sim.T).T
    return scores


def pure_svd_scores(train_matrix, user_rows, rank, seed=123):
    """Truncated SVD of the binary enrolment matrix, a standard top-N baseline."""
    svd = TruncatedSVD(n_components=rank, random_state=seed).fit(train_matrix)
    return svd.transform(train_matrix[user_rows]) @ svd.components_
