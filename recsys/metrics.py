import numpy as np


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def ranking_metrics(scores, seen, targets, k=10):
    """HitRate@k and NDCG@k with one held-out item per user.

    scores: (n_users, n_items) array, higher is better.
    seen: (n_users, n_items) boolean array of training enrolments, excluded from ranking.
    targets: (n_users,) index of the held-out item.
    """
    scores = np.array(scores, dtype=float)
    scores[np.asarray(seen, dtype=bool)] = -np.inf
    target_scores = scores[np.arange(len(targets)), targets]
    # Rank is 1 + number of candidates scoring strictly higher. Ties count against the model.
    rank = 1 + (scores > target_scores[:, None]).sum(axis=1) + (
        (scores == target_scores[:, None]).sum(axis=1) - 1
    )
    hit = rank <= k
    ndcg = np.where(hit, 1.0 / np.log2(rank + 1), 0.0)
    return {"hit_rate@10": float(hit.mean()), "ndcg@10": float(ndcg.mean())}
