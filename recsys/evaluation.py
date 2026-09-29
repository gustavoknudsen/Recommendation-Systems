from recsys import metrics, splits


def evaluate_ranking(score_fn, fit_df, eval_df, index):
    """Fit on `fit_df` enrolments and score each `eval_df` user's held-out course.

    score_fn(train_matrix, user_rows) must return an (n_eval_users, n_items) score array.
    """
    matrix = splits.interaction_matrix(fit_df, index)
    users, targets = index.encode(eval_df)
    scores = score_fn(matrix, users)
    seen = matrix[users].toarray() > 0
    return metrics.ranking_metrics(scores, seen, targets)


def select_on_validation(make_score_fn, grid, fit_df, val_df, index, label):
    """Pick the grid value with the best validation NDCG@10. The test set is never used here."""
    results = {value: evaluate_ranking(make_score_fn(value), fit_df, val_df, index)["ndcg@10"] for value in grid}
    best = max(results, key=results.get)
    summary = ", ".join(f"{value}: {score:.4f}" for value, score in results.items())
    print(f"{label} validation NDCG@10 -> {summary} | selected {best}")
    return best
