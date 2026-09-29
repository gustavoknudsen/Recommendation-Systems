from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def save(rows, name, source):
    """Write result rows to results/<name>.csv, replacing every row previously saved by `source`.

    Each notebook passes its own name as `source`, so rerunning a notebook never leaves stale rows behind.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    new = pd.DataFrame(rows).assign(source=source)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["source"] != source], new], ignore_index=True)
    new.to_csv(path, index=False, float_format="%.4f")
    return new


def load(name):
    return pd.read_csv(RESULTS_DIR / f"{name}.csv")
