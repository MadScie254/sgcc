"""
SGCC Theft Detector - Class-imbalance treatment (proposal section 3.9)

``Treatment`` wraps the data-level interventions compared in the study:

- ``"none"``: the training data is used as it is (XGBoost handles NaN natively).
- ``"smote"``: SMOTE oversampling of theft customers.
- ``"smote_enn"``: SMOTE, then Edited Nearest Neighbours cleaning (the proposal's framework).

Resampling needs complete, comparably scaled rows, so missing feature values are
filled with the training medians and distances are measured on standardised
features. Resampled rows are mapped back to the original units, so a model
trained on them scores ordinary feature rows; ``transform`` applies the same
median fill to any data scored later. Only training data is ever resampled.

``diagnostics`` measures what the treatment did to the training data
(Objective 1): class counts, separability (silhouette score, Fisher's
discriminant ratio) and boundary noise.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

TREATMENTS = ("none", "smote", "smote_enn")

DEFAULT_RESAMPLING_CONFIG = {
    "sampling_strategy": 0.5,  # theft : honest after SMOTE
    "smote_k_neighbors": 5,
    "enn_n_neighbors": 3,
}


def _counts(y) -> Dict[str, int]:
    y = np.asarray(y)
    return {"honest": int((y == 0).sum()), "theft": int((y == 1).sum())}


class Treatment:
    """A data-level imbalance treatment, fitted on training data only."""

    def __init__(self, kind: str = "none", config: Optional[dict] = None, random_state: int = 42):
        if kind not in TREATMENTS:
            raise ValueError(f"Unknown treatment {kind!r}; choose from {TREATMENTS}")
        self.kind = kind
        self.config = {**DEFAULT_RESAMPLING_CONFIG, **(config or {})}
        self.random_state = random_state
        self.medians_: Optional[pd.Series] = None
        self.stats_: Dict[str, Any] = {}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare rows for scoring: the training-median fill used before resampling."""
        return X if self.medians_ is None else X.fillna(self.medians_)

    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        if self.kind == "none":
            self.stats_ = {"before": _counts(y), "after": _counts(y)}
            return X, y
        self.medians_ = X.median().fillna(0.0)
        filled = self.transform(X)
        scaler = StandardScaler().fit(filled)
        Z = scaler.transform(filled)
        y_arr = np.asarray(y).astype(int)

        smote = SMOTE(sampling_strategy=self.config["sampling_strategy"],
                      k_neighbors=self.config["smote_k_neighbors"], random_state=self.random_state)
        Z_s, y_s = smote.fit_resample(Z, y_arr)
        synthetic = len(y_s) - len(y_arr)
        stats: Dict[str, Any] = {"before": _counts(y_arr), "after_smote": _counts(y_s), "synthetic_created": int(synthetic)}

        if self.kind == "smote_enn":
            enn = EditedNearestNeighbours(n_neighbors=self.config["enn_n_neighbors"], kind_sel="mode", sampling_strategy="all")
            Z_s, y_s = enn.fit_resample(Z_s, y_s)
            kept = np.asarray(enn.sample_indices_)
            is_synthetic = np.zeros(len(y_arr) + synthetic, dtype=bool)
            is_synthetic[len(y_arr):] = True
            removed = np.setdiff1d(np.arange(len(is_synthetic)), kept)
            original_removed = removed[~is_synthetic[removed]]
            stats.update({
                "synthetic_removed_by_enn": int(is_synthetic[removed].sum()),
                "honest_removed_by_enn": int((y_arr[original_removed] == 0).sum()),
                "theft_removed_by_enn": int((y_arr[original_removed] == 1).sum()),
            })
        stats["after"] = _counts(y_s)
        self.stats_ = stats
        X_res = pd.DataFrame(scaler.inverse_transform(Z_s), columns=X.columns).astype("float32")
        return X_res, pd.Series(y_s, name=y.name)


def diagnostics(X: pd.DataFrame, y, scaler: StandardScaler, sample_size: int = 5000, neighbors: int = 5,
                random_state: int = 42) -> Dict[str, float]:
    """
    Separability and boundary noise of a training set, in ``scaler``'s standardised space.

    - silhouette: mean silhouette of the two classes (higher = better separated), on a
      stratified sample of ``sample_size`` rows.
    - fisher_ratio_mean / fisher_ratio_max: (mu1 - mu0)^2 / (var1 + var0) per feature.
    - boundary_noise: share of rows whose ``neighbors`` nearest neighbours mostly carry the
      other label; boundary_noise_theft: the same, among theft rows only.
    """
    y = np.asarray(y).astype(int)
    Z = scaler.transform(X)
    rng = np.random.default_rng(random_state)
    if len(y) > sample_size:
        idx = np.concatenate([rng.choice(np.flatnonzero(y == c), size=max(1, round(sample_size * (y == c).mean())), replace=False)
                              for c in (0, 1)])
        Zs, ys = Z[idx], y[idx]
    else:
        Zs, ys = Z, y
    fisher = (Z[y == 1].mean(axis=0) - Z[y == 0].mean(axis=0)) ** 2 / np.maximum(Z[y == 1].var(axis=0) + Z[y == 0].var(axis=0), 1e-12)
    nn = NearestNeighbors(n_neighbors=neighbors + 1).fit(Zs)
    neighbour_labels = ys[nn.kneighbors(Zs, return_distance=False)[:, 1:]]
    noisy = (neighbour_labels != ys[:, None]).mean(axis=1) > 0.5
    return {
        "rows": int(len(y)),
        "theft_share": float(y.mean()),
        "silhouette": float(silhouette_score(Zs, ys)),
        "fisher_ratio_mean": float(fisher.mean()),
        "fisher_ratio_max": float(fisher.max()),
        "boundary_noise": float(noisy.mean()),
        "boundary_noise_theft": float(noisy[ys == 1].mean()),
    }
