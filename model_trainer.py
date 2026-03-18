import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold


def _to_finite_float_frame(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(np.float32)


def _winsorize_by_col(X: pd.DataFrame, lo_q: float = 0.001, hi_q: float = 0.999) -> pd.DataFrame:
    X = X.copy()
    lo = X.quantile(lo_q)
    hi = X.quantile(hi_q)
    return X.clip(lower=lo, upper=hi, axis=1)


def _power_calibrate(p: np.ndarray, power: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    a = np.power(p, power)
    b = np.power(1 - p, power)
    return a / (a + b)


def _smooth_and_clip(p: np.ndarray, smooth: float) -> np.ndarray:
    p = (1.0 - smooth) * p + smooth * 0.5
    return np.clip(p, 0.02, 0.98)


class ModelTrainer:
    """
    Competition-grade trainer:
    - Two HGB parameterizations (A/B), each multi-seed ensembled
    - Season GroupKFold (no leakage across seasons)
    - Robust preprocessing (finite -> winsorize)
    - Lightweight tuning:
        * blend weight between A/B
        * calibration power in ~[1.05..1.10]
        * smoothing in ~[0.03..0.07]
    - Fits final models on full data after CV
    """

    def __init__(
        self,
        seeds=(42, 123, 456),
        n_splits: int = 5,
        winsor_lo_q: float = 0.001,
        winsor_hi_q: float = 0.999,
        power_grid=(1.05, 1.07, 1.10),
        smooth_grid=(0.03, 0.05, 0.07),
        weight_grid=tuple(np.round(np.arange(0.50, 0.701, 0.01), 2)),
        params_a=None,
        params_b=None,
    ):
        self.seeds = list(seeds)
        self.n_splits = int(n_splits)
        self.winsor_lo_q = float(winsor_lo_q)
        self.winsor_hi_q = float(winsor_hi_q)
        self.power_grid = tuple(float(x) for x in power_grid)
        self.smooth_grid = tuple(float(x) for x in smooth_grid)
        self.weight_grid = tuple(float(x) for x in weight_grid)

        # Model A: more regularized / stable
        self.params_a = {
            "loss": "log_loss",
            "learning_rate": 0.03,
            "max_iter": 700,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 0.25,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 40,
        }
        # Model B: a bit more expressive (but still safe)
        self.params_b = {
            "loss": "log_loss",
            "learning_rate": 0.02,
            "max_iter": 1200,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 20,
            "l2_regularization": 0.10,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 60,
        }
        if params_a:
            self.params_a.update(params_a)
        if params_b:
            self.params_b.update(params_b)

        self.feature_cols_: list[str] | None = None
        self.models_a_: list[HistGradientBoostingClassifier] = []
        self.models_b_: list[HistGradientBoostingClassifier] = []
        self.best_power_: float | None = None
        self.best_smooth_: float | None = None
        self.best_weight_: float | None = None

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        Xp = _to_finite_float_frame(X)
        Xp = _winsorize_by_col(Xp, self.winsor_lo_q, self.winsor_hi_q)
        Xp = Xp.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return Xp

    def _fit_family(self, X_tr: pd.DataFrame, y_tr: pd.Series, params: dict) -> list[HistGradientBoostingClassifier]:
        models: list[HistGradientBoostingClassifier] = []
        for seed in self.seeds:
            m = HistGradientBoostingClassifier(random_state=int(seed), **params)
            m.fit(X_tr, y_tr)
            models.append(m)
        return models

    @staticmethod
    def _predict_family(models: list[HistGradientBoostingClassifier], X: pd.DataFrame) -> np.ndarray:
        preds = [m.predict_proba(X)[:, 1] for m in models]
        return np.mean(preds, axis=0)

    def train(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> float:
        X = X.copy()
        y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
        groups = pd.to_numeric(groups, errors="coerce").fillna(0).astype(int)

        self.feature_cols_ = list(X.columns)
        Xp = self._prep(X)
        assert np.isfinite(Xp.to_numpy()).all(), "Non-finite values remain in training features."

        gkf = GroupKFold(n_splits=self.n_splits)
        oof_a = np.zeros(len(Xp), dtype=np.float64)
        oof_b = np.zeros(len(Xp), dtype=np.float64)

        print(f"Starting GroupKFold training (n_splits={self.n_splits}, seeds={len(self.seeds)})...")
        fold_scores_raw = []

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(Xp, y, groups=groups), start=1):
            X_tr, X_va = Xp.iloc[tr_idx], Xp.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            models_a = self._fit_family(X_tr, y_tr, self.params_a)
            models_b = self._fit_family(X_tr, y_tr, self.params_b)

            pa = self._predict_family(models_a, X_va)
            pb = self._predict_family(models_b, X_va)

            oof_a[va_idx] = pa
            oof_b[va_idx] = pb

            # A quick raw (uncalibrated) diagnostic using 0.6/0.4
            raw = 0.6 * pa + 0.4 * pb
            raw = _smooth_and_clip(_power_calibrate(raw, 1.07), 0.05)
            score = brier_score_loss(y_va, raw)
            fold_scores_raw.append(score)
            print(f"Fold {fold} Brier (diag): {score:.4f}")

        diag_cv = float(np.mean(fold_scores_raw))
        print(f"Diag CV (not tuned): {diag_cv:.4f}")

        # ---- Tune calibration + blend weight on full OOF (fast, stable) ----
        best = (np.inf, None, None, None)  # (brier, weight, power, smooth)
        for w in self.weight_grid:
            blend = w * oof_a + (1.0 - w) * oof_b
            for power in self.power_grid:
                cal = _power_calibrate(blend, power)
                for smooth in self.smooth_grid:
                    p = _smooth_and_clip(cal, smooth)
                    s = brier_score_loss(y, p)
                    if s < best[0]:
                        best = (float(s), float(w), float(power), float(smooth))

        self.best_weight_ = best[1]
        self.best_power_ = best[2]
        self.best_smooth_ = best[3]

        print(
            "Best OOF tune "
            f"(weight_A={self.best_weight_:.2f}, power={self.best_power_:.2f}, smooth={self.best_smooth_:.2f}) "
            f"Brier={best[0]:.4f}"
        )

        # ---- Fit final ensembles on ALL data ----
        self.models_a_.clear()
        self.models_b_.clear()
        self.models_a_ = self._fit_family(Xp, y, self.params_a)
        self.models_b_ = self._fit_family(Xp, y, self.params_b)

        return best[0]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models_a_ or not self.models_b_:
            raise RuntimeError("Call train() before predict().")

        X = X.copy()
        # align columns
        for c in self.feature_cols_:
            if c not in X.columns:
                X[c] = 0.0
        extra = [c for c in X.columns if c not in self.feature_cols_]
        if extra:
            X = X.drop(columns=extra)
        X = X[self.feature_cols_]

        Xp = self._prep(X)
        assert np.isfinite(Xp.to_numpy()).all(), "Non-finite values remain in test features."

        pa = self._predict_family(self.models_a_, Xp)
        pb = self._predict_family(self.models_b_, Xp)

        w = float(self.best_weight_ if self.best_weight_ is not None else 0.6)
        power = float(self.best_power_ if self.best_power_ is not None else 1.07)
        smooth = float(self.best_smooth_ if self.best_smooth_ is not None else 0.05)

        p = w * pa + (1.0 - w) * pb
        p = _power_calibrate(p, power)
        p = _smooth_and_clip(p, smooth)
        return p