"""LACE vs logistic regression vs XGBoost with patient-grouped cross-validation.

No patient appears in both the training and test folds. Hyperparameters are chosen inside each training fold
(nested, also patient-grouped), so test folds never influence tuning. Metrics are computed on out-of-fold
predictions with patient-clustered bootstrap CIs.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

N_SPLITS, INNER_SPLITS = 5, 3
# grids reach strong regularization: the first run picked the edge of a narrower grid in every fold
XGB_GRID = {"max_depth": [1, 2, 3], "n_estimators": [50, 100, 300]}
LR_GRID = {"logisticregression__C": [0.001, 0.003, 0.01, 0.1, 1.0]}
ESTIMATORS = ["logistic", "xgboost"]


def make_xgb(seed: int = 0) -> XGBClassifier:
    # shallow, regularized trees: ~900 rows and ~180 events
    return XGBClassifier(learning_rate=0.05, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=1.0, tree_method="hist", eval_metric="logloss", n_jobs=8, random_state=seed)


def make_logistic() -> object:
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(max_iter=5000))


def tuned(name: str, seed: int) -> GridSearchCV:
    est, grid = (make_xgb(seed), XGB_GRID) if name == "xgboost" else (make_logistic(), LR_GRID)
    inner = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    return GridSearchCV(est, grid, cv=inner, scoring="roc_auc", n_jobs=1)


def outer_folds(y, groups, seed: int):
    return list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, groups))


def cross_validate(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, feature_sets: dict[str, list[str]],
                   seed: int = 0) -> dict:
    """Out-of-fold predictions: LACE score, LACE-only logistic, and logistic + XGBoost per feature set."""
    names = ["lace_prob"] + [f"{e}_{fs}" for fs in feature_sets for e in ESTIMATORS]
    oof = {k: np.full(len(y), np.nan) for k in names}
    chosen = []
    for train, test in outer_folds(y, groups, seed):
        assert not set(groups[train]) & set(groups[test]), "patient in both train and test folds"
        lace_lr = LogisticRegression().fit(X.iloc[train][["lace"]], y[train])
        oof["lace_prob"][test] = lace_lr.predict_proba(X.iloc[test][["lace"]])[:, 1]
        for fs, cols in feature_sets.items():
            for name in ESTIMATORS:
                search = tuned(name, seed).fit(X.iloc[train][cols], y[train], groups=groups[train])
                oof[f"{name}_{fs}"][test] = search.predict_proba(X.iloc[test][cols])[:, 1]
                chosen.append(f"{name}_{fs} {search.best_params_}")
    oof["lace"] = X["lace"].to_numpy(dtype=float)
    return {"oof": oof, "chosen_params": pd.Series(chosen).value_counts().to_dict()}


def fit_final_xgb(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> XGBClassifier:
    """Tune on all eligible data (grouped inner CV) and refit; used for SHAP, not for the reported metrics."""
    return tuned("xgboost", seed).fit(X, y, groups=groups).best_estimator_
