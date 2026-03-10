# ===============================================
# Ultimate Tabular ML Pipeline (GPU + Stacking)
# ===============================================

import os
import numpy as np
import pandas as pd
import joblib
import shap
import optuna
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# -----------------------------
# Logging & Seed
# -----------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
SEED = 42
np.random.seed(SEED)

# -----------------------------
# Target + Frequency Encoding
# -----------------------------
class CVTargetEncoder:
    def __init__(self, cols, n_splits=5, smoothing=20):
        self.cols = cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.global_mean = None

    def fit_transform(self, X, y):
        X = X.copy()
        self.global_mean = y.mean()
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=SEED)
        for col in self.cols:
            X[col + "_te"] = np.nan
            for train_idx, val_idx in skf.split(X, y):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                stats = y_tr.groupby(X_tr[col]).agg(["mean", "count"])
                smoothing = 1 / (1 + np.exp(-(stats["count"] - self.smoothing)))
                smooth_mean = self.global_mean * (1 - smoothing) + stats["mean"] * smoothing
                X.loc[val_idx, col + "_te"] = X_val[col].map(smooth_mean)
            X[col + "_te"].fillna(self.global_mean, inplace=True)
        return X

def frequency_encode(X, cols):
    X = X.copy()
    for col in cols:
        freq = X[col].value_counts(normalize=True)
        X[col + "_freq"] = X[col].map(freq)
    return X

# -----------------------------
# Feature Interactions
# -----------------------------
def add_interactions(X, numeric_cols, max_pairs=50):
    import itertools
    interactions = {}
    for i, (c1, c2) in enumerate(itertools.combinations(numeric_cols, 2)):
        interactions[f"{c1}_x_{c2}"] = X[c1] * X[c2]
        if i + 1 >= max_pairs:
            break
    return pd.DataFrame(interactions)

# -----------------------------
# Data Loading & Preprocessing
# -----------------------------
def load_and_preprocess(csv_path, target_col, id_col):
    df = pd.read_csv(csv_path).drop_duplicates()
    ids = df[id_col].copy()
    y = df[target_col]
    if y.dtype == "object":
        y = y.astype("category").cat.codes
    X = df.drop(columns=[target_col, id_col])
    # drop constant & high-missing
    X = X.loc[:, X.nunique() > 1]
    X = X.loc[:, X.isna().mean() < 0.95]
    # handle categorical
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    high_card = [c for c in cat_cols if X[c].nunique() > 50]
    X = X.drop(columns=high_card)
    cat_cols = [c for c in cat_cols if c not in high_card]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y, ids, cat_cols

# -----------------------------
# Imputation
# -----------------------------
def impute_train_test(X_tr, X_val):
    med = X_tr.median()
    return X_tr.fillna(med), X_val.fillna(med)

# -----------------------------
# SHAP Feature Selection
# -----------------------------
def shap_select(model, X, top_k=30):
    sample = shap.sample(X, min(1000, len(X)))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    shap_importance = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(shap_importance)[-top_k:]
    return idx

# -----------------------------
# Optuna Objective
# -----------------------------
def objective_xgb(trial, X, y):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2),
        "tree_method": "gpu_hist",
        "eval_metric": "logloss",
        "random_state": SEED
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    aucs = []
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        X_tr, X_val = impute_train_test(X_tr, X_val)
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict_proba(X_val)[:,1]
        aucs.append(roc_auc_score(y_val, pred))
    return np.mean(aucs)

# -----------------------------
# Train Ultimate Stacked Pipeline
# -----------------------------
def train_ultimate_pipeline(csv_path, target_col, id_col, save_dir="ultimate_model"):

    os.makedirs(save_dir, exist_ok=True)

    X, y, ids, cat_cols = load_and_preprocess(csv_path, target_col, id_col)

    # Frequency + Target encoding for high-cardinality categorical
    if cat_cols:
        te = CVTargetEncoder(cols=cat_cols)
        X = te.fit_transform(X, y)
        X = frequency_encode(X, cat_cols)

    # Numeric interactions
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    interactions = add_interactions(X, numeric_cols, max_pairs=50)
    X = pd.concat([X, interactions], axis=1)

    # Outer CV for stacking
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_oof = np.zeros(len(X))
    fold = 1

    for tr_idx, val_idx in outer_cv.split(X, y):
        logging.info(f"Fold {fold} starting")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # Imputation
        X_tr, X_val = impute_train_test(X_tr, X_val)

        # Hyperparameter tuning for XGB
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective_xgb(t, X_tr, y_tr), n_trials=20)
        best_params = study.best_params
        best_params["tree_method"] = "gpu_hist"
        best_params["eval_metric"] = "logloss"
        best_params["random_state"] = SEED

        # Fit base models
        xgb_model = XGBClassifier(**best_params)
        lgb_model = LGBMClassifier(n_estimators=800, learning_rate=0.05, device="gpu")
        cat_model = CatBoostClassifier(verbose=0, task_type="GPU", iterations=800, learning_rate=0.05)

        xgb_model.fit(X_tr, y_tr)
        lgb_model.fit(X_tr, y_tr)
        cat_model.fit(X_tr, y_tr)

        # SHAP Feature Selection (XGB)
        idx = shap_select(xgb_model, X_tr, top_k=40)
        X_tr_sel = X_tr.iloc[:, idx]
        X_val_sel = X_val.iloc[:, idx]

        # Refit base models on selected features
        xgb_model.fit(X_tr_sel, y_tr)
        lgb_model.fit(X_tr_sel, y_tr)
        cat_model.fit(X_tr_sel, y_tr)

        # Create meta-features
        meta_tr = np.column_stack([
            xgb_model.predict_proba(X_tr_sel)[:,1],
            lgb_model.predict_proba(X_tr_sel)[:,1],
            cat_model.predict_proba(X_tr_sel)[:,1]
        ])
        meta_val = np.column_stack([
            xgb_model.predict_proba(X_val_sel)[:,1],
            lgb_model.predict_proba(X_val_sel)[:,1],
            cat_model.predict_proba(X_val_sel)[:,1]
        ])

        # Meta-model
        meta_model = LogisticRegression()
        meta_model.fit(meta_tr, y_tr)
        all_oof[val_idx] = meta_model.predict_proba(meta_val)[:,1]

        logging.info(f"Fold {fold} AUC: {roc_auc_score(y_val, all_oof[val_idx]):.4f}")

        fold +=1

    # Train final models on full dataset
    X_imputed = X.fillna(X.median())
    final_xgb = XGBClassifier(**best_params).fit(X_imputed, y)
    final_lgb = LGBMClassifier(n_estimators=800, learning_rate=0.05, device="gpu").fit(X_imputed, y)
    final_cat = CatBoostClassifier(verbose=0, task_type="GPU", iterations=800, learning_rate=0.05).fit(X_imputed, y)
    final_meta_tr = np.column_stack([
        final_xgb.predict_proba(X_imputed)[:,1],
        final_lgb.predict_proba(X_imputed)[:,1],
        final_cat.predict_proba(X_imputed)[:,1]
    ])
    final_meta = LogisticRegression().fit(final_meta_tr, y)

    # Save all models
    joblib.dump({
        "xgb": final_xgb,
        "lgb": final_lgb,
        "cat": final_cat,
        "meta": final_meta,
        "oof_preds": all_oof,
        "selected_features": idx
    }, os.path.join(save_dir,"ultimate_pipeline.pkl"))

    logging.info("Training complete")
    return final_xgb, final_lgb, final_cat, final_meta, all_oof

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    final_xgb, final_lgb, final_cat, final_meta, oof = train_ultimate_pipeline(
        csv_path="data.csv",
        target_col="onus_target",
        id_col="accountid",
        save_dir="ultimate_model"
    )