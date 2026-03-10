# ===============================================
# Ultimate XGBoost Binary Classification Pipeline
# Fully YAML-Configurable
# ===============================================

import os
import numpy as np
import pandas as pd
import joblib
import logging
import shap
import optuna
import yaml
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, plot_importance

# -----------------------------
# Logging & Seed
# -----------------------------
logging.basicConfig(
    filename="xgb_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
SEED = 42
np.random.seed(SEED)

# -----------------------------
# YAML Loader
# -----------------------------
def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

# -----------------------------
# CV Target Encoding + Frequency Encoding
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

def group_rare_categories(X, cols, min_count=10):
    X = X.copy()
    for col in cols:
        counts = X[col].value_counts()
        rare = counts[counts < min_count].index
        X[col] = X[col].replace(rare, "RARE_CAT")
    return X

def add_interactions(X, numeric_cols, max_pairs=50):
    import itertools
    interactions = {}
    for i, (c1, c2) in enumerate(itertools.combinations(numeric_cols, 2)):
        interactions[f"{c1}_x_{c2}"] = X[c1] * X[c2]
        if i + 1 >= max_pairs:
            break
    return pd.DataFrame(interactions)

# -----------------------------
# Data Loader & Preprocessing
# -----------------------------
def load_and_preprocess(csv_path, target_col, id_col):
    df = pd.read_csv(csv_path).drop_duplicates()
    ids = df[id_col].copy()
    y = df[target_col]
    if y.dtype == "object":
        y = y.astype("category").cat.codes
    X = df.drop(columns=[target_col, id_col])
    X = X.loc[:, X.nunique() > 1]
    X = X.loc[:, X.isna().mean() < 0.95]
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X = group_rare_categories(X, cat_cols)
    return X, y, ids, cat_cols

# -----------------------------
# Utility Functions
# -----------------------------
def impute_train_test(X_tr, X_val):
    med = X_tr.median()
    return X_tr.fillna(med), X_val.fillna(med)

def shap_select(model, X, top_k=30):
    sample = shap.sample(X, min(1000, len(X)))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    shap_importance = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(shap_importance)[-top_k:]
    return idx

# -----------------------------
# Optuna Objective with YAML params
# -----------------------------
def objective_xgb(trial, X, y, param_ranges, gpu=True):
    params = {}
    for p, v in param_ranges.items():
        if v.get("type") == "int":
            params[p] = trial.suggest_int(p, v["low"], v["high"])
        elif v.get("type") == "float":
            params[p] = trial.suggest_float(p, v["low"], v["high"], log=v.get("log", False))
    params.update({
        "tree_method": "gpu_hist" if gpu else "hist",
        "eval_metric": "logloss",
        "n_estimators": 1000,
        "random_state": SEED
    })
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    aucs = []
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        X_tr, X_val = impute_train_test(X_tr, X_val)
        model = XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        pred = model.predict_proba(X_val)[:,1]
        aucs.append(roc_auc_score(y_val, pred))
    return np.mean(aucs)

# -----------------------------
# Ultimate YAML-Driven Training Pipeline
# -----------------------------
def train_ultimate_xgb(csv_path, target_col, id_col, pipeline_yaml_path, param_yaml_path, save_dir="ultimate_xgb_pipeline"):
    os.makedirs(save_dir, exist_ok=True)

    # Load YAML configs
    pipeline_cfg = load_yaml(pipeline_yaml_path)
    param_cfg = load_yaml(param_yaml_path)

    X, y, ids, cat_cols = load_and_preprocess(csv_path, target_col, id_col)

    # Encode categories
    if cat_cols:
        te = CVTargetEncoder(cols=cat_cols, n_splits=pipeline_cfg.get("n_splits",5))
        X = te.fit_transform(X, y)
        X = frequency_encode(X, cat_cols)

    # Add interactions
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    interactions = add_interactions(X, numeric_cols, max_pairs=pipeline_cfg.get("max_interactions",50))
    X = pd.concat([X, interactions], axis=1)

    # Outer CV
    outer_cv = StratifiedKFold(n_splits=pipeline_cfg.get("n_splits",5), shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    fold_aucs = []
    fold = 1
    best_params_all_folds = []

    for tr_idx, val_idx in outer_cv.split(X, y):
        logging.info(f"Starting Fold {fold}")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        X_tr, X_val = impute_train_test(X_tr, X_val)

        # Optuna tuning
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective_xgb(t, X_tr, y_tr, param_cfg, gpu=pipeline_cfg.get("gpu", True)), n_trials=pipeline_cfg.get("n_trials",15))
        best_params = study.best_params
        best_params.update({
            "tree_method": "gpu_hist" if pipeline_cfg.get("gpu", True) else "hist",
            "eval_metric": "logloss",
            "n_estimators": 1000,
            "random_state": SEED
        })
        best_params_all_folds.append(best_params)

        # Initial model to select SHAP features
        model = XGBClassifier(**best_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=pipeline_cfg.get("early_stopping_rounds",50),
            verbose=False
        )
        idx = shap_select(model, X_tr, top_k=pipeline_cfg.get("top_k_shap",40))
        X_tr_sel, X_val_sel = X_tr.iloc[:, idx], X_val.iloc[:, idx]

        # Train model on selected features
        model.fit(
            X_tr_sel, y_tr,
            eval_set=[(X_val_sel, y_val)],
            early_stopping_rounds=pipeline_cfg.get("early_stopping_rounds",50),
            verbose=False
        )

        oof_preds[val_idx] = model.predict_proba(X_val_sel)[:,1]
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_aucs.append(fold_auc)
        logging.info(f"Fold {fold} AUC={fold_auc:.4f}")
        fold += 1

    # CV AUC plot
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(fold_aucs)+1), fold_aucs, marker='o')
    plt.title("Fold AUCs")
    plt.xlabel("Fold")
    plt.ylabel("AUC")
    plt.ylim(0,1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cv_auc_per_fold.png"))

    # Train final model on full data
    X_imputed = X.fillna(X.median())
    X_final = X_imputed.iloc[:, idx]
    final_model = XGBClassifier(**best_params_all_folds[0])
    final_model.fit(
        X_final, y,
        eval_set=[(X_final, y)],
        early_stopping_rounds=pipeline_cfg.get("early_stopping_rounds",50),
        verbose=True
    )

    # Feature importance
    plt.figure(figsize=(10,8))
    plot_importance(final_model, max_num_features=40)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"))

    # Save YAML of final hyperparameters and pipeline settings
    with open(os.path.join(save_dir,"pipeline_config.yaml"), "w") as f:
        yaml.dump({
            "pipeline_cfg": pipeline_cfg,
            "best_params_all_folds": best_params_all_folds,
            "selected_features_idx": idx
        }, f)

    # Save artifacts
    joblib.dump({
        "model": final_model,
        "oof_preds": oof_preds,
        "fold_aucs": fold_aucs
    }, os.path.join(save_dir, "ultimate_xgb_pipeline.pkl"))

    logging.info(f"Pipeline complete. Artifacts saved in {save_dir}")
    return final_model, idx, oof_preds, fold_aucs