import pandas as pd
import numpy as np
import shap
import logging
import optuna
import joblib
import os
import yaml

from tqdm import tqdm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


# =========================================================
# Config
# =========================================================

SEED = 42
np.random.seed(SEED)

DATA_PATH = r"C:\Users\EricVonDohlen\raw_data.csv"
TARGET = "target"

OUT_DIR = "model_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================================================
# Data Loading
# =========================================================

def load_and_clean_data(path, target):

    df = pd.read_csv(path)
    df = df.drop_duplicates()

    # Drop columns with >95% missing
    missing = df.isna().mean()
    df = df.drop(columns=missing[missing > 0.95].index)

    # Drop constant columns
    nunique = df.nunique()
    df = df.drop(columns=nunique[nunique <= 1].index)

    y = df[target]
    X = df.drop(columns=[target])

    # Ensure binary
    if y.dtype == "object":
        y = y.astype("category").cat.codes

    assert set(y.unique()) <= {0,1}

    # Handle categorical variables
    cat_cols = X.select_dtypes(include=["object","category"]).columns

    # Remove extremely high cardinality
    high_card = [c for c in cat_cols if X[c].nunique() > 50]
    X = X.drop(columns=high_card)

    cat_cols = [c for c in cat_cols if c not in high_card]

    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y


# =========================================================
# Median Imputation
# =========================================================

def impute_train_test(X_train, X_test):

    med = X_train.median()

    X_train = X_train.fillna(med)
    X_test = X_test.fillna(med)

    return X_train, X_test


# =========================================================
# SHAP Feature Selection
# =========================================================

def shap_feature_selection(model, X, top_k):

    sample = shap.sample(X, min(len(X), 1000), random_state=SEED)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    importance = np.abs(shap_values).mean(axis=0)

    idx = np.argsort(importance)[-top_k:]

    return idx


# =========================================================
# Optuna Objective
# =========================================================

def objective(trial, X_train, y_train):

    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 5),
        "tree_method": "hist",
        "eval_metric": "logloss",
        "random_state": SEED
    }

    top_k = trial.suggest_int("top_k_features", 10, 50)
    calib_method = trial.suggest_categorical("calibration", ["sigmoid", "isotonic"])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    scores = []

    for tr_idx, val_idx in cv.split(X_train, y_train):

        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]

        y_tr = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        X_tr, X_val = impute_train_test(X_tr, X_val)

        # SHAP model
        shap_model = XGBClassifier(**params)
        shap_model.fit(X_tr, y_tr)

        selected = shap_feature_selection(shap_model, X_tr, top_k)

        X_tr_sel = X_tr.iloc[:, selected]
        X_val_sel = X_val.iloc[:, selected]

        calibrated = CalibratedClassifierCV(
            XGBClassifier(**params),
            method=calib_method,
            cv=3
        )

        calibrated.fit(X_tr_sel, y_tr)

        prob = calibrated.predict_proba(X_val_sel)[:,1]

        scores.append(roc_auc_score(y_val, prob))

    return np.mean(scores)


# =========================================================
# Main Pipeline
# =========================================================

if __name__ == "__main__":

    X, y = load_and_clean_data(DATA_PATH, TARGET)

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    outer_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(list(outer_cv.split(X,y)), desc="Outer CV"),1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        study = optuna.create_study(
            direction="maximize",
            study_name=f"xgb_nested_cv_fold_{fold}",
            storage="sqlite:///optuna_studies.db",
            load_if_exists=True
        )

        study.optimize(
            lambda trial: objective(trial, X_train, y_train),
            n_trials=50
        )

        best = study.best_params

        final_params = {
            k:v for k,v in best.items()
            if k not in ["top_k_features","calibration"]
        }

        final_params.update({
            "tree_method":"hist",
            "eval_metric":"logloss",
            "random_state":SEED
        })

        # Impute
        X_train, X_test = impute_train_test(X_train, X_test)

        # Train temp model for SHAP
        temp_model = XGBClassifier(**final_params)
        temp_model.fit(X_train, y_train)

        selected = shap_feature_selection(
            temp_model,
            X_train,
            best["top_k_features"]
        )

        X_train_sel = X_train.iloc[:, selected]
        X_test_sel = X_test.iloc[:, selected]

        calibrated = CalibratedClassifierCV(
            XGBClassifier(**final_params),
            method=best["calibration"],
            cv=3
        )

        calibrated.fit(X_train_sel, y_train)

        prob = calibrated.predict_proba(X_test_sel)[:,1]

        auc = roc_auc_score(y_test, prob)

        outer_scores.append(auc)

        # =================================================
        # Save artifacts
        # =================================================

        fold_dir = f"{OUT_DIR}/fold_{fold}"
        os.makedirs(fold_dir, exist_ok=True)

        joblib.dump(calibrated, f"{fold_dir}/calibrated_model.joblib")

        booster = temp_model.get_booster()
        booster.save_model(f"{fold_dir}/xgb_model.json")

        np.save(f"{fold_dir}/selected_features.npy", selected)

        with open(f"{fold_dir}/best_params.yaml","w") as f:
            yaml.dump(best,f)

        logging.info(f"Fold {fold} AUC {auc:.4f}")

    print("================================")
    print("Nested CV Results")
    print("================================")

    print("Mean AUC:", np.mean(outer_scores))
    print("Std AUC:", np.std(outer_scores))