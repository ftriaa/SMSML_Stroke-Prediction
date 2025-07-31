import pandas as pd
import numpy as np
import mlflow
import dagshub
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
import os
import sys

from dagshub import dagshub_logger
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
from mlflow.models.signature import infer_signature

warnings.filterwarnings("ignore")

# KONFIGURASI DAGSHUB DAN MLFLOW 
mlflow.set_tracking_uri("https://dagshub.com/ftriaa/msml_stroke_model.mlflow")
dagshub.init(repo_owner="ftriaa", repo_name="msml_stroke_model", mlflow=True)
mlflow.set_experiment("Model_Tuning_Advanced")

# FUNCTION HELPER 

def load_data(path=r"C:\Users\FITRIA\SMSML_Fitria-Anggraini\Membangun_model\stroke_dataset_preprocessing.csv"):
    df = pd.read_csv(path)
    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    return X, y

def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
    cm = confusion_matrix(y_test, y_pred)

    return y_pred, y_proba, acc, prec, rec, f1, roc_auc, cm

def save_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    path = f"{model_name}_confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    return path

def save_metrics_json(metrics, model_name):
    path = f"{model_name}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    return path

# MODEL TRAINING DAN LOGGING 
def tune_and_log_model(name, model, param_grid, X_res, y_res, X_test, y_test, feat_names):
    with mlflow.start_run(run_name=f"Tuned_{name}"):
        # Log versi environment
        mlflow.log_param("python_version", sys.version)
        mlflow.log_param("pandas_version", pd.__version__)
        mlflow.log_param("numpy_version", np.__version__)
        mlflow.log_param("mlflow_version", mlflow.__version__)

        grid = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
        grid.fit(X_res, y_res)
        best_model = grid.best_estimator_

        y_pred, y_proba, acc, prec, rec, f1, roc_auc, cm = evaluate_model(best_model, X_test, y_test)

        mlflow.log_param("model", name)
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_cv_score", grid.best_score_)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix_TN": cm[0][0],
            "confusion_matrix_FP": cm[0][1],
            "confusion_matrix_FN": cm[1][0],
            "confusion_matrix_TP": cm[1][1],
        })

        # Signature & input example
        signature = infer_signature(X_test, y_pred)
        input_example = X_test.iloc[:3]

        mlflow.sklearn.log_model(best_model, "model", registered_model_name=f"Tuned_{name}_model", 
                                 signature=signature, input_example=input_example)

        cm_path = save_confusion_matrix(cm, name)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        json_dict = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "roc_auc": roc_auc}
        json_path = save_metrics_json(json_dict, name)
        mlflow.log_artifact(json_path)
        os.remove(json_path)

        # Confusion matrix JSON
        cm_dict = {"TN": int(cm[0][0]), "FP": int(cm[0][1]), "FN": int(cm[1][0]), "TP": int(cm[1][1])}
        with open(f"{name}_confusion_matrix.json", "w") as f:
            json.dump(cm_dict, f)
        mlflow.log_artifact(f"{name}_confusion_matrix.json")
        os.remove(f"{name}_confusion_matrix.json")

        # Classification report
        class_text = classification_report(y_test, y_pred, zero_division=0)
        html_content = f"""
        <html><head><title>Classification Report - {name}</title></head>
        <body><h2>Classification Report for {name}</h2><pre>{class_text}</pre></body></html>
        """
        html_path = f"{name}_estimator.html"
        with open(html_path, "w") as f:
            f.write(html_content)
        mlflow.log_artifact(html_path)
        os.remove(html_path)

        # Feature importance
        if hasattr(best_model, "feature_importances_"):
            fi = best_model.feature_importances_
            plt.figure(figsize=(8, 6))
            sns.barplot(x=fi, y=feat_names)
            plt.title(f"Feature Importance - {name}")
            plt.tight_layout()
            fi_path = f"{name}_feature_importance.png"
            plt.savefig(fi_path)
            plt.close()
            mlflow.log_artifact(fi_path)
            os.remove(fi_path)

            fi_dict = dict(zip(feat_names, fi.tolist()))
            with open(f"{name}_feature_importance.json", "w") as f:
                json.dump(fi_dict, f, indent=4)
            mlflow.log_artifact(f"{name}_feature_importance.json")
            os.remove(f"{name}_feature_importance.json")

        # ROC Curve
        if y_proba is not None:
            RocCurveDisplay.from_predictions(y_test, y_proba)
            roc_path = f"{name}_roc_curve.png"
            plt.savefig(roc_path)
            plt.close()
            mlflow.log_artifact(roc_path)
            os.remove(roc_path)

        print(f"\n{name} Tuned Model")
        print(f"Best Params: {grid.best_params_}")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

# MAIN PROGRAM 
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_res, y_res = apply_smote(X_train, y_train)

    models = {
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [100, 200], "max_depth": [None, 10]}
        },
        "XGBoost": {
            "estimator": XGBClassifier(random_state=42, eval_metric="logloss"),
            "params": {"n_estimators": [100, 200], "learning_rate": [0.1, 0.01]}
        },
        "LightGBM": {
            "estimator": LGBMClassifier(random_state=42),
            "params": {"n_estimators": [100, 200], "learning_rate": [0.1, 0.01]}
        },
        "SVM": {
            "estimator": SVC(probability=True, random_state=42),
            "params": {"C": [1, 10], "kernel": ["rbf", "linear"]}
        },
        "NaiveBayes": {
            "estimator": GaussianNB(),
            "params": {}
        }
    }

    for name, cfg in models.items():
        tune_and_log_model(name, cfg["estimator"], cfg["params"], X_res, y_res, X_test, y_test, X.columns)

# RUN SCRIPT 
if __name__ == "__main__":
    main()