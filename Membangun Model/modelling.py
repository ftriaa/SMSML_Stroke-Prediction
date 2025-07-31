import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

# Set lokasi direktori MLflow tracking
mlflow.set_tracking_uri("file:///C:/Users/FITRIA/SMSML_Fitria-Anggraini/Membangun_model/mlruns")

# Set nama eksperimen (akan otomatis buat folder dan meta.yaml jika belum ada)
mlflow.set_experiment("Stroke Prediction Models")

# Aktifkan autolog untuk scikit-learn
mlflow.sklearn.autolog()

df = pd.read_csv(r"C:\Users\FITRIA\SMSML_Fitria-Anggraini\Membangun_model\stroke_dataset_preprocessing.csv")

X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
    "LightGBM": LGBMClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "NaiveBayes": GaussianNB()
}

for name, model in models.items():
    print(f"\nTraining model: {name}")
    
    with mlflow.start_run(run_name=name):
        model.fit(X_train_resampled, y_train_resampled)

        # Log model manual agar artifact muncul di UI
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Prediksi dan probabilitas
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = None

        # Hitung metrik utama
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

        # Log metrik manual agar lengkap di UI
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Buat dan log Confusion Matrix (PNG)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=12)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {name}')
        image_path = f"confusion_matrix_{name}.png"
        plt.savefig(image_path)
        plt.close(fig)
        mlflow.log_artifact(image_path)
        os.remove(image_path)

        # Buat dan log file JSON metrik
        metrics_dict = {
            "model_name": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
        json_path = f"metrics_{name}.json"
        with open(json_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        mlflow.log_artifact(json_path)
        os.remove(json_path)

        # Buat dan log estimator.html (classification report dalam format HTML)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        report_text = classification_report(y_test, y_pred)
        html_content = f"""
        <html>
        <head><title>Classification Report - {name}</title></head>
        <body>
        <h2>Classification Report for {name}</h2>
        <pre>{report_text}</pre>
        </body>
        </html>
        """
        html_path = f"estimator_{name}.html"
        with open(html_path, "w") as f:
            f.write(html_content)
        mlflow.log_artifact(html_path)
        os.remove(html_path)

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"ROC AUC  : {roc_auc:.4f}")