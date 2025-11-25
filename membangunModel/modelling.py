import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================
# 1. LOAD DATASET
# ==============================================
df = pd.read_csv("preprocessing/processed_dataset.csv")

X = df.drop(columns="PlacementStatus")
y = df["PlacementStatus"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Resampling
smote_tomek = SMOTETomek(random_state=32)
X_train_res, y_train_res = smote_tomek.fit_resample(X_train_scaled, y_train)

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Placement_Modelling.py")

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RF_Basic") as run:
    # MODEL
    model_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        class_weight='balanced',
        bootstrap=True,
        random_state=42
    )

    model_rf.fit(X_train_res, y_train_res)
    pred = model_rf.predict(X_test_scaled)

    # EVALUASI
    acc = accuracy_score(y_test, pred)
    print("Accuracy:", acc)
    print(classification_report(y_test, pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    # Simpan Confusion Matrix
    cm_path = "confusion_matrix_rf.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()
    
    # Cetak informasi run
    print(f"\n{'='*50}")
    print(f"Run ID: {run.info.run_id}")
    print(f"Artifact URI: {run.info.artifact_uri}")
    print(f"{'='*50}\n")
    print("Model dan artifacts berhasil disimpan!")
    print(f"Cek folder: mlruns/{run.info.experiment_id}/{run.info.run_id}/artifacts/model/")
