import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# ===============================================================
# 1. LOAD DATASET
# ===============================================================
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


# ===============================================================
# 2. MLflow Setup (Manual Logging)
# ===============================================================
mlflow.set_tracking_uri("file:./mlruns")   # Simpan lokal
mlflow.set_experiment("Placement_Model_Tuning")


# ===============================================================
# 3. Hyperparameter Tuning (Skilled Requirement)
# ===============================================================
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 7, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 3],
    "max_features": ["sqrt", "log2"]
}

model_base = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
)

grid = GridSearchCV(
    estimator=model_base,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1
)

grid.fit(X_train_res, y_train_res)

best_model = grid.best_estimator_
best_params = grid.best_params_


# ===============================================================
# 4. START MLflow RUN
# ===============================================================
with mlflow.start_run():

    # ================
    # Log Parameters
    # ================
    mlflow.log_params(best_params)

    # ================
    # Train final model
    # ================
    best_model.fit(X_train_res, y_train_res)
    pred = best_model.predict(X_test_scaled)

    # =====================
    # Metrics calculation
    # =====================
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    # =====================
    # Manual Log Metrics
    # =====================
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    print("Best Params:", best_params)
    print("Accuracy:", acc)
    print(classification_report(y_test, pred))

    # ============================
    # Confusion Matrix Artifact
    # ============================
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Random Forest (Tuning)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    cm_path = "confusion_matrix_tuned.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    # ============================
    # Log Model (Manual)
    # ============================
    mlflow.sklearn.log_model(best_model, "model_tuned")

