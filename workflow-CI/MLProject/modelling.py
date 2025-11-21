import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.combine import SMOTETomek
import matplotlib
matplotlib.use("Agg")  # FIX: no display for GitHub Actions
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 0. SAFE BASE DIRECTORY
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. LOAD DATASET
# ==========================================
dataset_path = os.path.join(BASE_DIR, "preprocessing", "processed_dataset.csv")
df = pd.read_csv(dataset_path)

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

# ==========================================
# 2. MLflow Setup
# ==========================================
mlflow_dir = os.path.join(BASE_DIR, "mlruns")
mlflow.set_tracking_uri(f"file:{mlflow_dir}")
mlflow.set_experiment("Placement_Model_Automated")

# ==========================================
# 3. TRAINING MODEL
# ==========================================
with mlflow.start_run():
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

    # Train model
    model_rf.fit(X_train_res, y_train_res)
    pred = model_rf.predict(X_test_scaled)

    # ==========================================
    # 4. EVALUATION
    # ==========================================
    acc = accuracy_score(y_test, pred)
    print("Accuracy:", acc)
    print(classification_report(y_test, pred))

    mlflow.log_metric("accuracy", acc)

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    # ==========================================
    # 5. ARTIFACT FOLDER (SAFE)
    # ==========================================
    artifacts_dir = os.path.join(BASE_DIR, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    cm_path = os.path.join(artifacts_dir, "confusion_matrix_rf.png")
    plt.savefig(cm_path)

    mlflow.log_artifact(cm_path)

    # ==========================================
    # 6. SAVE MODEL
    # ==========================================
    mlflow.sklearn.log_model(
        sk_model=model_rf,
        artifact_path="model"
    )

print("Model training & logging complete.")