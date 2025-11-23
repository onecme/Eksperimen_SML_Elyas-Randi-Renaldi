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
import sys

# ==============================================
# 0. PARSE ARGUMENTS
# ==============================================
data_path = sys.argv[1] if len(sys.argv) > 1 else "preprocessing/processed_dataset.csv"
target_var = sys.argv[2] if len(sys.argv) > 2 else "PlacementStatus"

print(f"Loading data from: {data_path}")
print(f"Target variable: {target_var}")

# ==============================================
# 1. LOAD DATASET
# ==============================================
df = pd.read_csv(data_path)

X = df.drop(columns=target_var)
y = df[target_var]

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

# ==============================================
# 2. TRAINING MODEL
# ==============================================
print("Training model...")
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

# ==============================================
# 3. EVALUASI
# ==============================================
acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc}")
print(classification_report(y_test, pred))

# ==============================================
# 4. MLFLOW LOGGING - Langsung log tanpa check active_run
# ==============================================
print("Logging to MLflow...")

# Log parameters
mlflow.log_param("n_estimators", 300)
mlflow.log_param("max_depth", 7)
mlflow.log_param("min_samples_split", 5)
mlflow.log_param("min_samples_leaf", 3)
mlflow.log_param("max_features", "sqrt")
mlflow.log_param("class_weight", "balanced")

# Log metrics
mlflow.log_metric("accuracy", acc)

# Confusion Matrix
print("Creating confusion matrix...")
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

# Save Confusion Matrix
cm_path = "confusion_matrix_rf.png"
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved to: {cm_path}")

# Log artifact
try:
    mlflow.log_artifact(cm_path)
    print("Confusion matrix logged to MLflow")
    # Clean up
    if os.path.exists(cm_path):
        os.remove(cm_path)
except Exception as e:
    print(f"Warning: Could not log confusion matrix: {e}")

# ==============================================
# 5. LOG MODEL 
# ==============================================
print("Logging model to MLflow...")
try:
    mlflow.sklearn.log_model(
        sk_model=model_rf,
        artifact_path="model",
        registered_model_name="RandomForest_Placement_Model"
    )
    print("Model successfully logged to MLflow")
except Exception as e:
    print(f"ERROR: Failed to log model: {e}")
    raise

print("=" * 50)
print("Training completed successfully!")
print(f"Final Accuracy: {acc:.4f}")
print("=" * 50)