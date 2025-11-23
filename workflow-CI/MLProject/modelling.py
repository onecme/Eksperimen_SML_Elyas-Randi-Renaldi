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
print(f"Current working directory: {os.getcwd()}")

# ==============================================
# 0.1. CONFIGURE MLFLOW - CRITICAL FIX!
# ==============================================
# Set tracking URI ke relative path untuk menghindari Windows path issue
mlflow_tracking_uri = f"file:{os.path.abspath('./mlruns')}"
mlflow.set_tracking_uri(mlflow_tracking_uri)
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# CRITICAL: Set artifact location explicitly
os.environ['MLFLOW_ARTIFACT_ROOT'] = os.path.abspath('./mlruns')
print(f"MLflow artifact root: {os.environ.get('MLFLOW_ARTIFACT_ROOT')}")

# ==============================================
# 1. LOAD DATASET
# ==============================================
print("Loading dataset...")
df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")

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
print("Applying SMOTE-Tomek resampling...")
smote_tomek = SMOTETomek(random_state=32)
X_train_res, y_train_res = smote_tomek.fit_resample(X_train_scaled, y_train)
print(f"Resampled training set shape: {X_train_res.shape}")

# ==============================================
# 2. TRAINING MODEL
# ==============================================
print("Training Random Forest model...")
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
print("Model training completed")

# ==============================================
# 3. EVALUASI
# ==============================================
acc = accuracy_score(y_test, pred)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred))

# ==============================================
# 4. MLFLOW LOGGING
# ==============================================
print("\n" + "="*50)
print("Logging to MLflow...")
print("="*50)

# Log parameters
mlflow.log_param("n_estimators", 300)
mlflow.log_param("max_depth", 7)
mlflow.log_param("min_samples_split", 5)
mlflow.log_param("min_samples_leaf", 3)
mlflow.log_param("max_features", "sqrt")
mlflow.log_param("class_weight", "balanced")
mlflow.log_param("test_size", 0.2)
mlflow.log_param("resampling", "SMOTETomek")
print("Parameters logged")

# Log metrics
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("training_samples", len(X_train_res))
mlflow.log_metric("test_samples", len(X_test))
print("Metrics logged")

# ==============================================
# 5. CONFUSION MATRIX
# ==============================================
print("\nCreating confusion matrix...")
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

# Save to absolute path to avoid issues
cm_path = os.path.join(os.getcwd(), "confusion_matrix_rf.png")
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
        print("Temporary file cleaned up")
except Exception as e:
    print(f"Warning: Could not log confusion matrix: {e}")

# ==============================================
# 6. LOG MODEL - MOST IMPORTANT!
# ==============================================
print("\nLogging model to MLflow...")
try:
    # Log model WITHOUT registered_model_name to avoid registry issues
    mlflow.sklearn.log_model(
        sk_model=model_rf,
        artifact_path="model"
        # Removed: registered_model_name="RandomForest_Placement_Model"
    )
    print("Model successfully logged to MLflow!")
except Exception as e:
    print(f"ERROR: Failed to log model: {e}")
    import traceback
    traceback.print_exc()
    raise

# ==============================================
# 7. SUMMARY
# ==============================================
print("\n" + "="*50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"Final Accuracy: {acc:.4f}")
print(f"Model saved to: {mlflow.get_artifact_uri()}")
print("="*50)