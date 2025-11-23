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
# 2. MLflow Setup - Use existing run from MLflow Project
# ==============================================
# JANGAN set tracking URI atau experiment - biarkan MLflow Project yang handle
# JANGAN gunakan autolog() - manual logging saja

# ==============================================
# 3. TRAINING MODEL
# ==============================================
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

print("Training model...")
model_rf.fit(X_train_res, y_train_res)
pred = model_rf.predict(X_test_scaled)

# ==============================================
# 4. EVALUASI & LOGGING
# ==============================================
acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc}")
print(classification_report(y_test, pred))

# Get the active run created by MLflow Project
active_run = mlflow.active_run()
if active_run:
    print(f"Using active run: {active_run.info.run_id}")
    
    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 7)
    mlflow.log_param("min_samples_split", 5)
    mlflow.log_param("min_samples_leaf", 3)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    
    # Save Confusion Matrix dengan path yang aman
    cm_path = os.path.join(os.getcwd(), "confusion_matrix_rf.png")
    print(f"Saving confusion matrix to: {cm_path}")
    plt.savefig(cm_path)
    plt.close()
    
    # Log artifact to MLflow
    try:
        mlflow.log_artifact(cm_path)
        print(f"Successfully logged artifact: {cm_path}")
        
        # Clean up file setelah di-log
        if os.path.exists(cm_path):
            os.remove(cm_path)
            print(f"Cleaned up temporary file: {cm_path}")
    except Exception as e:
        print(f"Warning: Could not log artifact: {e}")
    
    # ==============================================
    # 5. LOG MODEL 
    # ==============================================
    print("Logging model to MLflow...")
    mlflow.sklearn.log_model(
        sk_model=model_rf,
        artifact_path="model",
        registered_model_name="RandomForest_Placement_Model"
    )
    
    print("Training completed successfully!")
else:
    print("Warning: No active MLflow run found. Metrics will not be logged.")
    print(f"Model accuracy: {acc}")