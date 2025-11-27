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
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pkg_resources
from pathlib import Path


# Setup output path
OUTPUT_PATH = Path(r"C:\sistemMachienlearning\membangunModel")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv("membangunModel/processed_dataset.csv")

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


# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Placement_Modelling_Tuning.py")


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


with mlflow.start_run():

    # Log parameters
    mlflow.log_params(best_params)

    # Train and predict
    best_model.fit(X_train_res, y_train_res)
    pred = best_model.predict(X_test_scaled)
    pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    print("Best Params:", best_params)
    print("Accuracy:", acc)
    print(classification_report(y_test, pred))


    # 1. CONFUSION MATRIX
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Placed", "Placed"],
                yticklabels=["Not Placed", "Placed"])
    plt.title("Confusion Matrix - Random Forest (Tuned)", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    cm_path = OUTPUT_PATH / "confusion_matrix_modelling_tuning.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    mlflow.log_artifact(str(cm_path))


    # 2. CLASSIFICATION REPORT (Visualized)
    report = classification_report(y_test, pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1].astype(float),
                annot=True, fmt=".3f", cmap="YlGnBu",
                cbar_kws={'label': 'Score'}, ax=ax)
    plt.title("Classification Report Heatmap", fontsize=14, fontweight='bold')
    plt.ylabel("Class", fontsize=11)
    plt.xlabel("Metrics", fontsize=11)
    plt.tight_layout()
    report_path = OUTPUT_PATH / "classification_report_modelling_tuning.png"
    plt.savefig(report_path, dpi=300)
    plt.close()
    mlflow.log_artifact(str(report_path))



    # 3. ROC CURVE
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('ROC Curve - Random Forest (Tuned)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = OUTPUT_PATH / "roc_curve_modelling_tuning.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    mlflow.log_artifact(str(roc_path))
    mlflow.log_metric("roc_auc", roc_auc)


    # 4. PRECISION-RECALL CURVE
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, pred_proba)
    pr_auc = auc(recall_curve, precision_curve)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = OUTPUT_PATH / "precision_recall_curve_modelling_tuning.png"
    plt.savefig(pr_path, dpi=300)
    plt.close()
    mlflow.log_artifact(str(pr_path))
    mlflow.log_metric("pr_auc", pr_auc)


    # 5. FEATURE IMPORTANCE
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(5), x='importance', y='feature', palette='viridis')
    plt.title('Top 5 Feature Importances', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=11)
    plt.ylabel('Features', fontsize=11)
    plt.tight_layout()
    fi_path = OUTPUT_PATH / "feature_importance_modelling_tuning.png"
    plt.savefig(fi_path, dpi=300)
    plt.close()
    mlflow.log_artifact(str(fi_path))


    # 6. PREDICTION DISTRIBUTION
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Predicted probabilities distribution
    axes[0].hist(pred_proba[y_test == 0], bins=30, alpha=0.6, label='Not Placed', color='red')
    axes[0].hist(pred_proba[y_test == 1], bins=30, alpha=0.6, label='Placed', color='green')
    axes[0].set_xlabel('Predicted Probability', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Predicted Probabilities', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Prediction counts
    pred_counts = pd.Series(pred).value_counts().sort_index()
    true_counts = pd.Series(y_test).value_counts().sort_index()
    x_pos = np.arange(len(pred_counts))
    width = 0.35

    axes[1].bar(x_pos - width/2, true_counts.values, width, label='True', color='skyblue')
    axes[1].bar(x_pos + width/2, pred_counts.values, width, label='Predicted', color='orange')
    axes[1].set_xlabel('Class', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('True vs Predicted Class Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(['Not Placed', 'Placed'])
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    dist_path = OUTPUT_PATH / "prediction_distribution_modelling_tuning.png"
    plt.savefig(dist_path, dpi=300)
    plt.close()
    mlflow.log_artifact(str(dist_path))


    # 7. LEARNING CURVE (GridSearchCV Results)
    cv_results = pd.DataFrame(grid.cv_results_)

    plt.figure(figsize=(12, 6))

    # Group by n_estimators for visualization
    for n_est in param_grid['n_estimators']:
        subset = cv_results[cv_results['param_n_estimators'] == n_est]
        plt.plot(subset.index, subset['mean_test_score'],
                marker='o', label=f'n_estimators={n_est}')

    plt.xlabel('Parameter Configuration Index', fontsize=11)
    plt.ylabel('Mean CV Accuracy', fontsize=11)
    plt.title('GridSearchCV Performance Across Configurations', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    cv_path = OUTPUT_PATH / "gridsearch_performance_modelling_tuning.png"
    plt.savefig(cv_path, dpi=300)
    plt.close()
    mlflow.log_artifact(str(cv_path))


    # 8. METRICS SUMMARY VISUALIZATION
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'PR AUC'],
        'Score': [acc, prec, rec, f1, roc_auc, pr_auc]
    }
    metrics_df = pd.DataFrame(metrics_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(metrics_df['Metric'], metrics_df['Score'], color='teal', alpha=0.7)

    # Add value labels on bars
    for i, (metric, score) in enumerate(zip(metrics_df['Metric'], metrics_df['Score'])):
        ax.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Score', fontsize=11)
    ax.set_title('Model Performance Metrics Summary', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.1])
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    metrics_path = OUTPUT_PATH / "metrics_summary_modelling_tuning.png"
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    mlflow.log_artifact(str(metrics_path))


    # 9. MODEL INFO TEXT FILE
    model_info_path = OUTPUT_PATH / "model_info_modelling_tuning.txt"
    with open(model_info_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RANDOM FOREST MODEL - TUNING RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Parameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nPerformance Metrics:\n")
        f.write(f"  Accuracy:  {acc:.4f}\n")
        f.write(f"  Precision: {prec:.4f}\n")
        f.write(f"  Recall:    {rec:.4f}\n")
        f.write(f"  F1-Score:  {f1:.4f}\n")
        f.write(f"  ROC AUC:   {roc_auc:.4f}\n")
        f.write(f"  PR AUC:    {pr_auc:.4f}\n")
        f.write(f"\nDataset Info:\n")
        f.write(f"  Training samples: {len(X_train_res)}\n")
        f.write(f"  Test samples:     {len(X_test)}\n")
        f.write(f"  Number of features: {X.shape[1]}\n")
    mlflow.log_artifact(str(model_info_path))


    # Log the trained model
    mlflow.sklearn.log_model(best_model, "model_tuned")

    print("\n" + "="*60)
    print("All artifacts logged to MLflow successfully")
    print(f"Local artifacts saved to: {OUTPUT_PATH}")
    print("="*60)


required_packages = [
    'pandas',
    'mlflow',
    'scikit-learn',
    'imbalanced-learn',
    'matplotlib',
    'seaborn',
    'numpy'
]

requirements_list = []
for package in required_packages:
    try:
        version = pkg_resources.get_distribution(package).version
        requirements_list.append(f"{package}=={version}")
        print(f"{package}=={version}")
    except:
        requirements_list.append(package)
        print(f"{package} (version not found, added without version)")

requirements_path = OUTPUT_PATH / "requirements_modelling_tuning.txt"
with open(requirements_path, 'w') as f:
    f.write('\n'.join(requirements_list))

print(f"\nrequirements_modelling_tuning.txt saved to: {requirements_path}")
print("="*60)