"""
Comprehensive Machine Learning Full Pipeline on Heart Disease UCI Dataset
==========================================================================
Runs the complete pipeline end-to-end:
  1. Data preprocessing & cleaning
  2. Dimensionality reduction (PCA)
  3. Feature selection (Feature Importance, RFE, Chi-Square)
  4. Supervised learning (Logistic Regression, Decision Tree, Random Forest, SVM)
  5. Unsupervised learning (K-Means, Hierarchical Clustering)
  6. Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
  7. Model export (.pkl)

All plots are saved to results/plots/ and all metrics to results/evaluation_metrics.txt
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV
)
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, roc_curve, adjusted_rand_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_disease.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
RESULTS_TXT = os.path.join(BASE_DIR, "results", "evaluation_metrics.txt")
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

sns.set(style="whitegrid")

# ---------------------------------------------------------------------------
# 1. DATA PREPROCESSING & CLEANING
# ---------------------------------------------------------------------------
log("=" * 70)
log("1. DATA PREPROCESSING & CLEANING")
log("=" * 70)

df = pd.read_csv(DATA_PATH)
log(f"Dataset shape: {df.shape}")
log(f"Missing values per column:\n{df.isna().sum()}")

# No missing values in this dataset, but keep a defensive cleaning step
df = df.dropna().reset_index(drop=True)

numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_cols = ["cp", "restecg", "slope", "thal", "ca"]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_encoded = df_encoded.astype({c: int for c in df_encoded.columns if df_encoded[c].dtype == bool})

scaler = StandardScaler()
df_scaled = df_encoded.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

df_final = df_scaled.copy()
log(f"\nFinal preprocessed shape: {df_final.shape}")

# EDA plots
df.hist(bins=20, figsize=(16, 12), edgecolor="black")
plt.suptitle("Feature Distributions", fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_feature_distributions.png"), dpi=120)
plt.close()

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_correlation_heatmap.png"), dpi=120)
plt.close()

plt.figure(figsize=(16, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x="target", y=col, data=df)
    plt.title(f"{col} by Target")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_boxplots_by_target.png"), dpi=120)
plt.close()

log("Saved: 01_feature_distributions.png, 02_correlation_heatmap.png, 03_boxplots_by_target.png")

# ---------------------------------------------------------------------------
# 2. DIMENSIONALITY REDUCTION - PCA
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("2. DIMENSIONALITY REDUCTION (PCA)")
log("=" * 70)

features_all = df_final.drop("target", axis=1)
target_all = df_final["target"]

pca = PCA()
pca.fit(features_all)
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)
n_components_95 = int(np.argmax(cumulative_var >= 0.95) + 1)
log(f"Components needed for 95% variance: {n_components_95} / {len(explained_var)}")

plt.figure(figsize=(9, 6))
plt.plot(range(1, len(explained_var) + 1), cumulative_var, marker="o", linestyle="--")
plt.axhline(y=0.95, color="r", linestyle="--", label="95% Variance Threshold")
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_pca_cumulative_variance.png"), dpi=120)
plt.close()

pca_2 = PCA(n_components=2)
pca_2_comp = pca_2.fit_transform(features_all)
pca_df = pd.DataFrame(pca_2_comp, columns=["PC1", "PC2"])
pca_df["target"] = target_all.values

plt.figure(figsize=(8, 6))
sns.scatterplot(x="PC1", y="PC2", hue="target", palette="Set1", data=pca_df, s=60)
plt.title("PCA - First Two Components")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_pca_scatter.png"), dpi=120)
plt.close()

log("Saved: 04_pca_cumulative_variance.png, 05_pca_scatter.png")

# ---------------------------------------------------------------------------
# 3. FEATURE SELECTION
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("3. FEATURE SELECTION")
log("=" * 70)

rf_fs = RandomForestClassifier(random_state=42)
rf_fs.fit(features_all, target_all)
importances = pd.Series(rf_fs.feature_importances_, index=features_all.columns)
importances_sorted = importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importances_sorted.plot(kind="bar")
plt.title("Feature Importance from Random Forest")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_feature_importance.png"), dpi=120)
plt.close()

model_rfe = LogisticRegression(max_iter=2000)
rfe = RFE(model_rfe, n_features_to_select=10)
rfe.fit(features_all, target_all)
selected_rfe = features_all.columns[rfe.support_]
log(f"Selected by RFE: {list(selected_rfe)}")

features_chi = features_all.copy()
features_chi[features_chi < 0] = 0
chi2_selector = SelectKBest(score_func=chi2, k=10)
chi2_selector.fit(features_chi, target_all)
selected_chi2 = features_all.columns[chi2_selector.get_support()]
log(f"Selected by Chi-Square: {list(selected_chi2)}")

final_selected_features = sorted(set(selected_rfe) | set(selected_chi2))
log(f"Final combined feature set ({len(final_selected_features)} features): {final_selected_features}")

df_selected = df_final[final_selected_features + ["target"]]

log("Saved: 06_feature_importance.png")

# ---------------------------------------------------------------------------
# 4. SUPERVISED LEARNING - CLASSIFICATION MODELS
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("4. SUPERVISED LEARNING - CLASSIFICATION MODELS")
log("=" * 70)

X = df_selected.drop("target", axis=1)
y = df_selected["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}

results_summary = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    results_summary[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    log(f"\n=== {name} ===")
    log(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    log(classification_report(y_test, y_pred, zero_division=0))

# ROC curves (binary target: 0/1)
plt.figure(figsize=(9, 7))
for name, model in models.items():
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_score = roc_auc_score(y_test, y_score)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")
    results_summary[name]["auc"] = auc_score

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - All Models")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_roc_curves.png"), dpi=120)
plt.close()

log("\nSaved: 07_roc_curves.png")

# ---------------------------------------------------------------------------
# 5. UNSUPERVISED LEARNING - CLUSTERING
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("5. UNSUPERVISED LEARNING - CLUSTERING")
log("=" * 70)

X_clustering = df_selected.drop("target", axis=1)

inertia = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_clustering)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "08_elbow_method.png"), dpi=120)
plt.close()

# Fit final KMeans with K=2 (matches binary target, for comparison)
kmeans_final = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_clustering)
ari = adjusted_rand_score(df_selected["target"], clusters)
log(f"Adjusted Rand Index (KMeans, K=2 vs actual target): {ari:.4f}")

linked = linkage(X_clustering, method="ward")
plt.figure(figsize=(10, 6))
dendrogram(linked, truncate_mode="lastp", p=30, show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "09_dendrogram.png"), dpi=120)
plt.close()

log("Saved: 08_elbow_method.png, 09_dendrogram.png")

# ---------------------------------------------------------------------------
# 6. HYPERPARAMETER TUNING
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("6. HYPERPARAMETER TUNING")
log("=" * 70)

param_grid_rf = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5],
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring="accuracy")
grid_rf.fit(X_train, y_train)
log(f"Best RF params: {grid_rf.best_params_}")
log(f"Best RF CV accuracy: {grid_rf.best_score_:.4f}")

param_dist_svm = {
    "C": uniform(0.1, 10),
    "gamma": ["scale", "auto"],
    "kernel": ["rbf", "linear"],
}
random_svm = RandomizedSearchCV(
    SVC(probability=True), param_distributions=param_dist_svm,
    n_iter=20, cv=5, scoring="accuracy", random_state=42
)
random_svm.fit(X_train, y_train)
log(f"Best SVM params: {random_svm.best_params_}")

y_pred_rf = grid_rf.best_estimator_.predict(X_test)
y_pred_svm = random_svm.best_estimator_.predict(X_test)

rf_opt_acc = accuracy_score(y_test, y_pred_rf)
svm_opt_acc = accuracy_score(y_test, y_pred_svm)
log(f"\nOptimized Random Forest test accuracy: {rf_opt_acc:.4f}")
log(classification_report(y_test, y_pred_rf, zero_division=0))
log(f"Optimized SVM test accuracy: {svm_opt_acc:.4f}")
log(classification_report(y_test, y_pred_svm, zero_division=0))

# Comparison bar chart: baseline vs optimized
plt.figure(figsize=(8, 5))
labels = ["Random Forest\n(baseline)", "Random Forest\n(optimized)", "SVM\n(baseline)", "SVM\n(optimized)"]
values = [results_summary["Random Forest"]["accuracy"], rf_opt_acc,
          results_summary["SVM"]["accuracy"], svm_opt_acc]
bars = plt.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
plt.ylabel("Test Accuracy")
plt.title("Baseline vs. Hyperparameter-Tuned Accuracy")
plt.ylim(0, 1)
for bar, v in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "10_baseline_vs_tuned.png"), dpi=120)
plt.close()

log("\nSaved: 10_baseline_vs_tuned.png")

# ---------------------------------------------------------------------------
# 7. MODEL EXPORT & DEPLOYMENT
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("7. MODEL EXPORT")
log("=" * 70)

# Best model overall = optimized Random Forest (typically best performer)
best_model = grid_rf.best_estimator_ if rf_opt_acc >= svm_opt_acc else random_svm.best_estimator_
best_name = "Random Forest (tuned)" if rf_opt_acc >= svm_opt_acc else "SVM (tuned)"

from sklearn.pipeline import Pipeline
final_pipeline = Pipeline([
    ("model", best_model),
])
final_pipeline.fit(X_train, y_train)
joblib.dump(final_pipeline, MODEL_PATH)
joblib.dump(list(X.columns), os.path.join(os.path.dirname(MODEL_PATH), "feature_columns.pkl"))
joblib.dump(scaler, os.path.join(os.path.dirname(MODEL_PATH), "scaler.pkl"))

log(f"Best model: {best_name}")
log(f"Saved pipeline to: {MODEL_PATH}")

# ---------------------------------------------------------------------------
# Write results file
# ---------------------------------------------------------------------------
with open(RESULTS_TXT, "w") as f:
    f.write("\n".join(log_lines))

log(f"\nAll results written to {RESULTS_TXT}")
log("Pipeline complete.")
