"""
06 - Hyperparameter Tuning
=============================
Optimizes Random Forest (GridSearchCV) and SVM (RandomizedSearchCV),
compares against baseline, and exports the final model pipeline.
Requires 03_feature_selection.py to have been run first.

Run from the notebooks/ folder:
    python 06_hyperparameter_tuning.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from scipy.stats import uniform
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

df_selected = pd.read_csv("../data/heart_disease_selected_features.csv")
X = df_selected.drop("target", axis=1)
y = df_selected["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- GridSearchCV - Random Forest ---
param_grid_rf = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5],
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring="accuracy")
grid_rf.fit(X_train, y_train)
print("Best RF params:", grid_rf.best_params_)
print("Best RF CV accuracy:", grid_rf.best_score_)

# --- RandomizedSearchCV - SVM ---
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
print("Best SVM params:", random_svm.best_params_)

# --- Compare optimized models on the test set ---
y_pred_rf = grid_rf.best_estimator_.predict(X_test)
y_pred_svm = random_svm.best_estimator_.predict(X_test)

rf_opt_acc = accuracy_score(y_test, y_pred_rf)
svm_opt_acc = accuracy_score(y_test, y_pred_svm)

print("\nOptimized Random Forest accuracy:", rf_opt_acc)
print(classification_report(y_test, y_pred_rf, zero_division=0))
print("Optimized SVM accuracy:", svm_opt_acc)
print(classification_report(y_test, y_pred_svm, zero_division=0))

# --- Comparison bar chart ---
plt.figure(figsize=(8, 5))
labels = ["Random Forest\n(tuned)", "SVM\n(tuned)"]
values = [rf_opt_acc, svm_opt_acc]
bars = plt.bar(labels, values, color=["#55A868", "#8172B2"])
plt.ylabel("Test Accuracy")
plt.title("Tuned Model Comparison")
plt.ylim(0, 1)
for bar, v in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig("../results/plots/10_baseline_vs_tuned.png", dpi=120)
plt.close()

# --- Export the best model as a pipeline ---
best_model = grid_rf.best_estimator_ if rf_opt_acc >= svm_opt_acc else random_svm.best_estimator_
best_name = "Random Forest (tuned)" if rf_opt_acc >= svm_opt_acc else "SVM (tuned)"
print("\nBest model:", best_name)

final_pipeline = Pipeline([("model", best_model)])
final_pipeline.fit(X_train, y_train)
joblib.dump(final_pipeline, "../models/final_model.pkl")
joblib.dump(list(X.columns), "../models/feature_columns.pkl")

print("Model exported to ../models/final_model.pkl")
print("Saved plot: 10_baseline_vs_tuned.png")
