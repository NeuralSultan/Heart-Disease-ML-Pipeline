"""
04 - Supervised Learning: Classification Models
==================================================
Trains and evaluates Logistic Regression, Decision Tree, Random Forest,
and SVM on the selected features.
Requires 03_feature_selection.py to have been run first.

Run from the notebooks/ folder:
    python 04_supervised_learning.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, roc_curve
)

df_selected = pd.read_csv("../data/heart_disease_selected_features.csv")
X = df_selected.drop("target", axis=1)
y = df_selected["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="weighted", zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, average="weighted", zero_division=0))
    print("F1:", f1_score(y_test, y_pred, average="weighted", zero_division=0))
    print(classification_report(y_test, y_pred, zero_division=0))

# --- ROC Curves ---
plt.figure(figsize=(9, 7))
for name, model in models.items():
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_score = roc_auc_score(y_test, y_score)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - All Models")
plt.legend()
plt.tight_layout()
plt.savefig("../results/plots/07_roc_curves.png", dpi=120)
plt.close()

print("\nSaved plot: 07_roc_curves.png")
