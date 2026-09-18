"""
03 - Feature Selection
========================
Ranks and selects the most relevant features using Random Forest
importance, RFE, and the Chi-Square test.
Requires 01_data_preprocessing.py to have been run first.

Run from the notebooks/ folder:
    python 03_feature_selection.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectKBest, chi2

df_final = pd.read_csv("../data/heart_disease_cleaned.csv")
features = df_final.drop("target", axis=1)
target = df_final["target"]

# --- Feature importance (Random Forest) ---
rf = RandomForestClassifier(random_state=42)
rf.fit(features, target)

importances = pd.Series(rf.feature_importances_, index=features.columns)
importances_sorted = importances.sort_values(ascending=False)
print("Random Forest feature importances:")
print(importances_sorted)

plt.figure(figsize=(10, 6))
importances_sorted.plot(kind="bar")
plt.title("Feature Importance from Random Forest")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("../results/plots/06_feature_importance.png", dpi=120)
plt.close()

# --- Recursive Feature Elimination (RFE) ---
model = LogisticRegression(max_iter=2000)
rfe = RFE(model, n_features_to_select=10)
rfe.fit(features, target)
selected_rfe = features.columns[rfe.support_]
print("\nSelected by RFE:", list(selected_rfe))

# --- Chi-Square Test ---
features_chi = features.copy()
features_chi[features_chi < 0] = 0

chi2_selector = SelectKBest(score_func=chi2, k=10)
chi2_selector.fit(features_chi, target)
selected_chi2 = features.columns[chi2_selector.get_support()]
print("Selected by Chi-Square:", list(selected_chi2))

# --- Combine selected features (union) ---
final_selected_features = sorted(set(selected_rfe) | set(selected_chi2))
print(f"\nFinal combined feature set ({len(final_selected_features)} features):")
print(final_selected_features)

df_selected = df_final[final_selected_features + ["target"]]
df_selected.to_csv("../data/heart_disease_selected_features.csv", index=False)

print("\nSaved plot: 06_feature_importance.png")
print("Saved selected-feature dataset to ../data/heart_disease_selected_features.csv")
