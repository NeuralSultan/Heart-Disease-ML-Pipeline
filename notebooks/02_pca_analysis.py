"""
02 - Dimensionality Reduction (PCA)
=====================================
Applies PCA to the cleaned dataset and visualizes explained variance.
Requires 01_data_preprocessing.py to have been run first.

Run from the notebooks/ folder:
    python 02_pca_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

df_final = pd.read_csv("../data/heart_disease_cleaned.csv")
features = df_final.drop("target", axis=1)
target = df_final["target"]

# --- Explained variance ---
pca = PCA()
pca.fit(features)
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)
n_components_95 = int(np.argmax(cumulative_var >= 0.95) + 1)
print(f"Components needed for 95% variance: {n_components_95} / {len(explained_var)}")

plt.figure(figsize=(9, 6))
plt.plot(range(1, len(explained_var) + 1), cumulative_var, marker="o", linestyle="--")
plt.axhline(y=0.95, color="r", linestyle="--", label="95% Variance Threshold")
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.legend()
plt.tight_layout()
plt.savefig("../results/plots/04_pca_cumulative_variance.png", dpi=120)
plt.close()

# --- 2D PCA projection ---
pca_2 = PCA(n_components=2)
pca_2_comp = pca_2.fit_transform(features)
pca_df = pd.DataFrame(pca_2_comp, columns=["PC1", "PC2"])
pca_df["target"] = target.values

plt.figure(figsize=(8, 6))
sns.scatterplot(x="PC1", y="PC2", hue="target", palette="Set1", data=pca_df, s=60)
plt.title("PCA - First Two Components")
plt.tight_layout()
plt.savefig("../results/plots/05_pca_scatter.png", dpi=120)
plt.close()

print("Saved plots: 04_pca_cumulative_variance.png, 05_pca_scatter.png")
