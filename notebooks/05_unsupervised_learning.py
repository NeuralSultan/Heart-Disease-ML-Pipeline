"""
05 - Unsupervised Learning: Clustering
=========================================
Applies K-Means and Hierarchical Clustering and compares clusters
against actual diagnosis labels.
Requires 03_feature_selection.py to have been run first.

Run from the notebooks/ folder:
    python 05_unsupervised_learning.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score

df_selected = pd.read_csv("../data/heart_disease_selected_features.csv")
X_clustering = df_selected.drop("target", axis=1)

# --- Elbow method ---
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
plt.savefig("../results/plots/08_elbow_method.png", dpi=120)
plt.close()

# --- K-Means (K=2) vs actual labels ---
kmeans_final = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_clustering)
ari = adjusted_rand_score(df_selected["target"], clusters)
print("Adjusted Rand Index (KMeans, K=2 vs actual target):", ari)

# --- Hierarchical clustering dendrogram ---
linked = linkage(X_clustering, method="ward")
plt.figure(figsize=(10, 6))
dendrogram(linked, truncate_mode="lastp", p=30, show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("../results/plots/09_dendrogram.png", dpi=120)
plt.close()

print("Saved plots: 08_elbow_method.png, 09_dendrogram.png")
