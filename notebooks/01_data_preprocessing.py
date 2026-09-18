"""
01 - Data Preprocessing & Cleaning
====================================
Loads the Heart Disease UCI dataset, handles missing values, encodes
categorical features, scales numeric features, and produces EDA plots.

Run from the notebooks/ folder:
    python 01_data_preprocessing.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # saves plots to file instead of popping up a window
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")

# --- Load the dataset ---
df = pd.read_csv("../data/heart_disease.csv")
print("Dataset shape:", df.shape)
print(df.head())

# --- Check for missing values ---
print("\nMissing values per column:")
print(df.isna().sum())
df = df.dropna().reset_index(drop=True)

# --- Encode categorical variables and scale numeric features ---
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_cols = ["cp", "restecg", "slope", "thal", "ca"]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_encoded = df_encoded.astype({c: int for c in df_encoded.columns if df_encoded[c].dtype == bool})

scaler = StandardScaler()
df_scaled = df_encoded.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

df_final = df_scaled.copy()
print("\nFinal preprocessed shape:", df_final.shape)

# --- Exploratory Data Analysis ---
df.hist(bins=20, figsize=(16, 12), edgecolor="black")
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("../results/plots/01_feature_distributions.png", dpi=120)
plt.close()

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("../results/plots/02_correlation_heatmap.png", dpi=120)
plt.close()

plt.figure(figsize=(16, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x="target", y=col, data=df)
    plt.title(f"{col} by Target")
plt.tight_layout()
plt.savefig("../results/plots/03_boxplots_by_target.png", dpi=120)
plt.close()

print("\nSaved plots: 01_feature_distributions.png, 02_correlation_heatmap.png, 03_boxplots_by_target.png")

# --- Save cleaned dataset for downstream scripts ---
df_final.to_csv("../data/heart_disease_cleaned.csv", index=False)
print("Cleaned dataset saved to ../data/heart_disease_cleaned.csv")
