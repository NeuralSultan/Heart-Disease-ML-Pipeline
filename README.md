# Comprehensive Machine Learning Full Pipeline on Heart Disease UCI Dataset

An end-to-end machine learning project that analyzes, predicts, and visualizes heart
disease risk using the Heart Disease UCI dataset. The pipeline covers data
preprocessing, dimensionality reduction (PCA), feature selection, supervised
classification, unsupervised clustering, hyperparameter tuning, and deployment
through a Streamlit web app.

## 1. General Description

This project applies a full classical ML workflow to the Heart Disease UCI dataset:

- **Data preprocessing**: cleaning, encoding, and scaling
- **Dimensionality reduction**: PCA with explained-variance analysis
- **Feature selection**: Random Forest importance, RFE, and Chi-Square test
- **Supervised learning**: Logistic Regression, Decision Tree, Random Forest, SVM
- **Unsupervised learning**: K-Means (Elbow Method) and Hierarchical Clustering
- **Hyperparameter tuning**: GridSearchCV and RandomizedSearchCV
- **Deployment**: a Streamlit UI for real-time prediction, deployable via Ngrok

## 2. Results Summary

Baseline model comparison (test set, 20% holdout):

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|---------:|----------:|-------:|---------:|
| Logistic Regression |   0.869  |    0.882  |  0.869 |   0.866  |
| Decision Tree        |   0.721  |    0.721  |  0.721 |   0.720  |
| Random Forest        |   0.853  |    0.860  |  0.853 |   0.850  |
| SVM                   |   0.853  |    0.870  |  0.853 |   0.849  |

After hyperparameter tuning:

| Model                  | Test Accuracy |
|-------------------------|--------------:|
| Random Forest (tuned)   |     0.836     |
| SVM (tuned)              |     0.853     |

- **Best model deployed**: SVM (tuned) — `models/final_model.pkl`
- **Clustering vs. actual labels**: Adjusted Rand Index (K-Means, K=2) ≈ 0.30,
  indicating the natural K-Means clusters only partially align with the true
  diagnosis labels — expected, since clustering is unsupervised and unaware of
  the target.

Full logs and per-class metrics are in [`results/evaluation_metrics.txt`](results/evaluation_metrics.txt).
All generated charts are in [`results/plots/`](results/plots/).

## 3. Dataset

[Heart Disease UCI Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
— 303 patient records, 13 clinical features (age, sex, chest pain type, resting
blood pressure, cholesterol, etc.) plus a binary target (presence/absence of
heart disease). A local copy is included at `data/heart_disease.csv`.

## 4. How to Run

### Reproduce the full pipeline (plots + model)
```bash
pip install -r requirements.txt
python run_pipeline.py
```
This regenerates every plot in `results/plots/`, the metrics log in
`results/evaluation_metrics.txt`, and the trained model in `models/final_model.pkl`.

### Run step-by-step scripts
Run the scripts in `notebooks/` in order (01 → 06) from inside that folder —
each one saves its own plots/artifacts and some write intermediate CSVs the
next script depends on:
```bash
cd notebooks
python 01_data_preprocessing.py
python 02_pca_analysis.py
python 03_feature_selection.py
python 04_supervised_learning.py
python 05_unsupervised_learning.py
python 06_hyperparameter_tuning.py
```
These mirror `run_pipeline.py` stage by stage and work great as plain `.py`
files in VS Code — just open the folder, open a terminal, and run each script.

### Run the Streamlit app
```bash
streamlit run ui/app.py
```
The app loads `models/final_model.pkl` and lets you enter patient data for a
real-time prediction. See [`deployment/ngrok_setup.txt`](deployment/ngrok_setup.txt)
for exposing it publicly via Ngrok.

## 5. File Structure

```
Heart_Disease_Project/
├── data/
│   └── heart_disease.csv
├── notebooks/
│   ├── 01_data_preprocessing.py
│   ├── 02_pca_analysis.py
│   ├── 03_feature_selection.py
│   ├── 04_supervised_learning.py
│   ├── 05_unsupervised_learning.py
│   └── 06_hyperparameter_tuning.py
├── models/
│   ├── final_model.pkl
│   ├── feature_columns.pkl
│   └── scaler.pkl
├── ui/
│   └── app.py
├── deployment/
│   └── ngrok_setup.txt
├── results/
│   ├── evaluation_metrics.txt
│   └── plots/
│       ├── 01_feature_distributions.png
│       ├── 02_correlation_heatmap.png
│       ├── 03_boxplots_by_target.png
│       ├── 04_pca_cumulative_variance.png
│       ├── 05_pca_scatter.png
│       ├── 06_feature_importance.png
│       ├── 07_roc_curves.png
│       ├── 08_elbow_method.png
│       ├── 09_dendrogram.png
│       └── 10_baseline_vs_tuned.png
├── run_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 6. Tools Used

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SciPy, Streamlit, Ngrok.

## 7. Limitations

- Small dataset (303 records) — results should be interpreted with appropriate
  caution regarding generalization.
- This is an educational project, **not a medical diagnostic tool**.
