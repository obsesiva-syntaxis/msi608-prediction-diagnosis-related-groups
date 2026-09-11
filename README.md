# Diagnosis-Related Group (GRD) Prediction — MSI608

Prediction of the Diagnosis-Related Group (Grupo Relacionado de Diagnóstico, GRD) of a hospital discharge from clinical codes, developed for the **MSI608 — Special Topics in Data Science** course, Master's in Computer Engineering (Magíster en Ingeniería Informática), Universidad Andrés Bello (UNAB).

This repository accompanies the IEEE Access-format paper *"Prediction of Diagnosis-Related Groups"*, which frames GRD assignment as a multiclass classification problem over the 20 most frequent GRDs in a real hospital discharge dataset (Hospital El Pino), and compares an embedding + LSTM sequence model against classical machine learning baselines.

---

## Overview

| | |
|---|---|
| **Dataset** | 14,561 hospital discharges (Hospital El Pino), 68 raw fields per record (up to 27 ICD-10 diagnoses, up to 30 ICD-9-CM procedures, age, sex, GRD) |
| **Task** | Multiclass classification — predict the GRD code from clinical codes, age, and sex |
| **Classes** | Top 20 most frequent GRDs, 5,861 discharges retained (40.3% of the full dataset) |
| **Models compared** | Dummy (majority class, baseline), Logistic Regression, Random Forest, XGBoost, Embedding + LSTM (sequence of clinical codes + age/sex metadata) |
| **Best model** | Embedding + LSTM — Accuracy 0.944, F1 macro 0.932, F1 weighted 0.944, macro AUC (OvR) 0.995 (test set) |
| **Data split** | Stratified 70/15/15 train/validation/test (4,102 / 879 / 880 discharges) |

## Results Summary

**Test set (n = 880):**

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|---|---|---|---|---|---|
| **Embedding + LSTM** | **0.9443** | **0.9320** | **0.9339** | **0.9320** | **0.9436** |
| XGBoost | 0.9420 | 0.9344 | 0.9251 | 0.9281 | 0.9404 |
| Logistic Regression | 0.9284 | 0.9340 | 0.9024 | 0.9126 | 0.9241 |
| Random Forest | 0.8705 | 0.9260 | 0.8041 | 0.8008 | 0.8364 |
| Dummy (majority class) | 0.1386 | 0.0069 | 0.0500 | 0.0122 | 0.0338 |

The LSTM was selected as the best model based on **macro F1** (to avoid favoring the majority class in a strongly imbalanced label distribution) and achieves a macro-averaged one-vs-rest AUC of **0.995** across the 20 GRDs, with per-class AUC ranging from 0.974 to 1.000. Full per-class metrics (precision/recall/F1) and the confusion matrix are reported in the paper and reproducible from `analisis_multiclase.ipynb`.

## Repository Structure

```
.
├── dataset/
│   ├── dataset_elpino.csv                        # Raw discharge-level dataset (Hospital El Pino)
│   ├── IR-GRD V3.1 CON PRECIOS FONASA 2016.xlsx  # GRD code -> group name / price mapping table
│   ├── CIE-10.xlsx                                # ICD-10 diagnosis code dictionary
│   ├── CIE-9.xlsx                                 # ICD-9-CM procedure code dictionary
│   └── Tablas maestras bases GRD.xlsx             # Additional GRD reference tables
├── models/                                        # Trained model artifacts (LSTM, RF, scaler, label encoder, tokens)
├── figs/                                          # Figures used in the paper (distribution, architecture, training curves, confusion matrix, AUC, feature importance)
├── logs/                                          # TensorBoard training logs
├── analisis_base.ipynb                            # Baseline notebook: multi-hot embedding model
├── analisis_multiclase.ipynb                      # Full pipeline: EDA, feature engineering, model comparison, evaluation
├── generar_figuras.py                             # Reproduces all paper figures (ES/EN) from the dataset and saved models
└── Prediction Diagnosis Related Groups.pdf        # IEEE Access paper (6 pages, double column)
```

## Methodology (brief)

1. **Data quality & EDA** — completeness and outlier checks on age, sex, and clinical code fields; the 20 most frequent GRDs are retained, covering 40.3% of all discharges (5,861 of 14,561) while keeping the classification problem tractable.
2. **Feature engineering** — two parallel representations of the clinical codes (up to 27 ICD-10 diagnoses + up to 30 ICD-9-CM procedures per discharge): a frequency-ranked **token sequence** (padded to length 65) for the LSTM, and a **multi-hot vector** (vocabulary size 2,250) for the classical models, both combined with standardized age and sex.
3. **Stratified split** — 70% train / 15% validation / 15% test, stratified by GRD to preserve class proportions across an imbalanced label distribution (the majority class outnumbers the smallest class more than 5:1).
4. **Modeling** — a majority-class Dummy baseline, three classical classifiers (Logistic Regression, Random Forest, XGBoost) on the multi-hot representation, and an Embedding(2,250→64, mask_zero) + LSTM(64) branch fused with a small Dense branch over age/sex, followed by Dropout(0.3) and a softmax output over the 20 classes.
5. **Evaluation** — Accuracy, macro/weighted Precision, Recall, F1, and one-vs-rest AUC per class, computed on the held-out test set; model selection is based on macro F1 to avoid over-weighting the dominant GRD (cesarean delivery).

## ⚙️ Requirements

```
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow joblib
```

## How to Run

```
jupyter notebook analisis_multiclase.ipynb
```

The notebook reads directly from `dataset/`, reproduces the full EDA, feature engineering, model comparison, and evaluation reported in the paper, and saves the trained models to `models/`. `analisis_base.ipynb` contains the earlier multi-hot-only baseline, and `generar_figuras.py` regenerates every figure in `figs/` from the dataset and the saved model artifacts (`python generar_figuras.py [es|en]`).

## 👥 Team

- Eric Silva <https://github.com/eRICasl>
- Brian Guzman <https://github.com/bguzmanm>
- Miguel González <https://github.com/obsesiva-syntaxis>
- Edson Quevedo <https://github.com/braulio20aa>
- Jaime Rivera <https://github.com/jrlatin2>

**Course:** MSI608 — Special Topics in Data Science, Master's in Computer Engineering, Universidad Andrés Bello (UNAB)
**Delivery date:** September 12, 2026

## 📄 License

Academic project for course purposes — MSI608, UNAB.
