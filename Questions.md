# 📘 Machine Learning PBL Repository

This repository contains all **Machine Learning (ML) Problem-Based Learning (PBL)** questions categorized into:

* 🧠 **ISA (In-Sem Assessment)** → Theory + applied ML problems
* 🧪 **EXP (Experiments / Lab Work)** → Practical assignments

---

## ⚠️ Important Notes

* ❌ No ISA for Assignment 5
* ❌ ISA 2, 9, 10, 11 → Only theory (No PBL)

---

## 📂 Repository Structure

```
ML-PBL/
│
├── ISA/
│   ├── ISA1_BreastCancer.md
│   ├── ISA3_BankMarketing.md
│   ├── ISA4_BikeRegression.md
│   ├── ISA6_SpamClassification.md
│   ├── ISA7_SVM_Churn.md
│   └── ISA8_MST_Clustering.md
│
├── EXP/
│   ├── EXP1_Titanic.md
│   ├── EXP2_HousePrices.md
│   ├── EXP3_WhatsAppAnalyzer.md
│   ├── EXP4_BankBinary.md
│   ├── EXP5_HeartDisease.md
│   ├── EXP6_OlympicsEDA.md
│   ├── EXP7_BlogBot.md
│   ├── EXP8_MNIST.md
│   ├── EXP9_Churn.md
│   ├── EXP10_LDA.md
│   └── EXP11_SVD.md
│
└── README.md
```

---

## 🧠 ISA Overview

| ISA   | Topic                         |
| ----- | ----------------------------- |
| ISA 1 | Breast Cancer Classification  |
| ISA 3 | Bank Marketing Classification |
| ISA 4 | Bike Demand Regression        |
| ISA 6 | Spam Classification Ensembles |
| ISA 7 | SVM Churn Prediction          |
| ISA 8 | MST Clustering                |

---

## 🧪 EXP Overview

| EXP    | Topic                      |
| ------ | -------------------------- |
| EXP 1  | Titanic ML Workflow        |
| EXP 2  | House Prices Regression    |
| EXP 3  | WhatsApp Analyzer          |
| EXP 4  | Bank Binary Classification |
| EXP 5  | Heart Disease Prediction   |
| EXP 6  | Olympic EDA                |
| EXP 7  | Blog Writing Bot           |
| EXP 8  | MNIST Classification       |
| EXP 9  | Customer Churn             |
| EXP 10 | LDA                        |
| EXP 11 | SVD Topic Modeling         |

---

## 🚀 How to Use

1. Pick any ISA or EXP
2. Open corresponding `.md` file
3. Follow dataset + tasks
4. Implement and upload results

---


# 🧠 ISA (Theory + Applied Problems)

---

## 🔹 ISA 1

### Breast Cancer Diagnosis (Classification Pipeline + Generalization Error)

**Objective:**
Build a supervised ML model to classify tumors as Malignant (M) or Benign (B) and demonstrate the full ML workflow.

**Dataset:**
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

**Input Example:**

```bash
python task1_cancer_classification.py --data data.csv --test_size 0.2 --model logistic
```

**Expected Output:**

* Train error & Test error
* Accuracy, Precision, Recall, F1-score
* Confusion Matrix
* Overfitting/underfitting conclusion

**Tasks:**

* Train/test split
* Feature scaling
* Logistic Regression + Decision Tree
* Compare performance
* Identify ML issues

---

## 🔹 ISA 2

❌ No PBL

---

## 🔹 ISA 3

### Term Deposit Subscription Prediction (Logistic Regression)

**Dataset:**
https://archive.ics.uci.edu/dataset/222/bank+marketing

**Objective:**
Predict whether a client subscribes to a term deposit.

**Expected Output:**

* Confusion Matrix
* Precision, Recall, F1
* ROC-AUC
* Sensitivity & Specificity
* Threshold analysis (0.5 + optimized)


**Tasks:**

* Encode categorical data
* Train Logistic Regression
* Explain ROC & threshold trade-off

---

## 🔹 ISA 4

### Bike Demand Forecasting (Ensembles)

**Dataset:**
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

**Objective:**
Compare Bagging, Subagging, and Boosting methods.

**Models:**

* Random Forest
* BaggingRegressor
* Gradient Boosting / XGBoost

**Output:**

* RMSE, MAE (CV mean ± std)
* Feature importance
* Model comparison

---

## 🔹 ISA 5

❌ No ISA

---

## 🔹 ISA 6

### SMS Spam Classification (Ensemble Learning)

**Dataset:**
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

**Models:**

* Naive Bayes, Logistic Regression, SVM
* Voting (Hard + Soft)
* Stacking
* AdaBoost (stumps)

**Tasks:**

* TF-IDF preprocessing
* Stratified K-Fold
* Compare models
* Recommendation

---

## 🔹 ISA 7

### Linear SVM for Customer Churn

**Dataset:**
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

**Objective:**
Analyze margin, support vectors, and effect of C.

**Output:**

* Metrics for multiple C values
* Number of support vectors
* Comparison table

---

## 🔹 ISA 8

### MST-Based Clustering vs K-Means

**Dataset:**
https://archive.ics.uci.edu/dataset/53/iris

**Objective:**
Perform graph-based clustering using MST.

**Tasks:**

* Build graph
* Construct MST
* Remove edges → clusters

**Comparison:**

* Silhouette score
* MST vs K-Means analysis

---

## 🔹 ISA 9, 10, 11

❌ No PBL

---

# 🧪 EXP (Lab / Practical Assignments)

---

## 🔹 EXP 1

### Titanic ML Project

https://www.kaggle.com/competitions/titanic

**Focus:**

* ML workflow
* Overfitting vs underfitting
* Feature engineering

---

## 🔹 EXP 2

### House Prices Regression

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

**Focus:**

* Regression
* Feature engineering
* Tree models

---

## 🔹 EXP 3

### WhatsApp Analyzer

---

## 🔹 EXP 4

### Bank Binary Classification

**Focus:**

* Tabular ML
* ROC-AUC
* Validation strategies

---

## 🔹 EXP 5

### Heart Disease Prediction (Kaggle Playground)

**Focus:**

* Stratified K-Fold
* Boosting models
* Log-loss

---

## 🔹 EXP 6

### Olympic Dataset EDA

---

## 🔹 EXP 7

### Agentic Blog Writing Bot

---

## 🔹 EXP 8

### Digit Recognizer (MNIST)

https://www.kaggle.com/competitions/digit-recognizer

**Focus:**

* Multiclass classification
* SVM
* Dimensionality reduction

---

## 🔹 EXP 9

### Customer Churn

---

## 🔹 EXP 10

### LDA for Class Separability

**Dataset:**
https://archive.ics.uci.edu/dataset/109/wine

**Objective:**

* Dimensionality reduction using LDA
* Compare classification performance

---

## 🔹 EXP 11

### SVD for Topic Modeling

**Dataset:**
https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset

**Objective:**

* TF-IDF + SVD
* Topic extraction
* Dimensionality reduction


