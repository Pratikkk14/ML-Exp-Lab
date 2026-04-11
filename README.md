# 📘 ML Experiment Repository

[Lab folder](#lab-experiments-index) contains all Machine Learning experiments, Kaggle competitions, and applied ML tasks.

[Theory folder](#theory-experiments-index) contins all Machine experiments given with Theory Assignments. These experiments focus more on exploring theoretical concepts thoughts in classvia experiments.


---



<h1 style="margin-top:50px; margin-bottom:40px;" id="lab-experiments-index">LAB EXPERIMENTS INDEX</h1>



## **1. Titanic Survival Prediction**
[ML_Exp_1.ipynb](Lab/ML_Exp_1.ipynb)  
Performed Titanic Survival Prediction for Kaggle Competition



## **2. House Price Prediction (XGBoost)**
[ML_Exp_2.py](Lab/ML_Exp_2.py)  
Performed House Price Prediction using XGBoost Regression



## **3. WhatsApp Chat Analysis**
[ML_Exp_3_WpAnalysis](Lab/ML_Exp_3_WpAnalysis)  
Performed WhatsApp chat data analysis and visualization



## **4. ML Experiment 4**
[ML_Exp_4.ipynb](Lab/ML_Exp_4.ipynb)



## **5. Heart Disease Risk Prediction**
[ML_Exp_5.ipynb](Lab/ML_Exp_5.ipynb)  

Kaggle Playground Competition (2026)

### Key Work:
- Data preprocessing & EDA
- Feature engineering on healthcare data
- Stratified K-Fold Cross Validation
- Model comparison: XGBoost, LightGBM, CatBoost
- Log-loss evaluation
- Ensembling & blending

---

## **6. Olympic Data Analysis**
[ML_Exp_6.ipynb](Lab/ML_Exp_6.ipynb)  

- Exploratory Data Analysis on Olympic dataset  
- Trends in nations & events over time  
- Country-wise medal analysis  

---

## **7. Linear SVM for Customer Churn**
[ML_Exp_7.ipynb](Lab/ML_Exp_7.ipynb)  

### 🔹 **Linear SVM + Margin Analysis + Support Vectors + Regularization (C) + Class Imbalance Handling**

<details>
<summary><b>📌 My Learnings for Linear SVM Churn Prediction</b></summary>

### 📖 What this notebook covers:
- End-to-end churn prediction using Linear SVM
- Pipeline-based preprocessing (encoding + scaling)
- Stratified train-test split
- Training across multiple C values (0.1, 1, 10)
- Evaluation:
  - Confusion Matrix
  - Precision, Recall, F1
  - ROC-AUC
- Support vector analysis
- Margin vs regularization study
- Class imbalance handling

### 🧠 Key Learnings:
- SVM maximizes margin, not probability
- Support vectors define the boundary
- C controls bias-variance trade-off
- Class imbalance affects performance heavily
- Recall is critical in churn prediction
- Scaling is mandatory for SVM
- Pipelines prevent data leakage

### 🎯 Key Takeaway:
**Understanding model behavior > just accuracy**

</details>

---

## **8. MNIST Classification + SVM + PCA**
[ML_Exp_8.ipynb](Lab/ML_Exp_8.ipynb)  

- Pixel normalization
- Models: Decision Tree, Linear SVM, RBF SVM
- Evaluation using accuracy & confusion matrix
- Support vector analysis
- PCA dimensionality reduction
- Kaggle submission

---

## **9. Customer Churn (CatBoost)**
[ML_Exp_9.ipynb](Lab/ML_Exp_9(CustomerChurn).ipynb)  

### 🔹 **CatBoost + Feature Engineering + Leakage Handling + AUC Optimization**

<details>
<summary><b>📌 My Learnings for Customer Churn Prediction</b></summary>

### 📖 What this notebook covers:
- End-to-end churn prediction pipeline
- Structured EDA
- Feature engineering:
  - Service usage
  - Infrastructure flags
  - Tenure buckets
  - Cost features
  - Interaction features
- Leakage prevention
- CatBoost training
- ROC-AUC evaluation

### 🧠 Key Learnings:
- Feature engineering > model complexity
- ROC-AUC focuses on ranking
- Interaction features boost performance
- Avoiding leakage is critical
- CatBoost handles categorical data well

### 🎯 Key Takeaway:
**Good features + understanding > complex models**

</details>

---

## **10. Graph-Based Clustering (MST)**
[ML_Exp_10.ipynb](Lab/ML_Exp_10(MST_Clustering).ipynb)  

### 🔹 **MST + Kruskal + Union-Find + Graph Clustering**

<details>
<summary><b>📌 My Learnings for Graph-Based Clustering</b></summary>

### 📖 What this notebook covers:
- Graph-based clustering using MST
- Distance graph construction
- Kruskal & Prim algorithms
- Union-Find for cycle detection
- PCA visualization
- Silhouette evaluation

### 🧠 Key Learnings:
- Graph theory in ML
- MST captures non-spherical clusters
- Union-Find improves efficiency
- MST vs K-Means differences

### 🎯 Key Takeaway:
Graph-based clustering reveals structures missed by centroid-based methods.

</details>

---



<h1 style="margin-top:50px; margin-bottom:40px;" id="theory-experiments-index">THEORY EXPERIMENTS INDEX</h1>



## **1. Classification Pipeline + Generalization Error**
[ML_Theory_01.ipynb](Theory/ML_Theory_01.ipynb)  

- Logistic Regression & Decision Tree
- Breast Cancer dataset
- Full ML workflow
- Generalization error analysis

---

## **2. Logistic Regression + Performance Metrics**
[ML_Theory_03.ipynb](Theory/ML_Theory_03.ipynb)  

- Binary classification (Bank Marketing dataset)
- Encoding & preprocessing
- Evaluation:
  - Confusion Matrix
  - Precision, Recall, F1
  - ROC-AUC
- Threshold analysis

---

## **3. Spam Detection (Ensemble Learning)**
[ML_Theory_06.ipynb](Theory/ML_Theory_06.ipynb)  

- TF-IDF vectorization
- Ensemble methods (Voting, Stacking)
- Stratified K-Fold validation
- Performance evaluation

---