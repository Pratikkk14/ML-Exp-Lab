# ML-Exp-Lab
`Lab folder` has all thevML Experiment lab folder for Kaggle Competitons and ML task performed


Index: Lab

1 [ML_Exp_1.ipynb](Lab/ML_Exp_1.ipynb) -> Performed Titanic Survival Prediction For Kaggle Competition

2 [ML_Exp_2.py](Lab/ML_Exp_2.py) -> Performed House Price Prediction with XGBoost regression for Kaggle Competition

3 [ML_Exp_3_WpAnalysis](Lab/ML_Exp_3_WpAnalysis) -> Performed Whatsapp chat data analysis and visualization using Python

4 [ML_Exp_4.ipynb](Lab/ML_Exp_4.ipynb)   
  
5 [ML_Exp_5.ipynb](Lab/ML_Exp_5.ipynb) -> Kaggle Playground Competiton 2026 for predicting heart disease risk. Performed following on the dataset: 
Data preprocessing, Exploratory Data Analysis (EDA), feature engineering on tabular healthcare data, Stratified K-Fold Cross Validation, comparison of boosting models (XGBoost, LightGBoost, CatBoost), log-loss based evaluation, and ensembling/blending techniques for improved generalization

6 [ML_Exp_6.ipynb](Lab/ML_Exp_6.ipynb) -> Performed Exploratory Data Analysis (EDA) on Olympic dataset, including data preprocessing, visualization of trends in participating nations and events over time, country-wise medal

8 [ML_Exp_8.ipynb](Lab/ML_Exp_8.ipynb) -> The MNIST digit dataset was **preprocessed** (pixel normalization), split into training and validation sets, and **multiple models** (Decision Tree, Linear SVM, and RBF SVM) were trained and evaluated using accuracy, confusion matrix, and classification report. Further analysis included **examining support vectors** and **multiclass strategy in SVM**, and performing **PCA-based dimensionality reduction** to test its effect on model performance before generating predictions for Kaggle submission.  

9 [ML_Exp_9.ipynb](Lab/ML_Exp_9(CustomerChurn).ipynb) -> ``CatBoost + Structured Feature Engineering + Leakage Handling + AUC Optimization`` 
-

<details>
<summary><b>My Learnings for Customer Churn Prediction</b></summary>

### What this notebook covers:
- End-to-end customer churn prediction pipeline on telecom dataset
- Data understanding and structured EDA to identify churn patterns
- **Feature Engineering** focused on behavioral signals:
    - Service usage features (num_services_used)
    - Infrastructure flags (has_internet, has_phone, has_multiple_lines)
    - Tenure-based segmentation (tenure_buckets)
    - Cost-based features (avg_charge)
    - Interaction-based features (risk_profile from contract + payment + billing)
    - Hierarchical categorical handling (e.g., "No internet service")
- Target leakage prevention during feature creation
- CatBoost model training with native categorical support
- ROC-AUC evaluation and feature importance analysis
- End-to-end Kaggle submission pipeline

### Key Learnings:
- Systematic approach to real-world tabular ML problems
- Feature engineering > model complexity
- Identifying and handling hidden data structure
- ROC-AUC focuses on ranking, not exact probabilities
- Interaction features significantly impact predictions
- Recognizing and preventing target leakage
- CatBoost advantages for categorical-heavy datasets
- Train-test consistency in pipelines
- Stratified validation for imbalanced data

### Key Takeaway:
**Strong feature engineering + correct problem understanding > complex models**

A well-structured tabular pipeline with CatBoost achieves competitive performance without heavy ensembling.

</details>


---

`Theory Folder`  has all the tasks assigned in Theory Sessions

Index: Theory

1 [ML_Theory_01.ipynb](Theory/ML_Theory_01.ipynb) -> `Classification Pipeline + Generalization Error` 
-
Implemented Logistic Regression (baseline) and Decision Tree models on the UCI Breast Cancer dataset, with full ML workflow including data prep, training, evaluation, and generalization error analysis. 

2 [ML_Theory_03.ipynb](Theory/ML_Theory_03.ipynb) -> `Logistic Regression + Performance Metrics` 
-
Built a binary classifier using Logistic Regression on the UCI Bank Marketing dataset to predict term deposit subscription. Tasks included categorical encoding, train/test split, model training, and evaluation using confusion matrix, precision, recall, F1-score, ROC-AUC, and threshold analysis. Explained ROC curve utility and effects of threshold changes on precision–recall trade-off.

3 [ML_Theory_06.ipynb](Theory/ML_Theory_06.ipynb) -> `Spam Detection using Ensemble Learning` 
-
Developed a text classification pipeline using TF-IDF vectorization and supervised learning models. Implemented and compared ensemble methods including Voting and Stacking.
Used Stratified K-Fold cross-validation for robust evaluation.
Analyzed performance using Precision, Recall, F1-score, and ROC-AUC.
