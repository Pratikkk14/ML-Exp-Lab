# 📘 ML Experiment Repository

[Lab folder](#lab-experiments-index) contains all Machine Learning experiments, Kaggle competitions, and applied ML tasks.

[Theory folder](#theory-experiments-index) contins all Machine experiments given with Theory Assignments. These experiments focus more on exploring theoretical concepts thoughts in classvia experiments.


---



<h1 style="margin-top:50px; margin-bottom:40px;" id="lab-experiments-index">LAB EXPERIMENTS INDEX</h1>



## **1. Titanic Survival Prediction** -> [ML_Exp_1](Lab/ML_Exp_1.ipynb)  
Performed Titanic Survival Prediction for Kaggle Competition



## **2. House Price Prediction (XGBoost)** -> [ML_Exp_2](Lab/ML_Exp_2.py)  
Performed House Price Prediction using XGBoost Regression



## **3. WhatsApp Chat Analysis** -> [ML_Exp_3_WpAnalysis](Lab/ML_Exp_3_WpAnalysis)  
Performed WhatsApp chat data analysis and visualization



## **4. ML Experiment 4** -> [ML_Exp_4](Lab/ML_Exp_4.ipynb)



## **5. Heart Disease Risk Prediction** -> [ML_Exp_5](Lab/ML_Exp_5.ipynb)  
Kaggle Playground Competition (2026)

### Key Work:
- Data preprocessing & EDA
- Feature engineering on healthcare data
- Stratified K-Fold Cross Validation
- Model comparison: XGBoost, LightGBM, CatBoost
- Log-loss evaluation
- Ensembling & blending

---

## **6. Olympic Data Analysis** -> [ML_Exp_6](Lab/ML_Exp_6.ipynb)  
- Exploratory Data Analysis on Olympic dataset  
- Trends in nations & events over time  
- Country-wise medal analysis  

---

## **7. Agentic AI - Planning, Research & Blog Generation** -> [ML_Exp_7](Lab/ML_Exp_7_AgenticAI)

### 🔹 **LangGraph + Planning Agent + Orchestrator-Worker Pattern + Multi-Modal Content**

<details>
<summary><b>📌 My Learnings for Agentic AI Systems</b></summary>

### 📖 What this project covers:
- Planning-based agent architecture (multi-phase workflow)
- Orchestrator-Worker pattern with parallel execution
- LLM-based Router for conditional path selection
- Research integration via Tavily API
- Evidence pack construction and synthesis
- Pydantic schemas for type-safe state management
- Reducer node for intelligent output stitching
- Automated image generation and placement (Gemini)
- Multi-modal blog post generation
- Streamlit UI integration for production deployment

### 🧠 Key Learnings:
- Multi-phase planning enables complex task decomposition
- Orchestrator-Worker parallelization reduces latency 3-5x
- Dynamic routing adapts execution strategy to query requirements
- Real-time research via Tavily addresses hallucination and knowledge cutoff
- Pydantic schemas ensure type-safe state transitions in LangGraph
- Reducer pattern intelligently synthesizes parallel outputs
- Multi-modal generation requires content-aware image placement
- Streamlit enables rapid production deployment of complex backends

### 🎯 Key Takeaway:
**LangGraph transforms LLM calls into production-grade agentic workflows with planning, research, parallelization, and multi-modal generation—enabling sophisticated problem-solving beyond simple prompting.**

</details>


---

## **8. MNIST Classification + SVM + PCA** -> [ML_Exp_8](Lab/ML_Exp_8.ipynb)  
- Pixel normalization
- Models: Decision Tree, Linear SVM, RBF SVM
- Evaluation using accuracy & confusion matrix
- Support vector analysis
- PCA dimensionality reduction
- Kaggle submission

---

## **9. Customer Churn (CatBoost)** -> [ML_Exp_9](Lab/ML_Exp_9(CustomerChurn).ipynb)  

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

## **10. SVD for Document Topic Discovery (LSA)** -> [ML_Exp_10](Lab/ML_Exp_10.ipynb)

### 🔹 **TF-IDF Vectorization + TruncatedSVD + LSA + Clustering Evaluation**

<details>
<summary><b>📌 My Learnings for SVD & Topic Discovery</b></summary>

### 📖 What this notebook covers:
- 20 Newsgroups dataset (3 categories)
- TF-IDF vectorization (10,000 features)
- TruncatedSVD with multiple component settings:
  - 50, 100, 200 components
- KMeans clustering (k=3)
- Explained variance analysis
- Silhouette score evaluation
- Latent topic interpretation (top 10 terms per topic)

### 🧠 Key Learnings:
- SVD achieves 50-200x compression while preserving semantics
- TruncatedSVD exploits sparsity in text data efficiently
- Component count affects clustering quality trade-off
- LSA successfully extracts human-interpretable topics
- Explained variance helps identify optimal dimensionality
- Silhouette scores validate cluster coherence

### 🎯 Key Takeaway:
**SVD transforms high-dimensional sparse text into interpretable semantic dimensions for efficient document clustering and topic discovery.**

</details>


<h1 style="margin-top:50px; margin-bottom:40px;" id="theory-experiments-index">THEORY EXPERIMENTS INDEX</h1>



## **1. Classification Pipeline + Generalization Error** -> [ML_Theory_01](Theory/ML_Theory_01.ipynb)  

- Logistic Regression & Decision Tree
- Breast Cancer dataset
- Full ML workflow
- Generalization error analysis

---

## **3. Logistic Regression + Performance Metrics** -> [ML_Theory_03](Theory/ML_Theory_03.ipynb)  

- Binary classification (Bank Marketing dataset)
- Encoding & preprocessing
- Evaluation:
  - Confusion Matrix
  - Precision, Recall, F1
  - ROC-AUC
- Threshold analysis

---

## **6. Spam Detection (Ensemble Learning)** -> [ML_Theory_06](Theory/ML_Theory_06.ipynb)

- TF-IDF vectorization
- Ensemble methods (Voting, Stacking)
- Stratified K-Fold validation
- Performance evaluation


---


## **7. Linear SVM for Customer Churn** -> [ML_Theory_07](Theory/ML_Theory_07.ipynb)  

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

## **8. Graph-Based Clustering (MST)** -> [ML_Theory_08](Theory/ML_Theory_08.ipynb)  

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

