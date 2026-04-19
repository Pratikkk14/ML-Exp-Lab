# 📘 ML Experiment Repository

[Lab folder](#lab-experiments-index) contains all Machine Learning experiments, Kaggle competitions, and applied ML tasks.

[Theory folder](#theory-experiments-index) contins all Machine experiments given with Theory Assignments. These experiments focus more on exploring theoretical concepts thoughts in classvia experiments.


---



<h1 style="margin-top:50px; margin-bottom:40px;" id="lab-experiments-index">LAB EXPERIMENTS INDEX</h1>



## **1. Titanic: Machine Learning from Disaster** -> [ML_Exp_1](Lab/ML_Exp_1.ipynb)

### 🔹 **Classification Pipeline + Missing Value Handling + Categorical Encoding + Random Forest**

<details>
<summary><b>📌 My Learnings for Classification Workflows</b></summary>

### 📖 What this notebook covers:
- Titanic survival prediction (Kaggle competition)
- Feature selection and data preparation
- Missing value handling (median for Age, mode for Embarked)
- Categorical encoding using LabelEncoder (Sex, Embarked)
- Random Forest classification (100 estimators)
- Train/test workflow (Kaggle train/test split)
- Prediction and submission generation

### 🧠 Key Learnings:
- Feature selection is crucial before model training
- Different missing value strategies for different columns
- LabelEncoder properly transforms categorical features for tree models
- Separating train and test data is essential
- Random Forest provides good baseline without extensive tuning
- Consistent preprocessing on both train and test sets
- Kaggle competitions provide realistic ML end-to-end workflow

### 🎯 Key Takeaway:
**Clean data + thoughtful preprocessing + simple models = strong baseline. Mastering fundamentals beats searching for perfect algorithms.**

</details>



## **2. House Prices: Advanced Regression Techniques** -> [ML_Exp_2](Lab/ML_Exp_2.py)

### 🔹 **Regression + Feature Engineering + XGBoost + K-Fold CV + Target Encoding**

<details>
<summary><b>📌 My Learnings for Regression & XGBoost</b></summary>

### 📖 What this notebook covers:
- House price prediction (Kaggle competition)
- Outlier removal and log transformation (SalePrice normalization)
- Strategic missing value handling (None-filling, groupby median, mode imputation)
- Feature engineering: TotalSF (area aggregation), HouseAge, RemodAge (temporal features)
- One-hot encoding for categorical variables
- 5-fold cross-validation with KFold for robust evaluation
- XGBoost regressor (2500 estimators, tuned hyperparameters)
- Performance tracking: RMSE per fold and mean CV RMSE
- Visualizations: price distributions, feature importance, predicted vs actual
- Inverse log transform for final predictions

### 🧠 Key Learnings:
- Log transformation normalizes skewed target distributions
- Thoughtful missing value strategies improve model robustness
- Feature engineering (aggregations, temporal features) drives performance
- 5-fold CV provides reliable performance estimates
- XGBoost hyperparameters (learning_rate, max_depth, subsample) significantly impact results
- Feature importance reveals which features drive predictions
- Predicted vs actual plots reveal systematic biases
- Inverse transforms restore interpretability to log-scaled predictions

### 🎯 Key Takeaway:
**Smart feature engineering + cross-validation + XGBoost tuning create production-grade regression pipelines. Domain knowledge in feature creation matters more than model selection.**

</details>



## **3. WhatsApp Chat Analysis** -> [ML_Exp_3_WpAnalysis](Lab/ML_Exp_3_WpAnalysis)

### 🔹 **Text Processing + NLP + Streamlit Dashboard + Temporal Analysis + Emoji Detection**

<details>
<summary><b>📌 My Learnings for Text Analytics & Interactive Dashboards</b></summary>

### 📖 What this project covers:
- WhatsApp chat export and raw text preprocessing
- Username and message separation from timestamps
- Time feature extraction (year, month, day, hour for temporal breakdown)
- Streamlit interactive dashboard with user/group filtering
- Core statistics: total messages, word count, media link tracking
- Most active users analysis (bar charts, pie charts with contribution %)
- WordCloud generation for visual word frequency representation
- Stop word removal (English + custom Hindi stop words) and text cleaning
- Emoji extraction and frequency analysis
- Monthly and daily timeline visualizations (line plots)
- Peak activity detection: busiest days and months
- Heatmap for identifying activity peaks by day × hour

### 🧠 Key Learnings:
- WhatsApp data export requires careful text parsing (timestamps, usernames, messages)
- Time feature extraction enables rich temporal analysis and trend discovery
- Streamlit dashboards make complex analyses accessible to non-technical users
- Filtering and grouping by user reveals participation patterns
- WordCloud provides visual insight into conversation focus areas
- Stop word removal significantly improves keyword relevance
- Emoji analysis adds cultural/sentiment dimension to text analysis
- Heatmaps effectively reveal behavioral patterns across dimensions (day × hour)
- Interactive dashboards require responsive filtering and instant re-computation

### 🎯 Key Takeaway:
**Text preprocessing + interactive visualization transforms raw chat data into actionable group insights. Dashboards make analytics accessible and exploratory analysis engaging.**

</details>

---

## **4. Binary Classification with Bank Dataset** -> [ML_Exp_4](Lab/ML_Exp_4.ipynb)

### 🔹 **Tabular Binary Classification + Feature Engineering + Stratified K-Fold + ROC-AUC Optimization**

<details>
<summary><b>📌 My Learnings for Tabular Classification</b></summary>

### 📖 What this notebook covers:
- Binary classification on bank marketing dataset
- Data loading and exploratory analysis
- Categorical encoding via LabelEncoder (applied on combined train+test)
- Train-validation split with stratification
- LightGBM classifier with hyperparameter tuning
- Feature engineering: ratio features (balance/age), intensity metrics, engagement metrics
- Stratified 5-fold cross-validation for robust evaluation
- ROC-AUC as main evaluation metric
- Hyperparameter optimization (learning_rate, max_depth, num_leaves)
- Submission generation with probability predictions

### 🧠 Key Learnings:
- Encoding consistency: encode train+test together to prevent unseen categories
- Feature engineering from domain knowledge significantly improves performance
- Stratified K-Fold ensures target distribution is consistent across folds
- ROC-AUC focuses on ranking quality, not calibration
- Hyperparameter tuning (max_depth, learning_rate) balances bias-variance
- LightGBM is efficient on tabular data with many rows
- Validation strategy prevents leakage and enables realistic performance estimates
- Probability predictions enable leaderboard climbing via threshold adjustment

### 🎯 Key Takeaway:
**Tabular classification succeeds through smart encoding + domain-driven feature engineering + proper validation strategy. ROC-AUC optimization requires understanding ranking vs. calibration.**

</details>

---

## **5. Heart Disease Prediction (CatBoost, XGBoost, LightGBM)** -> [ML_Exp_5](Lab/ML_Exp_5.ipynb)

### 🔹 **Multi-Model Comparison + Stratified K-Fold + Log-Loss Evaluation + Blending/Ensembling**

<details>
<summary><b>📌 My Learnings for Ensemble Methods & Multi-Model Evaluation</b></summary>

### 📖 What this notebook covers:
- Heart disease binary classification (Kaggle Playground Series)
- Data preprocessing and column normalization
- Categorical variable identification and separation
- One-hot encoding via ColumnTransformer within Pipelines
- Stratified 5-fold cross-validation
- Multi-model comparison: Random Forest, XGBoost, LightGBM, CatBoost
- Out-of-fold (OOF) predictions for blending
- Log-loss as evaluation metric (probabilistic calibration focus)
- Ensemble blending: weighted average (40% CatBoost + 30% XGBoost + 30% LightGBM)
- Final model training on full data for submission

### 🧠 Key Learnings:
- Stratified K-Fold ensures class balance across folds
- Pipelines prevent data leakage by bundling preprocessing + modeling
- Out-of-fold predictions enable parameter-free blending
- Log-loss rewards well-calibrated probabilities, not just ranking
- CatBoost handles categorical features natively without encoding
- Ensemble blending combines diverse model strengths and reduces variance
- Weighted blending outperforms unweighted averaging
- Training final model on full data maximizes available information
- Kaggle competitions drive systematic model selection and validation

### 🎯 Key Takeaway:
**Multi-model ensembling with OOF blending + stratified K-Fold + log-loss evaluation creates robust, well-calibrated classifiers. Diversity in ensemble members matters more than individual model strength.**

</details>

---

## **6. Olympic Data Analysis** -> [ML_Exp_6](Lab/ML_Exp_6.ipynb)

### 🔹 **Exploratory Data Analysis + Temporal Trends + Aggregation & Pivoting + Interactive Visualization**

<details>
<summary><b>📌 My Learnings for EDA & Data Storytelling</b></summary>

### 📖 What this notebook covers:
- Olympic athlete_events dataset with country region mapping
- Data preprocessing: filtering Summer Olympics, merging, deduplication, one-hot encoding medals
- Summary statistics: editions, cities, sports, events, athletes, nations
- Temporal analysis: participating nations trend, events growth over time
- Heatmap visualization: sports × year event frequency
- Country-specific performance: medal tally over years, most successful athletes
- Athlete demographics: age distribution (overall vs gold medalists)
- Physical attributes: height vs weight scatter by medal status and gender
- Gender participation trends: male vs female athletes over decades
- Helper functions for medal tally, heatmaps, athlete filtering

### 🧠 Key Learnings:
- Preprocessing must handle deduplication carefully for temporal analysis
- Temporal line plots reveal growth trends and plateau points
- Heatmaps effectively show sport diversification across time periods
- Country-specific filtering enables comparative performance analysis
- Aggregations (groupby, pivot_table) transform granular data into insights
- Gender and demographics analysis reveals historical participation shifts
- Plotly interactive visualizations engage audiences better than static plots
- Helper functions modularize analysis for reusability

### 🎯 Key Takeaway:
**Systematic EDA with temporal analysis + demographic breakdowns transforms raw Olympic data into rich historical insights. Modularized helper functions enable flexible, reusable analysis pipelines.**

</details>

---
## **7. Agentic AI Experiment** -> [ML_Exp_7](Lab/ML_Exp_7_AgenticAI)

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

## **8. MNIST Classification: SVM + Decision Trees + PCA** -> [ML_Exp_8](Lab/ML_Exp_8.ipynb)

### 🔹 **Multiclass Classification + SVM Kernels + Dimensionality Reduction + Support Vector Analysis**

<details>
<summary><b>📌 My Learnings for Kernel Methods & Multiclass SVM</b></summary>

### 📖 What this notebook covers:
- MNIST digit recognition (scikit-learn compatible format)
- Pixel normalization: 0-255 → 0-1 range for neural-network-style preprocessing
- Train-validation split (80-20) with stratification
- Multiple classifiers: Decision Tree, Linear SVM, RBF SVM
- Confusion matrix and classification reports for multiclass evaluation
- Support vector analysis: kernel-specific support vector counts and shapes
- Decision function shape inspection for multiclass strategy (one-vs-rest)
- PCA dimensionality reduction: 784 features → 50 components
- SVM + PCA pipeline: maintaining accuracy after aggressive feature reduction
- Kaggle submission generation with ImageId and Label columns
- Comparative accuracy table: Tree (85%) → Lin SVM (93%) → RBF SVM (97%) → RBF+PCA (97.8%)

### 🧠 Key Learnings:
- Pixel normalization improves model training and convergence
- Decision Trees baseline provides quick performance reference
- Linear SVM creates simpler decision boundaries with fewer support vectors
- RBF kernel maps data to higher dimensions via kernel trick, enabling non-linear separation
- Support vector count increases with kernel complexity (RBF > Linear)
- One-vs-rest strategy extends binary SVM to multiclass (10 classes for MNIST)
- PCA dramatically reduces features while preserving predictive power
- RBF + PCA achieves 97.8% accuracy despite 99.4% feature reduction
- Kaggle digit classification is ideal for kernel method experimentation

### 🎯 Key Takeaway:
**Kernel methods (especially RBF SVM) excel at multiclass image classification through non-linear feature transformation. Dimensionality reduction via PCA proves that pixel redundancy is high—most information concentrates in top principal components.**

</details>

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



## **1. Predict Breast Cancer Diagnosis (Classification Pipeline + Generalization Error)** -> [ML_Theory_01](Theory/ML_Theory_01.ipynb)

### 🔹 **Logistic Regression + Decision Tree + Generalization Gap + Error Analysis**

<details>
<summary><b>📌 My Learnings for Classification Workflow & Generalization</b></summary>

### 📖 What this notebook covers:
- Breast Cancer Wisconsin Diagnostic dataset classification (Malignant vs Benign)
- End-to-end supervised ML workflow:
  - problem framing
  - preprocessing
  - model training
  - evaluation and error analysis
- Train/test splitting with fixed random seed and controlled test size
- Baseline model: Logistic Regression
- Non-linear model: Decision Tree classifier
- Performance reporting:
  - Train error and Test error
  - Generalization gap (train vs test performance)
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix
- Practical ML risk analysis:
  - scaling requirements
  - data leakage prevention
  - class imbalance awareness
  - feature correlation effects

### 🧠 Key Learnings:
- Generalization error is the most important indicator beyond training accuracy
- Logistic Regression provides a strong, interpretable baseline
- Decision Trees can overfit if depth and complexity are not controlled
- Comparing train vs test metrics quickly reveals overfitting/underfitting
- Confusion matrix adds class-wise diagnostic depth beyond scalar metrics
- Proper preprocessing and leakage control are essential for reliable conclusions

### 🎯 Key Takeaway:
**A complete classification pipeline is not just about model fitting. Reliable generalization requires careful preprocessing, error-gap analysis, and metric-driven diagnosis of model behavior.**

</details>

---

## **3. Predict Term Deposit Subscription using Logistic Regression** -> [ML_Theory_03](Theory/ML_Theory_03.ipynb)

### 🔹 **Binary Classification + ROC-AUC + Threshold Tuning + Professional Metrics**

<details>
<summary><b>📌 My Learnings for Logistic Regression Evaluation</b></summary>

### 📖 What this notebook covers:
- UCI Bank Marketing binary classification (`y`: yes/no term deposit subscription)
- Categorical feature encoding and preprocessing pipeline
- Train/test split with reproducible random seed
- Logistic Regression model training and probability prediction
- Professional metric reporting:
  - Confusion Matrix
  - Precision, Recall, F1-score
  - Sensitivity (Recall) and Specificity
  - ROC-AUC
- Threshold analysis at:
  - default threshold (0.5)
  - optimized threshold selected by metric trade-off
- Prediction export file format:
  - `probabilities.csv` with record id, probability, predicted label

### 🧠 Key Learnings:
- Accuracy alone is insufficient for imbalanced or cost-sensitive decisions
- ROC-AUC evaluates ranking quality independent of a single threshold
- Threshold tuning changes precision-recall balance based on business objective
- Specificity complements recall when false positives have operational cost
- Logistic Regression remains a strong baseline for calibrated binary probabilities

### 🎯 Key Takeaway:
**The value of Logistic Regression comes from robust probability outputs and threshold-aware decisioning, not just default 0.5 classification.**

</details>

---

## **4. Bike Demand Forecasting: Subagging vs Bagging vs Boosting** -> [ML_Theory_4](Theory/ML_Theory_4.ipynb)

### 🔹 **Ensemble Regression + K-Fold CV + RMSE/MAE Comparison + Bias-Variance Analysis**

<details>
<summary><b>📌 My Learnings for Ensemble Regression</b></summary>

### 📖 What this notebook covers:
- Bike rental demand prediction using UCI Bike Sharing `hour.csv`
- Regression target setup: `cnt` (hourly rental count)
- Feature preparation by dropping leakage-style columns (`instant`, `dteday`, `casual`, `registered`)
- 5-fold cross-validation using `KFold(n_splits=5, shuffle=True, random_state=42)`
- Evaluation metrics with variability tracking:
  - RMSE mean and standard deviation
  - MAE mean and standard deviation
- Model comparison across ensemble paradigms:
  - Bagging: `RandomForestRegressor`
  - Subagging: `BaggingRegressor` with `max_samples=0.6`
  - Boosting: `GradientBoostingRegressor`
- Hyperparameter impact discussion:
  - RF: `n_estimators`, `max_depth`
  - Subagging: `max_samples`, `n_estimators`
  - Boosting: `learning_rate`, `n_estimators`
- Output file generation:
  - `cv_regression_results.csv`
  - `final_predictions.csv` (`ActualCnt`, `PredictedCnt`)
- Feature importance extraction and top-8 feature listing from best tree-based model
- Report section interpreting best generalizing model using bias-variance intuition

### 🧠 Key Learnings:
- K-Fold CV gives more reliable generalization estimates than a single split
- RMSE and MAE together provide balanced error interpretation
- Random Forest reduces variance via averaging many trees
- Subagging (`max_samples < 1.0`) increases diversity but can introduce more bias
- Gradient Boosting improves performance by sequentially correcting residual errors
- `learning_rate` and `n_estimators` jointly control boosting strength vs overfitting risk
- Feature importance helps explain model behavior and demand drivers
- Bias-variance framing improves model selection clarity beyond raw metric values

### 🎯 Key Takeaway:
**For this bike-demand regression task, boosting generalizes best because it balances bias and variance through staged error correction, while bagging methods primarily target variance reduction.**

</details>

---

## **6. SMS Spam Classification using Ensemble Combination** -> [ML_Theory_06](Theory/ML_Theory_06.ipynb)

### 🔹 **TF-IDF + Voting (Hard/Soft) + Stacking + AdaBoost Stumps + Stratified K-Fold**

<details>
<summary><b>📌 My Learnings for Classifier Combination Strategies</b></summary>

### 📖 What this notebook covers:
- SMS Spam Collection text classification (Spam vs Ham)
- Text preprocessing and TF-IDF feature extraction
- Stratified 5-fold cross-validation for robust model comparison
- Base learner benchmarking (for example: Naive Bayes, Logistic Regression, Linear SVM)
- Ensemble combination methods:
  - Hard Voting classifier
  - Soft Voting classifier
  - Stacking classifier (meta-learner: Logistic Regression)
- Boosting baseline with decision stumps:
  - AdaBoost + DecisionTree depth=1
- Metric reporting:
  - Confusion Matrix
  - Precision, Recall, F1-score
  - ROC-AUC
- Output artifacts:
  - `ensemble_comparison.csv`
  - `final_model_predictions.csv` (message id, actual, predicted, probability)

### 🧠 Key Learnings:
- TF-IDF transforms sparse text into effective numerical signals
- Stratified K-Fold is critical when class distributions are imbalanced
- Soft voting can outperform hard voting by leveraging probability confidence
- Stacking learns complementary strengths across diverse base models
- AdaBoost stumps provide a strong bias-controlled boosting baseline
- Best ensemble choice depends on both metric performance and interpretability

### 🎯 Key Takeaway:
**Combining classifiers usually outperforms single models in spam detection, especially when strong text features (TF-IDF) are paired with probability-aware ensembling and stratified validation.**

</details>


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

