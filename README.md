# Loan Approval Prediction System

## 📌 Project Overview
This project focuses on building an end-to-end Machine Learning pipeline to predict loan approval outcomes based on applicant profiles. In the banking sector, automating the credit risk assessment process is crucial to minimize financial defaults and optimize decision-making. 

A significant challenge in this dataset was the severe Class Imbalance, where approved loans heavily outnumbered rejections. To tackle this, advanced preprocessing techniques—including SMOTE (Synthetic Minority Over-sampling Technique) for class balancing and **Robust Feature Scaling**—were implemented strictly within the training pipeline to ensure an unbiased, realistic evaluation.

Three predictive models were benchmarked: Logistic Regression, Random Forest, and CatBoost. Despite the complexity of tree-based ensembles, the Logistic Regression model emerged as the most optimal and stable solution, achieving an overall accuracy of 82% and a solid 0.70 F1-Score for detecting rejected applications, effectively preventing overfitting and ensuring reliable performance on real-world test data.

---

## 🚀 Key Features
* Real-World Pipeline: Implements proper Feature Scaling and Train/Test splitting to strictly avoid Data Leakage.
* Imbalanced Data Handling: Demonstrates advanced handling of skewed target classes using SMOTE.
* Comprehensive Benchmarking: Compares traditional statistical models against modern gradient-boosting algorithms.

---

## 📊 Project Workflow & Methodology

### 1. Data Import & Exploratory Data Analysis (EDA)
* Libraries Import: Loading core data science and machine learning libraries (`pandas`, numpy, seaborn, matplotlib, and `sklearn`).
* Data Loading & Initial Exploration: Reading the dataset to check shape, column names, data types, summary statistics (`describe`), and missing values.

### 2. Data Visualization & Insights
* Target Variable Distribution: A count plot illustrating the severe class imbalance between approved and rejected requests.
* Education vs. Loan Status: Comparative analysis tracking approval rates between Graduate and Non-Graduate applicants.
* Credit History Impact: Analyzing how historical credit background dictates final loan approvals.

### 3. Data Preprocessing & Pipeline Engineering
* Missing Data Handling: Imputing missing values across features to prevent modeling errors.
* Categorical Encoding: Utilizing LabelEncoder to convert non-numerical variables into structured numeric formats.
* Class Balancing (SMOTE): Applying SMOTE strictly on the training set (`X_train`, `y_train`) to artificially balance the target classes.
* Feature Scaling: Standardizing features using StandardScaler to stabilize model convergence and prevent feature dominance.

### 4. Model Implementation & Evaluation
Three diverse classification models were trained and comprehensively evaluated against the genuine test set (`X_test`):
* Logistic Regression
* Random Forest Classifier
* CatBoost Classifier

---

## 🏆 Final Model Comparison Benchmark

The models were evaluated using accuracy_score, classification_report, and confusion_matrix on the test subset:

| Model | Accuracy | F1-Score (Rejected) | False Positives (FP) |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 82.00% | 0.70 | 13 |
| CatBoost | 80.49% | 0.67 | 14 |
| Random Forest | 80.00% | 0.68 | 13 |

### 🔍 Key Conclusion:
The simpler Logistic Regression model outperformed the complex tree-based ensembles (Random Forest and CatBoost). This indicates that tree models suffered from overfitting on the training data, while Logistic Regression maintained better generalization capabilities, making it the most robust and stable deployment choice for the bank to minimize risky credit decisions.

---

## 🤝 Acknowledgment & Credits
This project was initially developed as a collaborative group work with my team members.Later, I independently enhanced the project by resolving data leakage issues, fixing the feature scaling pipeline, balancing the classes using SMOTE, and re-evaluating the model benchmarks to achieve these accurate results.
