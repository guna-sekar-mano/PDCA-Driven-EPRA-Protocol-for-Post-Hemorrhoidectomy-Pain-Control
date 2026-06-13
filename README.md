# PDCA-Driven-EPRA-Protocol-for-Post-Hemorrhoidectomy-Pain-Control

📌 **Overview**

This project presents an intelligent clinical decision-support framework for optimizing perioperative pain management in hemorrhoidectomy patients using the EPRA (Enhanced Perioperative Recovery Approach) integrated with the PDCA (Plan–Do–Check–Act) cycle.

The system employs a Particle Swarm Optimization (PSO)-tuned XGBoost classifier to predict postoperative pain severity levels (Mild, Moderate, Severe) from patient clinical data. Based on the predicted pain level, a PDCA-driven clinical recommendation engine generates personalized analgesic, rehabilitation, and risk-management strategies to support enhanced recovery.

## **Machine Learning Pipeline**

**1. Data Preprocessing**

Load patient data from Excel files.
Handle missing values.
Encode categorical variables using Label Encoding.
Normalize features using Min-Max Scaling.
Detect and remove outliers using Z-score analysis.

**2. Pain Classification**

Pain levels are categorized from VAS scores:

Pain Score	Category
0 – 3	Mild
4 – 6	Moderate
7 – 10	Severe

The maximum pain score across multiple postoperative assessments is used as the target variable.

**3. Feature Selection**

An initial XGBoost model is trained to:

Calculate feature importance
Rank predictors
Select top clinical variables

**4. Dimensionality Reduction**

Principal Component Analysis (PCA) is applied to:

Visualize patient distributions
Explore class separability

**5. Hyperparameter Optimization**

Particle Swarm Optimization (PSO) optimizes:

Learning Rate
Maximum Tree Depth
Number of Estimators

Objective:

Minimize Prediction Error = (1 - Accuracy)

**6. Final Model**

A PSO-optimized XGBoost classifier is trained and evaluated using:

Accuracy
Precision
Recall
F1-Score
ROC-AUC

🛠 **Technologies Used**

Python
Pandas
NumPy
Scikit-Learn
XGBoost
Particle Swarm Optimization (PSO)
Matplotlib
Seaborn
SciPy

**How to Run**

Install Dependencies

`pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy openpyxl`

`python main_pipeline.py`
