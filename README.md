# Bank Deposit Prediction: Machine Learning & Web Deployment

## What is this project?
A complete end-to-end Machine Learning project designed to predict whether a client will subscribe to a fixed-term bank deposit. The development covers everything from Exploratory Data Analysis (EDA) and hyperparameter optimization for multiple classification algorithms, to the serialization of the winning model and its deployment in an interactive web application.

## Key Features
* **Data Processing and Feature Engineering:** Data cleaning, handling of high-cardinality variables, and creation of new business attributes (such as the binary variable `contacted`).
* **Scikit-Learn Pipelines:** Design of robust preprocessing pipelines integrating missing value imputation, One-Hot encoding for categorical variables, and numerical variable scaling (RobustScaler) to prevent data leakage.
* **Model Evaluation and Comparison:** Training and StratifiedKFold cross-validation of algorithms such as K-Nearest Neighbors (KNN), Decision Trees, Logistic Regression (with and without L1 regularization), and Support Vector Machines (SVM).
* **Hyperparameter Optimization (HPO):** Exhaustive search using `GridSearchCV` to find the optimal generalization point, mitigating overfitting and underfitting issues. It includes advanced experimentation with ensemble methods (Gradient Boosting) evaluated using `neg_log_loss`.
* **Interactive Deployment:** Web application developed with Streamlit that loads the exported model (`.joblib`) and generates real-time predictions based on user-input parameters.

## Tech Stack
* **Language:** Python 3
* **Machine Learning:** Scikit-Learn (Pipelines, SVM, Logistic Regression, Decision Trees, KNN, Gradient Boosting).
* **Data Analysis:** Pandas, NumPy.
* **Visualization:** Matplotlib.
* **Deployment and Web:** Streamlit, Joblib.

## Installation and Usage
1. Clone this repository on your local machine.
2. Install the necessary dependencies by running:
