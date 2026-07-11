#  🏦 Bank_Customer_Churn_Analysis

An end-to-end Machine Learning project that predicts whether a bank customer is likely to leave (churn) based on customer demographic and banking information. The project includes data preprocessing, model comparison, deep learning, hyperparameter tuning, and deployment using Streamlit.

---

## 📌 Project Overview

Customer churn is one of the major challenges faced by banks. This project helps identify customers who are likely to leave the bank, enabling organizations to take proactive retention measures.

The workflow includes:

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Feature Engineering
- Outlier Treatment
- Power Transformation
- Feature Scaling
- Model Comparison
- Artificial Neural Network (ANN)
- Hyperparameter Tuning
- Streamlit Deployment

---

## 📂 Dataset

**Dataset Name:** Bank Customer Churn Dataset

**Features**

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

**Target Variable**

- Exited
  - 1 = Customer Churned
  - 0 = Customer Stayed

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Keras Tuner
- XGBoost
- Joblib
- Streamlit

---

## 📊 Exploratory Data Analysis

Performed:

- Data inspection
- Missing value check
- Duplicate value check
- Distribution analysis
- Correlation analysis
- Categorical analysis
- Skewness analysis
- Boxplots for outlier detection

---

## ⚙ Data Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary columns
    - RowNumber
    - CustomerId
    - Surname

- Power Transformation on Age

- Outlier Treatment using IQR Method

- One-Hot Encoding using `pd.get_dummies()`

- Feature Scaling using StandardScaler

---

## 🤖 Models Compared

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gradient Boosting
- XGBoost Classifier
- Artificial Neural Network (ANN)
- Tuned ANN using Keras Tuner

---

## 📈 Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

After comparison, **Gradient Boosting** achieved the best overall performance and was selected as the final model.

---

## 🧠 Deep Learning

Built an Artificial Neural Network using TensorFlow/Keras with:

- Dense Layers
- ReLU Activation
- Sigmoid Output Layer
- Dropout Regularization

Applied:

- Early Stopping
- Model Checkpoint

Hyperparameter tuning was performed using **Keras Tuner Random Search**.

---

## 💻 Streamlit Application

The project includes an interactive Streamlit application where users can:

- Enter customer details
- Predict customer churn
- View churn probability
- View customer summary

The application automatically performs:

- Power Transformation
- One-Hot Encoding
- Feature Scaling
- Prediction using the trained Gradient Boosting model

---

## 📁 Project Structure

```
Bank-Customer-Churn-Prediction/
│
├── app.py
├── model_training.py
├── Churn Modeling.csv
├── gradient_boosting_model.pkl
├── scaler.pkl
├── power_transformer_age.pkl
├── features_names.pkl
├── requirements.txt
├── README.md
└── images/
```

## 📌 Future Improvements

- Hyperparameter optimization using Optuna
- Cross-validation
- SHAP Explainability
- Feature Importance Dashboard
- Docker Deployment
- Cloud Deployment

---

## 📜 Key Learnings

- End-to-end Machine Learning workflow
- Data preprocessing techniques
- Model comparison
- ANN development using TensorFlow
- Hyperparameter tuning
- Model serialization using Joblib
- Building interactive applications with Streamlit



---

⭐ If you found this project helpful, don't forget to give it a star!
