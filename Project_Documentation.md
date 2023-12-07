# Project Documentation

## Project Overview

### Project Title
Bank Customer Churn Prediction

### Project Executive Summary

In this project, we aim to predict bank customer churn using predictive analytics. The goal is to determine whether a customer is more likely to stay or exit from the bank. We will compare various machine learning algorithms to identify the best model for this prediction.

### Project Author
Naufal Mu'afi
naufalmuafi@mail.ugm.ac.id

---

## Domain Project

In the banking industry, customer churn is a critical concern as it directly impacts a bank's revenue and profitability. Understanding and predicting customer churn can help banks take proactive measures to retain customers. This project addresses the need for accurate prediction of bank customer churn using machine learning.

---

## Business Understanding

### Problem Statements

1. Banks want to proactively identify customers who are likely to leave, allowing them to implement retention strategies.
2. Accurate prediction of customer churn can significantly impact customer satisfaction and overall business performance.

### Goals

1. Develop a machine learning model to predict bank customer churn.
2. Achieve high accuracy in predicting whether a customer will stay or exit from the bank.

### Solution Statements

To achieve our goals, we will explore multiple machine learning algorithms, including K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Classifier (SVC), and Random Forest. We will fine-tune hyperparameters and select the best-performing model based on accuracy.

**References**
- [Customer Churn Prediction in Banking Industry](https://www.researchgate.net/publication/337454440_Customer_Churn_Prediction_in_Banking_Industry)

---

## Data Understanding

The dataset used for this project is the [Bank Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction/data) obtained from Kaggle. It contains information about bank customers and their likelihood to churn.

### Variables in the Dataset:

1. **CreditScore**: Explains how the bank customer's credit score is rated (ranging from 350 to 850).
2. **Geography**: Customer demographics based on the country.
3. **Gender**: Gender of the bank customer.
4. **Age**: Bank customer age (ranging from 18 to 92 years old).
5. **Tenure**: Duration a customer has been associated with or held an account with the bank.
6. **Balance**: Average amount of money held in a customer's account.
7. **NumOfProducts**: The count of financial products or services that a customer holds with the bank.
8. **HasCrCard**: Binary classification of whether a bank customer has a credit card or not.
9. **IsActiveMember**: Indicates whether the customer is still an active member in the bank.
10. **EstimatedSalary**: An approximation of a customer's individual income in a month.
11. **Exited**: The target variable indicating whether a customer has exited from the bank.

---

## Data Preparation

### Handle Imbalanced Data with Resample

The target feature, 'Exited,' exhibited imbalanced data. We addressed this by oversampling the minority class.

### Category Feature Encoding

Categorical features such as 'Geography' and 'Gender' were encoded into binary values (0s and 1s) using one-hot encoding.

### Correlation Analysis

A correlation matrix was generated to understand the relationships between features. Low correlations were observed between both categorical and numerical features and the target variable 'Exited.'

### Train Test Split

The dataset was divided into training and test datasets with an 80-20 split.

### Feature Scaling

Numerical features were scaled to a range of 0 to 1 using StandardScaler.

---

## Model Development

### K-Nearest Neighbourhood Algorithm

The KNN algorithm was employed, and hyperparameter tuning was performed using GridSearchCV. The model achieved an accuracy of 81.24%.

### Logistic Regression Algorithm

Logistic Regression was implemented, and hyperparameter tuning was conducted using GridSearchCV. The model achieved an accuracy of 72.48%.

### Support Vector Classifier Algorithm

SVC was applied, and hyperparameter tuning was performed using GridSearchCV. The model achieved an impressive accuracy of 97.62%.

### Random Forest Algorithm

Random Forest was developed, and hyperparameter tuning was conducted using GridSearchCV. The model achieved an accuracy of 94.89%.

---

## Model Evaluation

### Confusion Matrix

Confusion matrices for each model were generated, indicating good performance in predicting true positive and true negative values.

### Model Comparison

A comparison of model accuracies revealed that the Support Vector Classifier (SVC) achieved the highest accuracy on both training and test sets.

---

## Model Prediction

To test the model, predictions were generated using sample data. The SVC algorithm consistently provided the best results.

### Sample Predictions:

1. Customer with features (CreditScore: 815, Geography: Spain, Gender: Female, Age: 39, ...):
   - Predicted: The Customer is more likely to Stay

2. Customer with features (CreditScore: 654, Geography: Germany, Gender