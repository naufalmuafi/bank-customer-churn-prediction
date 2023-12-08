# **Project Documentation Report**

In an era dominated by digital transformations and rapidly evolving customer expectations, the banking industry is faced with the critical challenge of retaining its customer base. Recognizing the significance of proactive strategies, this project endeavors to address the imminent concern of customer churn through the lens of predictive analytics.

## Project Overview

### Project Title
Bank Customer Churn Prediction

### Project Executive Summary

In this project, we aim to predict bank customer churn using predictive analytics. The goal is to determine whether a customer is more likely to stay or exit from the bank. We will compare various machine learning algorithms to identify the best model for this prediction.

### Project Author
Naufal Mu'afi
[naufalmuafi@mail.ugm.ac.id](mailto:naufalmuafi@mail.ugm.ac.id)

---

## Project Background or Domain


![Customer-Churn-Illustration-960x343](https://github.com/naufalmuafi/bank-customer-churn-prediction/assets/72964378/a2216848-5581-4617-a506-f4c8d44e17f6)


In the banking industry, customer churn is a critical concern as it directly impacts a bank's revenue and profitability. Understanding and predicting customer churn can help banks take proactive measures to retain customers. Customer churn, defined as the likelihood of customers discontinuing their association with a company within a specific timeframe, presents a significant challenge faced by many global enterprises [^1]. Termed as customer agitation in business, it occurs when customers express dissatisfaction with the provided service or product, leading to attrition or the cessation of engagement with the business. In the contemporary business landscape, an increasing number of enterprises prioritize customer retention due to the adverse effects of customer churn. The repercussions include substantial premium losses, reduced profit margins, and the potential loss of referral business from loyal clientele [^2].

In the current fiercely competitive environment, customer retention stands as a crucial component of banking strategy. Bank management must identify and enhance factors that could limit customer defection [^3]. The importance of carefully considering factors that contribute to increased customer retention rates is evident. Various studies underscore the significance of customer retention in the banking industry [^3]. Pioneering research by F. F. Reichheld and W. E. Sasser Jr. established a robust correlation between customer retention and company profits, revealing that a mere 5 percent increase in customer retention leads to improved profitability, ranging from 20 to 85 percent across diverse businesses. Furthermore, research consistently demonstrates that retaining existing clients costs approximately five times less than acquiring new ones [^2].

Some studies have integrated customer segmentation and machine learning techniques to enhance the accuracy of predictive models. Customer segmentation, involving the categorization of customers based on shared characteristics, allows companies to effectively target and market to each group. This step proves pivotal in increasing the conversion rate for businesses, enabling them to allocate advertising budgets more efficiently. Additionally, customer segmentation aids in a better understanding of consumers, identification of the target customer category, and consideration of factors influencing customer churn [^2].


## Business Understanding

### Problem Statements

Based on the background, we can identify the problem that can be solved in this project.

1. Banks want to proactively identify customers who are likely to leave, allowing them to implement retention strategies. So, how can the data be defined for use in creating a good model?
2. Accurate prediction of customer churn can significantly impact customer satisfaction and overall business performance. Then, how can we create a machine learning model to predict the churn of bank customers?

### Goals

Next, we can outline the desired objectives of this project.

1. Develop a machine learning model to predict bank customer churn using well-defined and well-analyzed data.
2. Aim to achieve high accuracy, surpassing 85%, in predicting whether a customer will choose to stay or exit from the bank.

### Solution Statements

To achieve our goals, we will explore the data with strong analysis and multiple machine learning algorithms, including K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Classifier (SVC), and Random Forest. We will fine-tune hyperparameters and select the best-performing model based on accuracy.

- First, we load the data from an open source dataset, such as Kaggle. Then, it is essential to comprehend the definition of each feature in the dataset.
  
- During the Data Analysis phase, we analyze the data using:
   - Assesing and Cleaning the Data
   - Univariate Analysis
   - Multivariate Analysis
     
- Then, in the Data Preparation phase, we employ various methods, including:
   - Handle Imbalanced Data with Resample
   - Category Feature Encoding
   - Correlation Analysis
   - Train Test Split
   - Feature Scaling
     
- In the development of machine learning models, we will implement several algorithms. Therefore, this project will include the implementation of four algorithms, namely:
   - K-Nearest Neighbourhood Algorithm: K-Nearest Neighbors (KNN) is a simple and effective algorithm used for classification and regression tasks. It classifies a data point based on the majority class of its k-nearest neighbors in the feature space.
   - Logistic Regression Algorithm: Logistic Regression is a statistical model used for binary classification. It estimates the probability that a given instance belongs to a particular category and makes predictions based on a logistic function.
   - Support Vector Classifier (SVC) Algorithm: Support Vector Classifier, or Support Vector Machine (SVM), is a powerful algorithm for both classification and regression tasks. It works by finding the hyperplane that best separates the data into different classes while maximizing the margin.
   - Random Forest Algorithm: Random Forest is an ensemble learning method that builds a multitude of decision trees during training and outputs the mode of the classes (classification) or the mean prediction (regression) of the individual trees. It is known for its robustness and high accuracy.

- In aiming to achieve the optimal version of the model, this project will utilize the Grid Search Cross Validation method to determine the best parameters for the model
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



## Data Preparation

### Handle Imbalanced Data with Resample

The target feature, 'Exited,' exhi- In aiming to achieve the optimal version of the model, this project will utilize the Grid Search Cross Validation method to determine the best parameters for the model.ncoded into binary values (0s and 1s) using one-hot encoding.

### Correlation Analysis

A correlation matrix was generated to understand the relationships between features. Low correlations were observed between both categorical and numerical features and the target variable 'Exited.'

### Train Test Split

The dataset was divided into training and test datasets with an 80-20 split.

### Feature Scaling

Numerical features were scaled to a range of 0 to 1 using StandardScaler.



## Model Development

### K-Nearest Neighbourhood Algorithm

The KNN algorithm was employed, and hyperparameter tuning was performed using GridSearchCV. The model achieved an accuracy of 81.24%.

### Logistic Regression Algorithm

Logistic Regression was implemented, and hyperparameter tuning was conducted using GridSearchCV. The model achieved an accuracy of 72.48%.

### Support Vector Classifier Algorithm

SVC was applied, and hyperparameter tuning was performed using GridSearchCV. The model achieved an impressive accuracy of 97.62%.

### Random Forest Algorithm

Random Forest was developed, and hyperparameter tuning was conducted using GridSearchCV. The model achieved an accuracy of 94.89%.



## Model Evaluation

### Confusion Matrix

Confusion matrices for each model were generated, indicating good performance in predicting true positive and true negative values.

### Model Comparison

A comparison of model accuracies revealed that the Support Vector Classifier (SVC) achieved the highest accuracy on both training and test sets.



## Model Prediction

To test the model, predictions were generated using sample data. The SVC algorithm consistently provided the best results.

### Sample Predictions:

1. Customer with features (CreditScore: 815, Geography: Spain, Gender: Female, Age: 39, ...):
   - Predicted: The Customer is more likely to Stay



## Conclusion

The Support Vector Classifier (SVC) demonstrated superior performance in predicting bank customer churn in comparison to other models. The model achieved an accuracy of 97.62%, making it a reliable choice for implementation.

### Recommendations

1. Implement the SVC model for real-time customer churn prediction in the banking system.
2. Continuously monitor and update the model with new data to ensure its accuracy over time.

### Future Work

1. Explore additional features and data sources to enhance model accuracy.
2. Consider deploying the model in a production environment and integrate it with the bank's systems.


## References

[^1]: [Xie, Y., Li, X., Ngai, E. W. T., & Ying, W. (2009). Customer churn prediction using improved balanced random forests. Expert Systems with Applications, 36(3), 5445-5449.](https://www.sciencedirect.com/science/article/pii/S0957417408004326)
[^2]: [Tran, H., Le, N., & Nguyen, V. H. (2023). CUSTOMER CHURN PREDICTION IN THE BANKING SECTOR USING MACHINE LEARNING-BASED CLASSIFICATION MODELS. Interdisciplinary Journal of Information, Knowledge & Management, 18.](https://www.researchgate.net/publication/368911804_Customer_Churn_Prediction_in_the_Banking_Sector_Using_Machine_Learning-Based_Classification_Models)
[^3]: [Cohen, D. A., Gan, C., Hwa, A., & Chong, E. Y. (2006). Customer satisfaction: a study of bank customer retention in New Zealand.](https://researcharchive.lincoln.ac.nz/items/cecd1d6f-5d98-4522-a730-9b65e0c7adad)
