# **Project Documentation Report**

In an era dominated by digital transformations and rapidly evolving customer expectations, the banking industry is faced with the critical challenge of retaining its customer base. Recognizing the significance of proactive strategies, this project endeavors to address the imminent concern of customer churn through the lens of predictive analytics.


# Table of Contents

1. [**Project Overview**](#project-overview)<br><t>
    1.1. [Project Title](#project-title)<br><t>
    1.2. [Project Executive Summary](#project-executive-summary)<br><t>
    1.3. [Project Author](#project-author)<br><t>

2. [**Project Background or Domain**](#project-background-or-domain)<br><t>

3. [**Business Understanding**](#business-understanding)<br><t>
    3.1. [Problem Statements](#problem-statements)<br><t>
    3.2. [Goals](#goals)<br><t>
    3.3. [Solution Statements](#solution-statements)<br><t>

4. [**Data Understanding**](#data-understanding)<br><t>
    4.1. [Dataset Information](#dataset-information)<br><t>
        4.1.1. [Variables in the Dataset](#variables-in-the-dataset)<br><t>

5. [**Data Preparation**](#data-preparation)<br><t>
    5.1. [Handle Imbalanced Data with Resample](#handle-imbalanced-data-with-resample)<br><t>
    5.2. [Category Feature Encoding](#category-feature-encoding)<br><t>
    5.3. [Correlation Analysis](#correlation-analysis)<br><t>
    5.4. [Train Test Split](#train-test-split)<br><t>
    5.5. [Feature Scaling](#feature-scaling)<br><t>

6. [**Model Development**](#model-development)<br><t>
    6.1. [K-Nearest Neighbourhood Algorithm](#k-nearest-neighbourhood-algorithm)<br><t>
    6.2. [Logistic Regression Algorithm](#logistic-regression-algorithm)<br><t>
    6.3. [Support Vector Classifier Algorithm](#support-vector-classifier-algorithm)<br><t>
    6.4. [Random Forest Algorithm](#random-forest-algorithm)<br><t>

7. [**Model Evaluation**](#model-evaluation)<br><t>
    7.1. [Confusion Matrix](#confusion-matrix)<br><t>
    7.2. [Model Comparison](#model-comparison)<br><t>

8. [**Model Prediction**](#model-prediction)<br><t>
    8.1. [Sample Predictions](#sample-predictions)<br><t>

9. [**Conclusion**](#conclusion)<br><t>
    9.1. [Recommendations](#recommendations)<br><t>
    9.2. [Future Work](#future-work)<br><t>

10. [**References**](#references)<br><t>


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


![Customer-Churn-Illustration-960x343[^pict1]](https://github.com/naufalmuafi/bank-customer-churn-prediction/assets/72964378/a2216848-5581-4617-a506-f4c8d44e17f6)

[^pict1]: [Illustration by CLEARTOUCH](https://www.cleartouch.in/what-is-customer-churn-and-how-do-you-prevent-it/)


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
   - Assessing and Cleaning the Data:
     Identify and handle missing values, outliers, and errors to ensure data quality.
   - Univariate Analysis:
     Analyze individual features to gain insights into their distributions and characteristics.
   - Multivariate Analysis:
     Explore relationships and interactions between multiple features to uncover patterns.

- Then, in the Data Preparation phase, we employ various methods, including:
   - Handle Imbalanced Data with Resample:
     Address imbalanced datasets by resampling techniques such as oversampling or undersampling.
   - Category Feature Encoding:
     Transform categorical features into a format suitable for machine learning algorithms.
   - Correlation Analysis:
     Examine the correlation between features to identify and handle multicollinearity.
   - Train Test Split:
     Split the dataset into training and testing sets for model evaluation.
   - Feature Scaling:
     Standardize or normalize numerical features to ensure they are on a similar scale, benefiting certain machine learning algorithms.
     
- In the development of machine learning models, we will implement several algorithms. Therefore, this project will include the implementation of four algorithms, namely:
   - K-Nearest Neighbourhood Algorithm: K-Nearest Neighbors (KNN) is a simple and effective algorithm used for classification and regression tasks. It classifies a data point based on the majority class of its k-nearest neighbors in the feature space.
   - Logistic Regression Algorithm: Logistic Regression is a statistical model used for binary classification. It estimates the probability that a given instance belongs to a particular category and makes predictions based on a logistic function.
   - Support Vector Classifier (SVC) Algorithm: Support Vector Classifier, or Support Vector Machine (SVM), is a powerful algorithm for both classification and regression tasks. It works by finding the hyperplane that best separates the data into different classes while maximizing the margin.
   - Random Forest Algorithm: Random Forest is an ensemble learning method that builds a multitude of decision trees during training and outputs the mode of the classes (classification) or the mean prediction (regression) of the individual trees. It is known for its robustness and high accuracy.

- In aiming to achieve the optimal version of the model, this project will utilize the **Grid Search Cross Validation** method to determine the best parameters for the model.

  Grid Search Cross Validation is a hyperparameter tuning technique that systematically evaluates a predefined set of hyperparameter combinations for a machine learning model. It works by creating a grid of all possible hyperparameter values and performing cross-validation for each combination to identify the set that yields the best performance.

  **Algorithm:**
  1. Define a grid of hyperparameter values for the model.
  2. For each combination of hyperparameters in the grid:
     - a. Split the dataset into training and validation sets.
     - b. Train the model on the training set.
     - c. Evaluate the model on the validation set using a chosen performance metric.
     - d. Repeat the process for different folds in cross-validation if applicable.
  3. Select the hyperparameter combination that resulted in the best performance.

  **Mathematical Expression:**
  Let $H$ be the set of hyperparameter combinations, $S$ be the performance metric, and $K$ be the number of folds in cross-validation. The optimal hyperparameter combination $\theta^*$ is obtained by:
  
  $\theta^* = \arg\min_{\theta \in H} \left( \frac{1}{K} \sum_{k=1}^K S(\text{Model}_{\theta}, \text{Validation}_k) \right)$
  
  where $\text{Model}_{\theta}$ represents the model trained with hyperparameter combination $\theta$ and $\text{Validation}_k$ denotes the validation set for the $k$-th fold.
  
  Grid Search Cross Validation aids in finding hyperparameter values that optimize the model's performance and generalization to unseen data.  


## Data Understanding

The dataset used for this project is the [Bank Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction/data) obtained from Kaggle. It contains information about bank customers and their likelihood to churn.


![dataset_info](https://github.com/naufalmuafi/bank-customer-churn-prediction/assets/72964378/e706020e-5584-4cc0-bfd4-092d2f17b9f1)


**Dataset Information:**

Type | Information
--- | ---
Source | [Kaggle Dataset : Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction/data)
License | Other
Category | `Economy`, `Business`, `Finance`, `Banking`
Usability Rating | 9.71
File Type and Size | CSV (268 kb)

### Variables in the Dataset:

`Churn_Modelling.csv` dataset file has 10,000 rows and 14 columns. This means the dataset contains information for 10.000 bank customers, including 14 various details such as Gender, Age, Geography, and many more. The dataset comprises 9 columns of `int64` data type, 2 columns of `float64` data type, and 3 columns of `object` data type. It's not possible to determine whether the data represents numerical or categorical features based solely on the data type. Initially, we identify 3 dummy features in the dataset:  `RowNumber`, `CustomerId`, and `Surname`. Upon further examination, we classify the dataset as having 4 numerical features, and 7 categorical features.


Explanation of 14 columns or feature from the `Churn_Modelling.csv` dataset:

1. `RowNumber` - Indicates the row number.

2. `CustomerId` - Each customer has a unique ID stored in this feature.

3. `Surname` - The Bank customer's surname.

4. `CreditScore` - Explains how the Bank Customer's Credit Score is rated. In this dataset, the value ranges from 350 to 850.

5. `Geography` - Customer demographics based on the country.

6. `Gender` - Gender of the Bank Customer.

7. `Age` - Bank Customer Age, ranging from 18 to 92 years old.

8. `Tenure` - Duration a customer has been associated with or held an account with the bank. In this dataset, the distribution of this feature ranges from 0 to 10 years.

9. `Balance` - Refers to the average amount of money held in a customer's account. It's a continuous feature with minimum balance is $0, and the maximum balance is $250898.09.

10. `NumOfProducts` - The variety or count of financial prodcuts or services that a customer holds with the Bank. In this dataset, the bank has a total of 4 products.

11. `HasCrCard` - Binary Classification of whether a Bank Customer has a credit card or not.

12. `IsActiveMember` - a feature for customer classification; it indicates whether the customer is still active member in the bank or has become a passive member.

13. `EstimatedSalary` - an Approximation or prediction of a customer's individual income in a month. In this dataset, incomes range from $11.58 to $199992.48.

14. `Exited` - The target or output that provides various details to decide either a customer is more likely to stay in the bank or has exited from the bank.



## Data Preparation

As mentioned in the solution statement, we employ certain techniques, including:

### Handle Imbalanced Data with Resample

In order to give a comprehensive view of the unbalanced data problem, we have started with exploring the target data.

![Target-Distribution](https://github.com/naufalmuafi/bank-customer-churn-prediction/assets/72964378/4becac2a-bd83-4222-8931-15d6f740e8c9)

Figure above shows that the Target Data has an Imbalanced Data, so we can handle it with resample method. Random **over-sampling**, classified as a non-heuristic algorithm, aims to address class imbalance by randomly duplicating instances of the minority target, thereby promoting a more balanced distribution [^over-sampling-1]. Nevertheless, this approach has two drawbacks. Firstly, it increases the risk of overfitting by generating identical reproductions of minority class instances [^over-sampling-1]. Secondly, it intensifies the time-consuming nature of the learning process, particularly when the original dataset is large but imbalanced, mirroring the characteristics of our dataset.

The resample() function from scikit-learn's imbalanced-learn module is a convenient tool for oversampling that can be expressed as:

$Resampled Data = resample(minorityClass, nSamples=desired_count)$

This function randomly replicates instances from the minority class to match the number of instances in the majority class, thereby mitigating class imbalance.

### Category Feature Encoding

In this section of the project, we focus on the encoding of categorical features within the dataset to facilitate their integration into machine learning models. By converting categorical variables into binary representations, we ensure that the data is in a suitable format for various algorithms. The core operation involves the utilization of the pd.get_dummies() method on the 'churn' DataFrame. The categorical columns subjected to one-hot encoding are 'Geography' and 'Gender'. The drop_first=True parameter is employed to exclude the first category level during encoding, mitigating potential multicollinearity issues in subsequent analyses, enhancing the robustness of subsequent analyses.

### Correlation Analysis

After encoding the categorical data, we can go further to the next step of the analysis, a correlation matrix was generated to comprehensively examine the relationships between features within the dataset. The correlation matrix provides insights into how variables, both categorical and numerical, are interrelated. Notably, during this analysis, it was observed that the correlations between the features and the target variable 'Exited' were relatively low. This finding suggests that the dataset exhibits a diverse set of features with varying degrees of influence on the target variable.

Correlation analysis assesses the linear relationship between pairs of variables in a dataset. A correlation matrix provides a comprehensive overview of these relationships. The heatmap() function from the seaborn library is often employed to visualize correlation matrices. Mathematically, correlation (Pearson correlation coefficient, $ρ$) between variables $X$ and $Y$ can be expressed as:

$ρ\left(X, Y\right)=\frac{Cov\left(X, Y\right)​}{σ_X⋅ \ σ_Y}$

The heatmap visualizes these correlation coefficients, with warmer colors indicating stronger positive correlations, cooler colors indicating stronger negative correlations, and neutral colors for weaker correlations.

### Train Test Split

Following the correlation analysis, the dataset underwent a crucial step known as train-test split. This process involved dividing the dataset into two subsets: a training dataset and a test dataset. Dividing the dataset into training and testing sets is crucial for evaluating model performance. The split was executed with an 80-20 ratio, allocating 80% of the data for training machine learning models and reserving the remaining 20% for evaluating model performance. This approach helps ensure that the model is trained on a sufficiently large dataset while still having an independent set of data for validation and testing. More, this division ensures that the model is trained on a sufficiently large dataset while maintaining an independent dataset for unbiased model evaluation.

### Feature Scaling

After the train-test split, the numerical features within the dataset were subjected to feature scaling using the StandardScaler. Feature scaling is a vital preprocessing step that standardizes the range of numerical features, bringing them to a consistent scale. In this case, the scaling was performed to constrain numerical features within a range of 0 to 1. Standardizing the features helps prevent variables with larger scales from dominating the modeling process, ensuring fair consideration of all features during model training and evaluation. The use of StandardScaler is a common practice to achieve this normalization.

Feature scaling ensures that numerical features are on a similar scale, preventing certain features from dominating the model training process. The StandardScaler() from scikit-learn standardizes features by transforming them to have a mean $(μ)$ of $0$ and a standard deviation $(σ)$ of $1$. Mathematically, for a feature $X$:

$X_{scaled}=\frac{X − μ​}{σ}$

This transformation maintains the relative differences between feature values while placing them on a comparable scale, enhancing the stability and convergence of machine learning models during training.


## Model Development

### **K-Nearest Neighbourhood Algorithm**

The K-Nearest Neighbors (KNN) algorithm is a type of instance-based learning, where the function is only approximated locally and all computation is deferred until function evaluation. It is a simple and effective algorithm for classification and regression tasks. The KNN algorithm is based on the principle that similar data points are close to each other in the feature space. Mathematically, the KNN algorithm predicts the classification of a data point based on the majority class of its K nearest neighbors. The distance between data points is typically calculated using methods such as Euclidean distance or Manhattan distance [^knn1]. The prediction for a new data point x is based on a majority vote of its k-nearest neighbors, and can be expressed as:

$\hat{y}^​ = majority vote\left(y_1, y_2, ..., y_k\right)$

where $\hat{y}$ is the predicted class for $x$, and $y_1, y_2, ..., y_k$ are the classes of the k-nearest neighbors [^knn2]. The KNN algorithm was employed, and hyperparameter tuning was performed using GridSearchCV. The model achieved an accuracy of 81.24%.

|              | precision | recall  | f1-score | support     |
|--------------|-----------|---------|----------|-------------|
| Stay         | 0.858744  | 0.748047| 0.799582 | 1536.000000 |
| Exit         | 0.776688  | 0.876873| 0.823745 | 1535.000000 |
| accuracy     | 0.812439  | 0.812439| 0.812439 | 0.812439    |
| macro avg    | 0.817716  | 0.812460| 0.811664 | 3071.000000 |
| weighted avg | 0.817729  | 0.812439| 0.811660 | 3071.000000 |


### **Logistic Regression Algorithm**

Logistic Regression is a statistical model that uses a logistic function to model the probability of a binary outcome. It is widely used for binary classification problems. The logistic function, also known as the sigmoid function, is defined as:

$σ\left(z\right)=\frac{1}{1+e^{-z}}$

where z is a linear combination of the input features and model parameters. The logistic regression algorithm minimizes the logistic loss function to find the best-fitting model parameters. It is a linear model and makes predictions based on the weighted sum of the input features [^lr1]. The probability of the output being in a particular class is given by the logistic function, and the prediction can be expressed as:

$P\left(Y=1∣X\right)=\frac{1}{1+e^{−\left(β_0​+β_1​X_1​+...+β_p​X_p​\right)}}$

where $P(Y=1|X)$ is the probability of the output being in class 1 given input features $X$, $\beta_0, \beta_1, ..., \beta_p$ are the model parameters, and $X_1, X_2, ..., X_p$ are the input features [^lr2]. Logistic Regression was implemented, and hyperparameter tuning was conducted using GridSearchCV. The model achieved an accuracy of 72.48%.

|              | precision | recall  | f1-score | support     |
|--------------|-----------|---------|----------|-------------|
| Stay         | 0.719644  | 0.736979| 0.728208 | 1536.000000 |
| Exit         | 0.730307  | 0.712704| 0.721398 | 1535.000000 |
| accuracy     | 0.724845  | 0.724845| 0.724845 | 0.724845    |
| macro avg    | 0.724976  | 0.724841| 0.724803 | 3071.000000 |
| weighted avg | 0.724974  | 0.724845| 0.724804 | 3071.000000 |


### Support Vector Classifier Algorithm

The Support Vector Classifier (SVC) is a supervised learning algorithm that can be used for both classification and regression tasks. In the context of classification, the SVC algorithm finds the hyperplane that best separates the data into different classes. It is particularly effective in high-dimensional spaces. The algorithm works by finding the maximum-margin hyperplane, which is the hyperplane that maximizes the distance to the nearest data points of any class. Mathematically, the objective of the SVC algorithm is to solve the optimization problem that maximizes the margin and minimizes the classification error [^svc1]. The decision function for the SVC is given by:

$f\left(x\right)=sign\left(\sum ^n_{i=1}​α_i​y_i​K\left(x,x_i\right)+b\right)$

where $f(x)$ is the decision function, $\alpha_i$ are the learned Lagrange multipliers, $y_i$ are the class labels, $K(x, x_i)$ is the kernel function, and b is the bias term [^svc1]. SVC was applied, and hyperparameter tuning was performed using GridSearchCV. The model achieved an impressive accuracy of 97.62%.

|              | precision | recall    | f1-score  | support     |
|--------------|-----------|-----------|-----------|-------------|
| Stay         | 0.954630  | 1.000000  | 0.976789  | 1536.000000 |
| Exit         | 1.000000  | 0.952443  | 0.975642  | 1535.000000 |
| accuracy     | 0.976229  | 0.976229  | 0.976229  | 0.976229    |
| macro avg    | 0.977315  | 0.976221  | 0.976215  | 3071.000000 |
| weighted avg | 0.977308  | 0.976229  | 0.976216  | 3071.000000 |


### Random Forest Algorithm

The Random Forest algorithm is an ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random forests correct for decision trees' habit of overfitting to their training set. The algorithm introduces randomness when growing the trees, which leads to a diverse set of trees. During prediction, the random forest aggregates the predictions of the individual trees to make a final prediction. Mathematically, the algorithm combines the predictions of multiple decision trees to improve generalizability and robustness [^rf1]. The prediction of the random forest for a new data point can be expressed as the mode of the predictions of the individual trees:

$\hat{Y}^​ = model\left(Y_1, Y_2, ..., Y_n\right)$

where $\hat{Y}$ is the predicted class for the new data point, and $Y_1, Y_2, ..., Y_n$ are the predictions of the individual trees [^rf2]. Random Forest was developed, and hyperparameter tuning was conducted using GridSearchCV. The model achieved an accuracy of 94.89%.

|              | precision | recall    | f1-score  | support     |
|--------------|-----------|-----------|-----------|-------------|
| Stay         | 0.977163  | 0.919271  | 0.947333  | 1536.000000 |
| Exit         | 0.923739  | 0.978502  | 0.950332  | 1535.000000 |
| accuracy     | 0.948877  | 0.948877  | 0.948877  | 0.948877    |
| macro avg    | 0.950451  | 0.948886  | 0.948833  | 3071.000000 |
| weighted avg | 0.950460  | 0.948877  | 0.948832  | 3071.000000 |


## Model Evaluation

### Evaluation Metrics

The precision, recall, and F1-score are important metrics for evaluating the performance of classification models. They are particularly useful in assessing the model's ability to correctly identify positive cases and avoid misclassification.

#### Precision

Precision is the ratio of true positive predictions to the total number of positive predictions made by the model. It is calculated using the following formula:

$\text{Precision} = \frac{TP}{TP + FP}$

Where:<br>
$(TP)$ is the number of true positive predictions (correctly predicted positive cases).<br>
$(FP)$ is the number of false positive predictions (negative cases incorrectly classified as positive).

Precision is a measure of the accuracy of the positive predictions. A high precision value indicates that when the model predicts a positive case, it is likely to be correct.

#### Recall

Recall, also known as sensitivity, is the ratio of true positive predictions to the total number of actual positive cases in the dataset. It is calculated using the following formula:

$\text{Recall} = \frac{TP}{TP + FN}$

Where:<br>
( FN ) is the number of false negative predictions (positive cases incorrectly classified as negative).

Recall measures the model's ability to identify all actual positive cases. A high recall value indicates that the model is able to correctly identify a large proportion of the positive cases in the dataset.

#### F1-score

The F1-score is the harmonic mean of precision and recall, and it provides a balance between the two metrics. It is calculated using the following formula:

$F1\text{-}score = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

The F1-score takes both false positives and false negatives into account, making it a useful metric for imbalanced datasets where the number of negative cases is much larger than the number of positive cases. A high F1-score indicates that the model has both good precision and recall. These metrics are commonly used to evaluate the performance of classification models and are particularly useful when the class distribution is imbalanced. They provide valuable insights into how well the model is performing in terms of identifying positive cases and avoiding misclassifications.

### Confusion Matrix

Confusion matrices for each model were generated, indicating good performance in predicting true positive and true negative values.

![confussion-matrix](https://github.com/naufalmuafi/bank-customer-churn-prediction/assets/72964378/b7d1790f-765c-45a9-b437-05b24bd37591)


### Model Comparison

In terms of advantages and disadvantages, the KNN algorithm is simple and easy to implement, but it can be computationally expensive for large datasets and requires careful selection of the value of K. Logistic Regression is a linear model that is easy to interpret and can handle both binary and continuous input features, but it may not perform well when the relationship between the input features and the output is non-linear. The SVC algorithm is effective in high-dimensional spaces and can handle non-linear decision boundaries, but it can be sensitive to the choice of kernel function and hyperparameters. The Random Forest algorithm is robust to overfitting and can handle both categorical and continuous input features, but it can be computationally expensive and difficult to interpret.

|       | train     | test      |
|-------|-----------|-----------|
| KNN   | 88.773101 | 81.243894 |
| LR    | 73.044045 | 72.484533 |
| SVC   | 100.0     | 97.622924 |
| RF    | 100.0     | 94.887659 |

A comparison of model accuracies revealed that the Support Vector Classifier (SVC) achieved the highest accuracy on both training and test sets.

![model-comparison](https://github.com/naufalmuafi/bank-customer-churn-prediction/assets/72964378/1e82fe2c-8559-4acd-85e4-748ec01f22fe)

The confusion matrix is a table that summarizes the performance of a classification model by comparing the predicted and actual class labels. It consists of four values: true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN).


## Model Prediction

To test the model, predictions were generated using sample data. The SVC algorithm consistently provided the best results.

### Sample Predictions:

1. Customer with features (CreditScore: 815, Geography: Spain, Gender: Female, Age: 39, ...):
   - Predicted: The Customer is more likely to Stay


## Conclusion

The Support Vector Classifier (SVC) demonstrated superior performance in predicting bank customer churn in comparison to other models. The model achieved an accuracy of 97.62%, making it a reliable choice for implementation.


## References

[^1]: [Xie, Y., Li, X., Ngai, E. W. T., & Ying, W. (2009). Customer churn prediction using improved balanced random forests. Expert Systems with Applications, 36(3), 5445-5449.](https://www.sciencedirect.com/science/article/pii/S0957417408004326)
[^2]: [Tran, H., Le, N., & Nguyen, V. H. (2023). CUSTOMER CHURN PREDICTION IN THE BANKING SECTOR USING MACHINE LEARNING-BASED CLASSIFICATION MODELS. Interdisciplinary Journal of Information, Knowledge & Management, 18.](https://www.researchgate.net/publication/368911804_Customer_Churn_Prediction_in_the_Banking_Sector_Using_Machine_Learning-Based_Classification_Models)
[^3]: [Cohen, D. A., Gan, C., Hwa, A., & Chong, E. Y. (2006). Customer satisfaction: a study of bank customer retention in New Zealand.](https://researcharchive.lincoln.ac.nz/items/cecd1d6f-5d98-4522-a730-9b65e0c7adad)
[^over-sampling-1]: [Mohammed, R., Rawashdeh, J., & Abdullah, M. (2020, April). Machine learning with oversampling and undersampling techniques: overview study and experimental results. In 2020 11th international conference on information and communication systems (ICICS) (pp. 243-248). IEEE.](https://ieeexplore.ieee.org/abstract/document/9078901?casa_token=zwQWkVHTTbYAAAAA:Sr0rIrgCaloLp83pnimWRu2Tx8C0E_2u1Pw6whfmiv3GQW7_9bbm2ennh4JAjxzwGSmXYkFeVi1_)
[^knn1]: [Ghunimat, D.M., Alzoubi, A.E., Alzboon, A.A., & Hanandeh, S. (2022). Prediction of concrete compressive strength with GGBFS and fly ash using multilayer perceptron algorithm, random forest regression and k-nearest neighbor regression. Asian Journal of Civil Engineering, 24, 169-177.](https://www.semanticscholar.org/paper/Prediction-of-concrete-compressive-strength-with-Ghunimat-Alzoubi/2bb0f80d2914eeeccdbf156c2d0c12fd1a0b2b5b)
[^knn2]: [Adeshina, A.M. (2023). Prediction of Diabetes Mellitus using Machine Learning Algorithms: Comparative Analysis of K-Nearest Neighbor, Random Forest and Logistic Regression. SLU Journal of Science and Technology.](https://www.semanticscholar.org/paper/Prediction-of-Diabetes-Mellitus-using-Machine-of-Adeshina/79fe14595faeaab90b6fe242d86f49e118e9d750)
[^lr1]: [Adeshina, A.M. (2023). Prediction of Diabetes Mellitus using Machine Learning Algorithms: Comparative Analysis of K-Nearest Neighbor, Random Forest and Logistic Regression. SLU Journal of Science and Technology.](https://www.semanticscholar.org/paper/Prediction-of-Diabetes-Mellitus-using-Machine-of-Adeshina/79fe14595faeaab90b6fe242d86f49e118e9d750)
[^lr2]: [Das, S., Bhattacharyya, K., & Sarkar, S. (2023). Performance Analysis of Logistic Regression, Naive Bayes, KNN, Decision Tree, Random Forest and SVM on Hate Speech Detection from Twitter. International Research Journal of Innovations in Engineering and Technology.](https://www.semanticscholar.org/paper/Performance-Analysis-of-Logistic-Regression%2C-Naive-Das-Bhattacharyya/43b6d317c5ecf72d4bf6bd46e182b2b5fc97d43b)
[^svc1]: [Afrianto, M.A., & Wasesa, M. (2020). Booking Prediction Models for Peer-to-peer Accommodation Listings using Logistics Regression, Decision Tree, K-Nearest Neighbor, and Random Forest Classifiers. Journal of Information Systems Engineering and Business Intelligence.](https://www.semanticscholar.org/paper/Booking-Prediction-Models-for-Peer-to-peer-Listings-Afrianto-Wasesa/4dd5ae54caac18ee0efe38dd4a704e9e2e8c4cf2)
[^rf1]: [Afrianto, M.A., & Wasesa, M. (2020). Booking Prediction Models for Peer-to-peer Accommodation Listings using Logistics Regression, Decision Tree, K-Nearest Neighbor, and Random Forest Classifiers. Journal of Information Systems Engineering and Business Intelligence.](https://www.semanticscholar.org/paper/Booking-Prediction-Models-for-Peer-to-peer-Listings-Afrianto-Wasesa/4dd5ae54caac18ee0efe38dd4a704e9e2e8c4cf2)
[^rf2]: [Mohsin, M.A., & Hamad, A.H. (2022). Performance Evaluation of SDN DDoS Attack Detection and Mitigation Based Random Forest and K-Nearest Neighbors Machine Learning Algorithms. Revue d'Intelligence Artificielle.](https://www.semanticscholar.org/paper/Performance-Evaluation-of-SDN-DDoS-Attack-Detection-Mohsin-Hamad/9b5b52f3e6a80328a227898695bd6f02c4ddb39e)
