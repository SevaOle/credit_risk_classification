## The Goal  
The goal of this project was to build a predictive model that determines whether a loan applicant is
likely to **default** on a loan or **successfully repay it**.

## The Dataset Used
I used the [Loan Default Dataset from Kaggle](https://www.kaggle.com/datasets/yasserh/loan-default-dataset) which contains loan application data along with information about borrowers and loan characteristics. This dataset includes financial, demographic, and loan-related variables such as income, credit score, loan amount, loan-to-value ratio, and property value.

Each record in the dataset represents a loan application. The target variable Status indicates whether the borrower eventually defaulted on the loan (1) or successfully repaid it (0).

## Data Preprocessing and Cleaning
To preprocess data before training, firstly, I dropped duplicate rows and the ID column.

Then, I discovered that about a third of the entries contained missing values, so instead of dropping them, I decided on using the median of each column to fill them. I used median as it was not influenced as much by extremes as the mean was.

I have also encoded categorical features and used the StandardScaler to scale the numerical features.

Lastly, as the point of the project was to predict whether a loan applicant is likely to default or repay a loan, I dropped variables like the Interest_rate_spread, Upfront_charges, rate_of_interest, and more, as they reflected lender's decisions rather than borrower's characteristics.

## Model Training
I used two supervised learning models: a **Logistic Regression** model, and a **Random Forest** model.

The Logistic Regression model was used as a baseline model to predict whether a borrower is high risk.

The Random Forest was used as a more advanced model that builds many decision trees and combines their predictions, allowing it to capture more complex patterns in the data.

The dataset was split into 70% training data and 30% testing data.

## Evaluation
I evaluated the models using accuracy, confusion matrices and classification reports. 

**Logistic regression** achieved an **accuracy of 75.5%**, but it struggled to identify high-risk borrowers as the recall was very low (**2%**), so this model missed most of the borrowers that ended up defaulting on their loans.

The **Random Forest** model performed significantly better, with an **accuracy of 87.7%** and was much better at detecting defaulting borrowers, with a **recall of 52%**.

Feature importance analysis showed that the top-3 most predictive features ended up being the **loan-to-value ratio** (LTV), **property value** and the **debt-to-income ratio** (dtir1).

## Conclusion
Using supervised machine learning techniques, I was able to classify loan applications based on their risk of default.

While Logistic Regression acted as a useful baseline, the best result was achieved with the Random Forest model, being much more effective at identifying high-risk borrowers.




