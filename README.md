# Enron-Fraud-Analysis-with-Machine-Learning
## By Sagarnil Das, in fulfillment of Udacity's Data Analyst Nanodegree, Project 6

Enron, a multibillion dollar company in the heart of the Wall Street was one of the largest companies in the United States in 2000. By 2002, it collapsed into bankruptcy due to widespread corporate fraud. They reached a dizzying height only to face a plummeting collapse. The bankruptcy of Enron affected thousands of its employees and shook the whole Wall Street by its foundation. In the federal investigation that followed, all the emails from Enron employees were made public. Since then these emails have spreaded like a wildfire through the world and many analysis and investigations have been done on this data. In this project, my target is to build a Person of Interest identifier/Label and try to build a Machine Learning Algorithm to predict the possible Persons of Interest based on various features.

I built a "person of interest" (POI) identifier to detect and predict culpable persons utilizing scikit-learn and machine learning methodologies, using features from financial data, email data, and labeled data--POIs who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

# Goal of this Project

The goal of this project was to utilize the financial and email data from Enron to build a predictive, analytic model that could identify whether an individual could be considered a "person of interest" (POI). Since the dataset contained labeled data--culpable persons were already listed as POIs--the value of this model on the existing dataset is limited. Rather, the potential value such a model may provide is in application to other datasets from other companies, to potentially identify suspects worth investigating further. The dataset contained 146 records with 14 financial features, 6 email features, and 1 labeled feature (POI). Of the 146 records, 18 were labeled, a priori, as persons of interest. Through exploratory data analysis and cursory spreadsheet/CSV review, I was able to identify 2 candidate records for removal:

1. TOTAL: This was an extreme outlier for most numerical features, as it was likely a spreadsheet artifact.
2. THE TRAVEL AGENCY IN THE PARK: This record did not represent an individual.

After data cleaning, 144 records remained.

# Features used and Feature scaling
In this project, I used scikit-learn's SelectKBest module to get the 11 best features among the 21 features present.

Id         Feature                 Score 
1. exercised_stock_options: 24.815079733218194 
2. total_stock_value: 24.182898678566879 
3. bonus: 20.792252047181535 
4. salary: 18.289684043404513 
5. fraction_to_poi: 16.409712548035799 
6. deferred_income: 11.458476579280369
7. total_payments: 8.7727777300916792 
8. shared_receipt_with_poi: 8.589420731682381
9. loan_advances: 7.1840556582887247






restricted_stock: 9.2128106219771002, 
long_term_incentive: 9.9221860131898225
