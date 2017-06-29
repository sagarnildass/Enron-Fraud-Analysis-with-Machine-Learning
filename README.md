# Enron Fraud Analysis with Machine Learning
## By Sagarnil Das

Enron, a multibillion dollar company in the heart of the Wall Street was one of the largest companies in the United States in 2000. By 2002, it collapsed into bankruptcy due to widespread corporate fraud. They reached a dizzying height only to face a plummeting collapse. The bankruptcy of Enron affected thousands of its employees and shook the whole Wall Street by its foundation. In the federal investigation that followed, all the emails from Enron employees were made public. Since then these emails have spreaded like a wildfire through the world and many analysis and investigations have been done on this data. In this project, my target is to build a Person of Interest identifier/Label and try to build a Machine Learning Algorithm to predict the possible Persons of Interest based on various features.

I built a "person of interest" (POI) identifier to detect and predict culpable persons utilizing scikit-learn and machine learning methodologies, using features from financial data, email data, and labeled data--POIs who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

# Important files of the repository

1. Enron Fraud Analysis with Machine Learning - Final Report.ipynb - Final report in Jupyter Notebook Markdown form.
2. Enron+Fraud+Analysis+with+Machine+Learning_Final+Report.html - Final report in HTML format
3. poi_id.py - All the preprocessing, feature scaling and Machine Learning codes
4. final_project_dataset.pkl - The email features in final_project_dataset.pkl are aggregated from the email dataset, and they record the number of messages to or from a given person/email address, as well as the number of messages to or from a known POI email address and the number of messages that have shared receipt with a POI.
5. tester.py : Codes to test your classifiers, features and dataset. You don't need to change this code.
6. emails_by_address : This directory contains many text files, each of which contains all the messages to or from a particular email address. It is for your reference, if you want to create more advanced features based on the details of the emails dataset.


# Goal of this Project

The goal of this project was to utilize the financial and email data from Enron to build a predictive, analytic model that could identify whether an individual could be considered a "person of interest" (POI). Since the dataset contained labeled data--culpable persons were already listed as POIs--the value of this model on the existing dataset is limited. Rather, the potential value such a model may provide is in application to other datasets from other companies, to potentially identify suspects worth investigating further. The dataset contained 146 records with 14 financial features, 6 email features, and 1 labeled feature (POI). Of the 146 records, 18 were labeled, a priori, as persons of interest. Through exploratory data analysis and cursory spreadsheet/CSV review, I was able to identify 2 candidate records for removal:

1. TOTAL: This was an extreme outlier for most numerical features, as it was likely a spreadsheet artifact.
2. THE TRAVEL AGENCY IN THE PARK: This record did not represent an individual.

After data cleaning, 144 records remained.

# Features used and Feature scaling
In this project, I used scikit-learn's SelectKBest module to get the 11 best features among the 21 features present. In the below table you can see those features along with their scores. The K-best approach is an automated univariate feature selection algorithm, and in using it, I was concerned with the lack of email features in the resulting dataset. Thus, I engineered three features, poi_ratio, fraction_to_poi and fraction_from_poi which were the proportion of email interaction with POIs, proportion of email fraction to POIs and proportion of email fraction from POIs. So here we mutate our original data dictionary and add these new features to the features' list.


1. exercised_stock_options: 24.815079733218194 
2. total_stock_value: 24.182898678566879 
3. bonus: 20.792252047181535 
4. salary: 18.289684043404513 
5. fraction_to_poi: 16.409712548035799 
6. deferred_income: 11.458476579280369
7. long_term_incentive: 9.9221860131898225
8. restricted_stock: 9.2128106219771002
9. total_payments: 8.7727777300916792 
10. shared_receipt_with_poi: 8.589420731682381
11. loan_advances: 7.1840556582887247


Before training the machine learning algorithm classifiers, I scaled all features using a min-max scaler. This was vitally important, as the features had different units (e.g. # of email messages and USD) and varied significantly by several orders of magnitude. Feature-scaling ensured that for the applicable classifiers, the features would be weighted evenly. Only after this step, I split my data into training and testing set.

# Final Model used

I used Decision Tree Classifier to predict the possible POIs in the dataset. The main motivation for me behind using this classifier was that the Nonlinear relationships between parameters do not affect tree performance and Decision trees implicitly perform variable screening or feature selection. I also noticed that this was one of the classifiers which was giving one of the highest accuracies. Apart from Decision trees, I also tried fitting the train data and predict the test data with Random Forest, Naive Bayes, Adaboost, K Nearest Neighbours and Standard Vector Machine classifiers. Naive Bayes and SVM also gave higher accuracy scores. With Decision trees, at this point I was getting an accuracy score ranging between 83-85%

# Model Tuning

I tuned my Decision Tree Classifier model with 10 fold Stratified Shuffle Split and GridSearchCV with my customized scoring function to put a threshold for both precision_score and the recall_score. The final accuracy I got after tuning my model ranged between 87-89% with a precision score of 0.56 and the recall score of 0.45.

# Conclusion

So we see by applying Stratified Shuffle Split to our Decision Tree Classifier, we are able to improve the accuracy of our classifier by 1 - 2% . So right now, it is giving us an accuracy of ~87% . We also see that ratio of the false positives to true positives is much higher than the ratio of false negatives to true negatives. This is actually good for us because we would rather mark a person as POI who in reality is not a POI than not able to catch the Real POIs and marking them as False POIs. My argument here is even though we mark someone as a POI wrongly, he/she will still get through unscathed after investigation. But on the other hand, if we miss someone who is indeed guilty, then that person is just not facing the justice as he/she should have.






