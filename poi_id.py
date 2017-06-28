#!/usr/bin/python

import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn import cross_validation
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pprint
### Task 1: Create Features list
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person',
                 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Sample data for one of the Top executives - Jeffrey Skilling
pprint.pprint(data_dict["SKILLING JEFFREY K"])

#Number of people in the datasets

print "Total number of people in the dataset: " + str(len(data_dict))

#Number of features available in the dataset

print "Total number of features in the dataset: " + str(len(data_dict["SKILLING JEFFREY K"]))

#Number of POIs in the dataset

def poi_count(file):
    count = 0
    for data in file:
        if file[data]['poi'] == True:
            count += 1
    print "Number of POIs in the dataset: " + str(count)

poi_count(data_dict)


### Task 2: Remove outliers
features = ["bonus","salary"]
data = featureFormat(data_dict, features)
print(data.max())

for point in data:
    bonus = point[0]
    salary = point[1]
    plt.scatter( bonus, salary )

plt.xlabel("bonus")
plt.ylabel("salary")
plt.show()

#That's a huge outlier! Let's investigate it.

from pprint import pprint
outliers_bonus = []
for key in data_dict:
    val = data_dict[key]['bonus']
    if val == 'NaN':
        continue
    outliers_bonus.append((key,int(val)))

pprint(sorted(outliers_bonus,key=lambda x:x[1],reverse=True)[:2])


outliers_salary = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers_salary.append((key,int(val)))

pprint(sorted(outliers_salary,key=lambda x:x[1],reverse=True)[:2])

#Removing Total and Travel Agency in the Park outliers which did not represent an individual

features = ["salary", "bonus"]

data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
my_feature_list = features_list

data = featureFormat(data_dict, features)

#Again plotting the same graph after outlier removal

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

#Now we will investigate from_this_person_to_poi and from_poi_to_this_person for any outliers as they are important features

features = ["from_this_person_to_poi", "from_poi_to_this_person"]
data = featureFormat(data_dict, features)

print data.max()
#Plotting the graph
for point in data:
    from_this_person_to_poi = point[0]
    from_poi_to_this_person = point[1]
    plt.scatter( from_this_person_to_poi, from_poi_to_this_person )

plt.xlabel("from_this_person_to_poi")
plt.ylabel("from_poi_to_this_person")
plt.show()

#We see a few outliers, but on investigation I see that they are real persons. So I am going to keep them in the dataset

#Making dataframes in pandas one for the keys and one for the values

df_values = pd.DataFrame.from_records(list(data_dict.values()))
df_values.head()

df_persons = pd.Series(list(data_dict.keys()))
df_persons.head()

#We see lot's of NaN values. So we have to take care of that. We will convert them to numpy nan and then to zero.
df_values.replace(to_replace='NaN', value=np.nan, inplace=True)

# Count number of NaN's for columns
print df_values.isnull().sum()

# DataFrame dimension
print df_values.shape


df_null = df_values.replace(to_replace=np.nan, value=0)
df_null = df_values.fillna(0).copy(deep=True)
df_null.columns = list(df_values.columns.values)
print df_null.isnull().sum()
print df_null.head()

df_null.describe()

#Feature selection:
#For selecting the best features I am going to use SelectKBest method from scikit-learn.

def get_k_best(enron_data, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: {1}\n".format(k, k_best_features.keys())
    print k_best_features
    return k_best_features

target_label = 'poi'
num_features = 11 # 11 best features
top_features = get_k_best(data_dict, features_list, num_features)
print top_features
my_feature_list = [target_label] + top_features.keys()
# print my_feature_list

print "{0} selected features: {1}\n".format(len(my_feature_list) - 1, my_feature_list[1:])

features_list = my_feature_list

#Feature engineering
#Now let's add new features. We will use functions to do that

def add_poi_ratio(data_dict, features_list):
    """ mutates data dict to add proportion of email interaction with pois """
    fields = ['to_messages', 'from_messages',
              'from_poi_to_this_person', 'from_this_person_to_poi']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            total_messages = person['to_messages'] +\
                             person['from_messages']
            poi_messages = person['from_poi_to_this_person'] +\
                           person['from_this_person_to_poi']
            person['poi_ratio'] = float(poi_messages) / total_messages
        else:
            person['poi_ratio'] = 'NaN'
    features_list += ['poi_ratio']



def add_fraction_to_poi(data_dict, features_list):
    """ mutates data dict to add proportion of email fraction_to_poi """
    fields = ['from_messages', 'from_this_person_to_poi']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            total_messages = person['from_messages']
            poi_messages =   person['from_this_person_to_poi']
            person['fraction_to_poi'] = float(poi_messages) / total_messages
        else:
            person['fraction_to_poi'] = 'NaN'
    features_list += ['fraction_to_poi']


def add_fraction_from_poi(data_dict, features_list):
    """ mutates data dict to add proportion of email fraction_to_poi """
    fields = ['to_messages', 'from_poi_to_this_person']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            total_messages = person['to_messages']
            poi_messages =   person['from_poi_to_this_person']
            person['fraction_from_poi'] = float(poi_messages) / total_messages
        else:
            person['fraction_from_poi'] = 'NaN'
    features_list += ['fraction_from_poi']



#Adding them to the features list
add_poi_ratio(data_dict, my_feature_list)
add_fraction_to_poi(data_dict, my_feature_list)
add_fraction_from_poi(data_dict, my_feature_list)

print my_feature_list

features_list = my_feature_list
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Scaling the features by MinMaxScaler
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

#Splitting the data into train data and test data

features_train,features_test,labels_train,labels_test = cross_validation.train_test_split(features,labels, test_size=0.3, random_state=42)


### Task 4: Try a varity of classifiers
### We will iterate through variety of classifiers to see which one's prediction is the best.
### We see that Decision Tree's gives us the best accuracy of almost 83.7%. So we will use that and comment out the rest

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


clf = DecisionTreeClassifier(max_depth=10)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)
print("Decision Tree Classifier: ")
print "Accuracy: " + str(accuracy)
print "Precision Score: " + str(precision_score(labels_test,pred))
print "Recall Score: " + str(recall_score(labels_test,pred))

'''
clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=10)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)
print("Random Forest Classifier: ")
print "Accuracy: " + str(accuracy)
print "Precision Score: " + str(precision_score(labels_test,pred))
print "Recall Score: " + str(recall_score(labels_test,pred))


clf = GaussianNB()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)
print("Naive Bayes Classifier: ")
print "Accuracy: " + str(accuracy)
print "Precision Score: " + str(precision_score(labels_test,pred))
print "Recall Score: " + str(recall_score(labels_test,pred))

clf = AdaBoostClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)
print("Adaboost Classifier: ")
print "Accuracy: " + str(accuracy)
print "Precision Score: " + str(precision_score(labels_test,pred))
print "Recall Score: " + str(recall_score(labels_test,pred))

clf = SVC(gamma=3, C=2)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)
print("SVM Classifier: ")
print "Accuracy: " + str(accuracy)
print "Precision Score: " + str(precision_score(labels_test,pred))
print "Recall Score: " + str(recall_score(labels_test,pred))

clf = KNeighborsClassifier(3)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)
print("SVM Classifier: ")
print "Accuracy: " + str(accuracy)
print "Precision Score: " + str(precision_score(labels_test,pred))
print "Recall Score: " + str(recall_score(labels_test,pred))
'''


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function.

from sklearn import grid_search
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

cv = cross_validation.StratifiedShuffleSplit(labels, n_iter=10)


def scoring(estimator, features_test, labels_test):
    labels_pred = estimator.predict(features_test)
    p = precision_score(labels_test, labels_pred, average='micro')
    r = recall_score(labels_test, labels_pred, average='micro')
    if p > 0.3 and r > 0.3:
        return f1_score(labels_test, labels_pred, average='macro')
    return 0


# DecisionTreeClassifier tuning
t0 = time()
parameters = {'max_depth': [1, 2, 3, 4, 5, 6, 8, 9, 10], 'min_samples_split': [2],
              'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8], 'criterion': ('gini', 'entropy')}

decTree_clf = DecisionTreeClassifier(max_depth=10)
decTreeclf = grid_search.GridSearchCV(decTree_clf, parameters, scoring=scoring, cv=cv)

decTreeclf.fit(features, labels)
print decTreeclf.best_estimator_
print decTreeclf.best_score_
print 'Processing time:', round(time() - t0, 3), 's'

# Classifier validation
##DecisionTreeClassifier Validation 1 (StratifiedShuffleSplit, folds = 1000)
from tester import test_classifier
t0 = time()
decTree_best_clf = decTreeclf.best_estimator_
test_classifier(decTree_best_clf, my_dataset, features_list)
print 'Processing time:', round(time() - t0, 3), 's'

##DecisionTreeClassifier Validation 2  (Cross validation)

from sklearn.model_selection import cross_val_score
t0 = time()
decTree_best_clf = decTreeclf.best_estimator_
scores = cross_val_score(decTree_best_clf, features, labels, cv=5,scoring = 'accuracy')
print("Accuracy and Deviation: " + str((scores.mean(), scores.std() * 2)))
print 'Processing time:', round(time() - t0, 3), 's'
test_classifier(decTree_best_clf, my_dataset, features_list)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(decTree_best_clf, my_dataset, features_list)