#!/usr/bin/python

import sys
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as DTree
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier


### Task 1: Select what features you'll use.

features_list = ['poi','total_payments','total_stock_value',
                'salary','bonus', 'long_term_incentive', 'expenses',
                'from_this_person_to_poi','from_poi_to_this_person']


# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print 'Number of data: ', len(data_dict)



### Task 2: Remove outliers

# I want to remove the person whose total_payments or total_stock_value is missing.
# Also, I want to remove the 'TOTAL' item
# Also, detele Belfer robert who has a total stock value of -44093
def outliers_remove(data_dict):
    new_dataset = data_dict
    for k,v in new_dataset.items():
        if k=='THE TRAVEL AGENCY IN THE PARK':
            del new_dataset[k]
        if k=='TOTAL':
            del new_dataset[k]
        if k=='BELFER ROBERT':
            del new_dataset[k]
    return new_dataset

my_dataset = outliers_remove(data_dict)


#Explore my data set
print 'Number of my dataset: ', len(my_dataset)
count = 0
for k,v in my_dataset.items():
    if v['poi'] == True:
        count +=1
print 'Number of my poi: ', count



### Task 3: Create new feature(s)

# I would like to have the total_package to be the new feature
#total_package = total_payments + total_stock_value
for k,v in my_dataset.items():
    if v['total_payments'] == 'NaN' and v['total_stock_value'] == 'NaN':
        v['total_package'] = 0
    elif v['total_payments'] == 'NaN':
        v['total_package'] = v['total_stock_value']
    elif v['total_stock_value'] == 'NaN':
        v['total_package'] = v['total_payments']
    else:
        v['total_package'] = v['total_payments'] + v['total_stock_value']


#The final feature list through importances of decision tree.
features_list = ['poi','total_stock_value',
                'bonus',  'expenses']



#Extract features and labels
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers

## Comment out from Task 4 to Task 6 and uncomment the end part
## of this code if I want to try SVM.

#My classifier
clf = DTree()
parameters_dt = {'criterion': ('gini','entropy'),
              'splitter':('best','random'),
              'min_samples_split':[2,5,10,15,20],
                'max_leaf_nodes':[5,10,30,50,100]}




### Task 5: Tune your classifier to achieve better than .3 precision and recall

cv = StratifiedShuffleSplit(labels, 100, random_state = 46)
grid_search = GridSearchCV(clf, parameters_dt, cv=cv, scoring='f1')
grid_search.fit(features, labels)
print 'best score: ', grid_search.best_score_
clf = grid_search.best_estimator_

#Test the classifier
test_classifier(clf, my_dataset, features_list)
print 'Importances: ', clf.feature_importances_



### Task 6: Dump your classifier, dataset, and features_list

dump_classifier_and_data(clf, my_dataset, features_list)






'''
# All the feature chosen
features_list = ['poi','total_payments','total_stock_value',
                'salary','bonus', 'long_term_incentive', 'expenses',
                'from_this_person_to_poi','from_poi_to_this_person']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# Using selectKBest for feature chosen
selector = SelectKBest(k=5)
features_select = selector.fit(features,labels)
features_list = [features_list[i] for i in features_select.get_support(indices=True)]
print features_list


# The classifier
clf = SVC()
parameters_svc = {'kernel':('rbf','sigmoid'),
                 'C':[1, 10,1000], 'gamma':[0.1, 1,10]}


# Tune my classifier
cv = StratifiedShuffleSplit(labels, 100, random_state = 46)
grid_search = GridSearchCV(clf, parameters_svc, cv=cv)
grid_search.fit(features, labels)
print 'best score: ', grid_search.best_score_
clf = grid_search.best_estimator_


#Dump the classifier
dump_classifier_and_data(clf, my_dataset, features_list)
'''












