#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print("Feature length: ", len(features_train[0]))
# Feature length is 3577


#########################################################
### your code goes here ###
clf = tree.DecisionTreeClassifier(min_samples_split=40)
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time: ", round(time() - t0, 3) )

t1 = time()
label_pred = clf.predict(features_test)
print("Predict Time: ", round(time() - t1, 3) )

metric = accuracy_score(labels_test, label_pred)

print("accuracy: " + str(metric))

#########################################################

"""
min_samples_split  = 40
Training Time:  38.897
Predict Time:  0.095
accuracy: 0.9730754645430413

min_samples_split  = 100
Training Time:  39.552
Predict Time:  0.081
accuracy: 0.965301478953356

min_samples_split  = 20
Training Time:  39.093
Predict Time:  0.108
accuracy: 0.978574137277209

"""

