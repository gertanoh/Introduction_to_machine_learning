#!/usr/bin/python


"""
    Recall: True Positive / (True Positive + False Negative).
    Out of all the items that are truly positive, 
    how many were correctly classified as positive. 
    Or simply, how many positive items were 'recalled' from the dataset.

    Precision: True Positive / (True Positive + False Positive). 
    Out of all the items labeled as positive, 
    how many truly belong to the positive class.
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

data_dict = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys_modified.pkl')
labels, features = targetFeatureSplit(data)


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


### it's all yours from here forward!  
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

labels_pred = clf.predict(features_test)

metric = accuracy_score(labels_test, labels_pred)
print("accuracy: " + str(metric))

print('len of test set: ', len(features_test))
print('precision_score: ', precision_score(labels_test, labels_pred))
print('recall_score: ', recall_score(labels_test, labels_pred))
print('confusion_matrix')
print(confusion_matrix(labels_test, labels_pred))

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]