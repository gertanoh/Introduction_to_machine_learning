#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# reducing to train faster
# features_train = features_train[:int(len(features_train)/100)] 
# labels_train = labels_train[:int(len(labels_train)/100)] 


#########################################################
### your code goes here ###

# Cs = [10, 100, 1000, 10000]

clf = svm.SVC(kernel='linear', C=10000)
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time: ", round(time() - t0, 3) )

t1 = time()

label_pred = clf.predict(features_test)
metric = accuracy_score(labels_test, label_pred)
print("Predict Time: ", round(time() - t1, 3) )
print("accuracy: " , metric)

print("10th : ", label_pred[10])
print("26th : ", label_pred[26])
print("50th : ", label_pred[50])


chris_pred = 0
for y in label_pred:
	if y == 1:
		chris_pred += 1
		
print("Chris predictions: ", chris_pred)

#########################################################



"""
Linear kernel

Training Time:  180.671
Predict Time:  67.794
accuracy: 0.9825559347743648

Rbf Kernel
Smaller data
Training Time:  0.103
Predict Time:  4.46
accuracy: 0.9091770951839211

Optimizing C
accuracy:  [0.49943117178612056, 0.49943117178612056, 0.9032992036405005, 0.9059537353052711]
"""
