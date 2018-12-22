#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset_modified_unix.pkl", "rb") )
features = ["salary", "bonus"]
data_dict.pop("TOTAL", 0 )

for k in data_dict:
    if data_dict[k]['bonus'] != 'NaN' and data_dict[k]['salary'] != 'NaN':    
        if data_dict[k]['bonus'] > 5000000 or data_dict[k]['salary'] > 1000000:
            print('key_outlier: ', k)

data = featureFormat(data_dict, features)
print('len of data :' , data.shape)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]    
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

