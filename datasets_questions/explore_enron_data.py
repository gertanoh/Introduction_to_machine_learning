#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pandas as pd
import numpy as np

def start_up():

	enron_data = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb"))
	export = data.to_csv('enron.csv', index = None, header=True)
	
	
data_1 = pd.read_csv('enron.csv')
print("data info")
print(data_1)
print("Nan Count")
print(data_1.isnull().sum())
poi_data = data_1[data_1.poi == True]
print("Number of poi : ", len(poi_data))
print('POI NaN count')
print(poi_data.isnull().sum())
