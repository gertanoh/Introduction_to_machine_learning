#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
	
    cleaned_data = []
        
    errors = predictions - net_worths
    errors = errors * errors    
    indexes = np.argsort(errors, axis=0)
        
    number_to_extract = 0.1 * len(ages)    
    indexes = indexes[-int(number_to_extract):]       
    
    for i in range(len(ages)):
        if [i] in indexes:
            pass
        else:
            cleaned_data.append([ages[i], net_worths[i], errors[i] ])
    
    print('len of data :', len(cleaned_data))
    return cleaned_data    
		
	

