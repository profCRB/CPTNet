## Use this Code only after downloading respective .mat files
import scipy.io
import numpy as np
import cv2

## HSI Image loading convert into Numpy Array
data = scipy.io.loadmat('Pavia_Centre.mat')
print(data.keys())  ## Observe the key related to dataset, use that key access to dataset.

data['pavia'].shape

temp=data['pavia']
temp=temp.astype(float)

##Normalization between 0 to 1
temp=temp-np.min(temp)
temp=temp/np.max(temp)

#Saving HSI image into numpy array file to access during experiments
np.save('pavia_centre.npy',temp)  ## Use thisnumpy array during experiments

## Loading Ground truth labels 
gt_data=scipy.io.loadmat('Pavia_Centre_gt.mat')
print(gt_data.keys())  # Observe the key related to ground truth, use that key to access groud truth values.
gt_temp=gt_data['pavia_gt']
print(gt_temp.shape)

#Saving ground truth into numpy array file to access during experiments
np.save('pavia_university_gt.npy',gt_temp)
