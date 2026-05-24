# Cascade Pixel Transformer with Distance-Driven Spatial Fusion for Hyperspectral Image Classification Using Limited Training Samples
## Paper Id: 10534
## Journal: IEEE Latin America Transactions
## Authors
Biri Chanakya Reddy, Research Scholar, Department od ECE, Natitional Institute of Technology Tiruchirappalli, India.

Subbian Deivalakshmi, Associate Professor, Department od ECE, Natitional Institute of Technology Tiruchirappalli, India.

# File Description
## datasets Folder
datasets/indian_pines.md file consists downloadable links for Indian Pines dataset along with ground truth labels.

datasets/salinas.md file consists downloadable links for Salinas dataset along with ground truth labels.

datasets/pavia_centre.md file consists downloadable links for Pavia Centre dataset along with ground truth labels.

## CPTNet.py
This files consists python implementation of proposed method. 
Follow the below instructions to implement proposed method against Indian Pines, Salinas, and Pavia Centre datasets.
### Implementation for Indian Pines dataset.
Download the dataset files "indian_pines_corrected.npy" and "indian_pines_gt.npy" by using links in the datasets/indian_pines.md file.

Keep the downloaded files in same folder as CPTNet.py file.

Modify the file loading statements in CPTNet.py as shown below.
```
file_path_temp = 'indian_pines_corrected.npy'
temp = np.load(file_path_temp)
file_path_gt_temp = 'indian_pines_gt.npy'
gt_temp=np.load(file_path_gt_temp)
temp_og=np.copy(temp)
temp=temp.reshape(temp.shape[0],temp.shape[1],20,10)
temp,pca_obj=PCA_fit_transform(temp,5)
temp=np.transpose(temp,axes=[0,1,3,2])
print(temp.shape)
```
Set the Number of samples for Training and Validation for Indian Pines datset by following modifications in CPTNet.py
```
Trn_IP=[0,10,71,41,11,24,36,10,23,10,48,122,29,10,63,19,10]
Vld_IP=[0,10,71,41,11,24,36,10,23,5,48,122,29,10,63,19,10]
```
Run the CPTNet.py to get the results for Indian Pines dataset.

### Implementation for Salinas dataset.
Download the dataset files "salinas_corrected.npy" and "salinas_gt.npy" by using links in the datasets/salinas.md file.

Keep the downloaded files in same folder as CPTNet.py file.

Modify the file loading statements in CPTNet.py as shown below.
```
file_path_temp = 'salinas_corrected.npy'
temp = np.load(file_path_temp)
file_path_gt_temp = 'salinas_gt.npy'
gt_temp=np.load(file_path_gt_temp)
temp_og=np.copy(temp)
temp=temp[:,:,0:200]
temp=temp.reshape(temp.shape[0],temp.shape[1],20,10)
temp,pca_obj=PCA_fit_transform(temp,5)
temp=np.transpose(temp,axes=[0,1,3,2])
print(temp.shape)
```
Set the Number of samples for Training and Validation for Salinas datset by following modifications in CPTNet.py
```
Trn_IP=[0,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
Vld_IP=[0,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
```
Run the CPTNet.py to get the results for Salinas dataset.

### Implementation for Pavia Centre dataset.
Download the dataset files "pavia_centre.npy" and "pavia_centre_gt.npy" by using links in the datasets/salinas.md file.

Keep the downloaded files in same folder as CPTNet.py file.

Modify the file loading statements in CPTNet.py as shown below.
```
file_path_temp = 'pavia_centre.npy'
temp = np.load(file_path_temp)
file_path_gt_temp = 'pavia_centre_gt.npy'
gt_temp=np.load(file_path_gt_temp)
temp_og=np.copy(temp)
temp=temp[:,:,0:100]
temp=temp.reshape(temp.shape[0],temp.shape[1],10,10)
temp,pca_obj=PCA_fit_transform(temp,5)
temp=np.transpose(temp,axes=[0,1,3,2])
print(temp.shape)
```
Set the Number of samples for Training and Validation for Pavia Centre datset by following modifications in CPTNet.py
```
Trn_IP=[0,20,20,20,20,20,20,20,20,20]
Vld_IP=[0,20,20,20,20,20,20,20,20,20]
```
Run the CPTNet.py to get the results for Pavia Centre dataset.

## dataset_creation.py  File
This is optional file in this repository.
This coding file is used prepare numpy datasets from .mat files available in following well known hyperspectral repository.
https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes  

After downloading .mat files, use the dataset_creation.py to generate numpy version of datasets.



