# CPTNet

## Creating Numpy array from.mat files
Download dataset related .mat files from https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
Then use the dataset_creation.py code to convert numpy arrays from .mat files

## Dataset loading selecting number of samples for training and validaion in CPTNet.py code
Use the following python code statements in CPTNet.py to load numpy arrays and select number of traning and validation samples.

### Loading numpy files for Indian Pines Dataset
```
file_path_temp = '/content/drive/MyDrive/IndianPines/indian_pines_corrected.npy'
temp = np.load(file_path_temp)
file_path_gt_temp = '/content/drive/MyDrive/IndianPines/indian_pines_gt.npy'
gt_temp=np.load(file_path_gt_temp)
temp_og=np.copy(temp)
temp=temp.reshape(temp.shape[0],temp.shape[1],20,10)
temp,pca_obj=PCA_fit_transform(temp,5)
temp=np.transpose(temp,axes=[0,1,3,2])
print(temp.shape)
```

### Selecting  Number of samples for Training and Validation for Indian Pines datset
```
Trn_IP=[0,10,71,41,11,24,36,10,23,10,48,122,29,10,63,19,10]
Vld_IP=[0,10,71,41,11,24,36,10,23,5,48,122,29,10,63,19,10]
```

### Loading numpy files for Salinas Dataset
```
file_path_temp = '/content/drive/MyDrive/Salinas/salinas_corrected.npy'
temp = np.load(file_path_temp)
file_path_gt_temp = '/content/drive/MyDrive/Salinas/salinas_gt.npy'
gt_temp=np.load(file_path_gt_temp)
temp_og=np.copy(temp)
temp=temp[:,:,0:200]
temp=temp.reshape(temp.shape[0],temp.shape[1],20,10)
temp,pca_obj=PCA_fit_transform(temp,5)
temp=np.transpose(temp,axes=[0,1,3,2])
print(temp.shape)
```
### Selecting  Number of samples for Training and Validation for Salinas datset
```
Trn_IP=[0,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
Vld_IP=[0,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
```

### Loading numpy files for Pavia Centre Dataset
```
file_path_temp = '/content/drive/MyDrive/PaviaCentre/pavia_centre.npy'
temp = np.load(file_path_temp)
file_path_gt_temp = '/content/drive/MyDrive/PaviaCentre/pavia_centre_gt.npy'
gt_temp=np.load(file_path_gt_temp)
temp_og=np.copy(temp)
temp=temp[:,:,0:100]
temp=temp.reshape(temp.shape[0],temp.shape[1],10,10)
temp,pca_obj=PCA_fit_transform(temp,5)
temp=np.transpose(temp,axes=[0,1,3,2])
print(temp.shape)
```
### Selecting  Number of samples for Training and Validation for Pavia Centre datset
```
Trn_IP=[0,20,20,20,20,20,20,20,20,20]
#Vld_IP=[0,20,20,20,20,20,20,20,20,20]
```

