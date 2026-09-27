# -*- coding: utf-8 -*-
"""
Code created on [current date]
@author: [Your Name]
"""

import h5py
import numpy as np
import scipy.io as sio
import scipy.misc as sc
import glob
from sklearn.utils import shuffle

# Parameters
height = 256
width  = 256
channels = 3

############################################################# Prepare ISIC 2017 data set #################################################
Dataset_add = './data/dataset_isic17/'
Train_data_folder = 'train/images'
Train_gt_folder = 'train/masks'
Test_data_folder = 'val/images'
Test_gt_folder = 'val/masks'

# Training data and masks
Train_list = glob.glob(Dataset_add + Train_data_folder + '/*.jpg')
Test_list = glob.glob(Dataset_add + Test_data_folder + '/*.jpg')

# Initialize arrays for training data
Data_train_2017    = np.zeros([len(Train_list), height, width, channels])
Label_train_2017   = np.zeros([len(Train_list), height, width])

print('Reading ISIC 2017 Training Data')
for idx in range(len(Train_list)):
    print(f"Processing training image {idx+1}/{len(Train_list)}")
    img = sc.imread(Train_list[idx])
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode='RGB'))
    Data_train_2017[idx, :,:,:] = img

    img_name = Train_list[idx].split('/')[-1].replace('.jpg', '_segmentation.png')
    mask_path = Dataset_add + Train_gt_folder + '/' + img_name
    img2 = sc.imread(mask_path)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_train_2017[idx, :,:] = img2

print('Reading ISIC 2017 Training Data finished')

# Shuffle the training dataset
Data_train_2017, Label_train_2017 = shuffle(Data_train_2017, Label_train_2017, random_state=42)

# Splitting the dataset into train, validation and test sets
Train_img      = Data_train_2017[:1285,:,:,:]
Validation_img = Data_train_2017[1285:,:,:,:]

Train_mask      = Label_train_2017[:1285,:,:]
Validation_mask = Label_train_2017[1285:,:,:]

# Initialize arrays for test data
Data_test_2017    = np.zeros([len(Test_list), height, width, channels])
Label_test_2017   = np.zeros([len(Test_list), height, width])

print('Reading ISIC 2017 Test Data')
for idx in range(len(Test_list)):
    print(f"Processing test image {idx+1}/{len(Test_list)}")
    img = sc.imread(Test_list[idx])
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode='RGB'))
    Data_test_2017[idx, :,:,:] = img

    img_name = Test_list[idx].split('/')[-1].replace('.jpg', '_segmentation.png')
    mask_path = Dataset_add + Test_gt_folder + '/' + img_name
    img2 = sc.imread(mask_path)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_test_2017[idx, :,:] = img2

print('Reading ISIC 2017 Test Data finished')

# Save datasets as .npy files
np.save('data_train', Train_img)
np.save('data_val', Validation_img)
np.save('data_test', Data_test_2017)

np.save('mask_train', Train_mask)
np.save('mask_val', Validation_mask)
np.save('mask_test', Label_test_2017)
