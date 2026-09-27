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

############################################################# Prepare ISIC 2016 data set #################################################
Dataset_add = './data/dataset_isic16/'
Train_data_folder = 'ISBI2016_ISIC_Part1_Training_Data'
Train_gt_folder = 'ISBI2016_ISIC_Part1_Training_GroundTruth'
Test_data_folder = 'ISBI2016_ISIC_Part1_Test_Data'
Test_gt_folder = 'ISBI2016_ISIC_Part1_Test_GroundTruth'

# Training data and masks
Train_list = glob.glob(Dataset_add + Train_data_folder + '/*.jpg')
Test_list = glob.glob(Dataset_add + Test_data_folder + '/*.jpg')

# Initialize arrays for training data
Data_train_2016    = np.zeros([len(Train_list), height, width, channels])
Label_train_2016   = np.zeros([len(Train_list), height, width])

print('Reading ISIC 2016 Training Data')
for idx in range(len(Train_list)):
    print(f"Processing training image {idx+1}/{len(Train_list)}")
    img = sc.imread(Train_list[idx])
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode='RGB'))
    Data_train_2016[idx, :,:,:] = img

    img_name = Train_list[idx].split('/')[-1].replace('.jpg', '_Segmentation.png')
    mask_path = Dataset_add + Train_gt_folder + '/' + img_name
    img2 = sc.imread(mask_path)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_train_2016[idx, :,:] = img2

print('Reading ISIC 2016 Training Data finished')

# Shuffle the training dataset
Data_train_2016, Label_train_2016 = shuffle(Data_train_2016, Label_train_2016, random_state=1234)

# Splitting the dataset into train, validation and test sets
Train_img      = Data_train_2016[:788,:,:,:]
Validation_img = Data_train_2016[788:,:,:,:]

Train_mask      = Label_train_2016[:788,:,:]
Validation_mask = Label_train_2016[788:,:,:]

# Initialize arrays for test data
Data_test_2016    = np.zeros([len(Test_list), height, width, channels])
Label_test_2016   = np.zeros([len(Test_list), height, width])

print('Reading ISIC 2016 Test Data')
for idx in range(len(Test_list)):
    print(f"Processing test image {idx+1}/{len(Test_list)}")
    img = sc.imread(Test_list[idx])
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode='RGB'))
    Data_test_2016[idx, :,:,:] = img

    img_name = Test_list[idx].split('/')[-1].replace('.jpg', '_Segmentation.png')
    mask_path = Dataset_add + Test_gt_folder + '/' + img_name
    img2 = sc.imread(mask_path)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_test_2016[idx, :,:] = img2

print('Reading ISIC 2016 Test Data finished')

# Save datasets as .npy files
np.save('data_train', Train_img)
np.save('data_val', Validation_img)
np.save('data_test', Data_test_2016)

np.save('mask_train', Train_mask)
np.save('mask_val', Validation_mask)
np.save('mask_test', Label_test_2016)
