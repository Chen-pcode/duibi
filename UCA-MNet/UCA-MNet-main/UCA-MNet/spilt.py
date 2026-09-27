import os
import shutil
import random

# Define the source directories
image_source_dir = '/DATA/home/zyw/Razan/Datasets/PH2/Dermoscopic_Images'  # Dermoscopic images are .jpg
segmentation_source_dir = '/DATA/home/zyw/Razan/Datasets/PH2/Segmentation_Images'  # Segmentation images are .png

# Define the target directories
train_image_dir = '/DATA/home/zyw/Razan/Datasets/ph2-new/train'
valid_image_dir = '/DATA/home/zyw/Razan/Datasets/ph2-new/valid'
test_image_dir = '/DATA/home/zyw/Razan/Datasets/ph2-new/test'

train_lab_dir = '/DATA/home/zyw/Razan/Datasets/ph2-new/train_lab'
valid_lab_dir = '/DATA/home/zyw/Razan/Datasets/ph2-new/valid_lab'
test_lab_dir = '/DATA/home/zyw/Razan/Datasets/ph2-new/test_lab'

# Create target directories if they don't exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(valid_image_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)

os.makedirs(train_lab_dir, exist_ok=True)
os.makedirs(valid_lab_dir, exist_ok=True)
os.makedirs(test_lab_dir, exist_ok=True)

# Get a list of all image files in the source directory
all_images = [f for f in os.listdir(image_source_dir) if os.path.isfile(os.path.join(image_source_dir, f))]

# Shuffle the list of images
random.shuffle(all_images)

# Define the number of images for each set
num_train = 140
num_valid = 20
num_test = 40

# Split the images into train, valid, and test sets
train_images = all_images[:num_train]
valid_images = all_images[num_train:num_train + num_valid]
test_images = all_images[num_train + num_valid:num_train + num_valid + num_test]

# Move images and their corresponding segmentation files to their respective directories
for image in train_images:
    # Copy dermoscopic image
    shutil.copy(os.path.join(image_source_dir, image), train_image_dir)
    
    # Determine corresponding segmentation file
    base_name = image.replace('.jpg', '')  # Remove .jpg extension
    seg_file = f"{base_name}_lesion.png"  # Create corresponding segmentation file name
    shutil.copy(os.path.join(segmentation_source_dir, seg_file), train_lab_dir)

for image in valid_images:
    # Copy dermoscopic image
    shutil.copy(os.path.join(image_source_dir, image), valid_image_dir)

    # Determine corresponding segmentation file
    base_name = image.replace('.jpg', '')
    seg_file = f"{base_name}_lesion.png"
    shutil.copy(os.path.join(segmentation_source_dir, seg_file), valid_lab_dir)

for image in test_images:
    # Copy dermoscopic image
    shutil.copy(os.path.join(image_source_dir, image), test_image_dir)

    # Determine corresponding segmentation file
    base_name = image.replace('.jpg', '')
    seg_file = f"{base_name}_lesion.png"
    shutil.copy(os.path.join(segmentation_source_dir, seg_file), test_lab_dir)

print("Images and corresponding segmentation files have been split into train, valid, and test sets.")