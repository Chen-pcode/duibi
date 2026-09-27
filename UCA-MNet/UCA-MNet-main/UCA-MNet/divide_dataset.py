import os
import shutil
import numpy as np

# Paths to your dataset folders
images_path = 'ISIC-2017/imgsContrast'
ground_truth_path = 'ISIC-2017/gt'

# Paths for the output dataset
output_base = 'divided_dataset'
train_path = os.path.join(output_base, 'train')
val_path = os.path.join(output_base, 'validation')
test_path = os.path.join(output_base, 'test')

# Create directories if they do not exist
for path in [train_path, val_path, test_path]:
    os.makedirs(os.path.join(path, 'imgsContrast'), exist_ok=True)
    os.makedirs(os.path.join(path, 'gt'), exist_ok=True)

# Get a list of filenames without file extension
filenames = [os.path.splitext(file)[0] for file in os.listdir(images_path)]
np.random.shuffle(filenames)

# Split filenames into train, validation, and test sets
n_total = len(filenames)
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.15)
# The remainder for test set
train_filenames, val_filenames, test_filenames = np.split(filenames, [n_train, n_train + n_val])

# Function to copy files
def copy_files(filenames, source, destination):
    for filename in filenames:
        for extension in ['.jpg', '.png', '.txt']:  # Add/remove extensions as needed
            src_file_path = os.path.join(source, filename + extension)
            if os.path.exists(src_file_path):  # Check if file exists to avoid errors
                shutil.copy(src_file_path, os.path.join(destination, filename + extension))

# Copy files to their respective directories
copy_files(train_filenames, images_path, os.path.join(train_path, 'images'))
copy_files(train_filenames, ground_truth_path, os.path.join(train_path, 'ground_truth'))
copy_files(val_filenames, images_path, os.path.join(val_path, 'images'))
copy_files(val_filenames, ground_truth_path, os.path.join(val_path, 'ground_truth'))
copy_files(test_filenames, images_path, os.path.join(test_path, 'images'))
copy_files(test_filenames, ground_truth_path, os.path.join(test_path, 'ground_truth'))

print(f'Dataset divided: {len(train_filenames)} train, {len(val_filenames)} validation, {len(test_filenames)} test')
