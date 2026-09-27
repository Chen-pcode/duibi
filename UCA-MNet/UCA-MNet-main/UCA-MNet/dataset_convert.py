import os
import random
import shutil

# Set paths for your image and groundtruth folders
image_folder = "Polyp/train"
groundtruth_folder = "Polyp/train_lab"
validation_image_folder = "Polyp/valid"
validation_groundtruth_folder = "Polyp/valid_lab"

# Create the validation folders if they don't exist
os.makedirs(validation_image_folder, exist_ok=True)
os.makedirs(validation_groundtruth_folder, exist_ok=True)

# List all the files in the image and groundtruth folders
image_files = os.listdir(image_folder)
groundtruth_files = os.listdir(groundtruth_folder)

# Calculate the number of files to move for validation (20% of the total)
num_validation_files = int(0.2 * len(image_files))

# Randomly select files for validation
validation_files = random.sample(image_files, num_validation_files)

# Move selected validation files from image folder to validation image folder
for file_name in validation_files:
    source_image_path = os.path.join(image_folder, file_name)
    target_image_path = os.path.join(validation_image_folder, file_name)
    shutil.move(source_image_path, target_image_path)

    corresponding_groundtruth_name = file_name  # Assuming the groundtruth has the same filenames
    source_groundtruth_path = os.path.join(groundtruth_folder, corresponding_groundtruth_name)
    target_groundtruth_path = os.path.join(validation_groundtruth_folder, corresponding_groundtruth_name)
    shutil.move(source_groundtruth_path, target_groundtruth_path)

print(f"{num_validation_files} files moved to validation folders.")

