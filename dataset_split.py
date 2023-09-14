import os
import shutil
from sklearn.model_selection import train_test_split

# Dataset path
src_directory = '/cs/home/psxrm17/db/plantvillage'
base_target_directory = '/cs/home/psxrm17/db/PlantVillageDataset'  

# Target directories
train_dir = os.path.join(base_target_directory, 'train')
val_dir = os.path.join(base_target_directory, 'val')
test_dir = os.path.join(base_target_directory, 'test')

#Splitting for each class
for class_name in os.listdir(src_directory):
    class_dir = os.path.join(src_directory, class_name)
    
    if os.path.isdir(class_dir):
        images = os.listdir(class_dir)
        train, temp = train_test_split(images, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

        def copy_files(files, target_dir):
            os.makedirs(target_dir, exist_ok=True)
            for file in files:
                shutil.copy(os.path.join(class_dir, file), os.path.join(target_dir, file))
        
        # Save the splitted dataset
        copy_files(train, os.path.join(train_dir, class_name))
        copy_files(val, os.path.join(val_dir, class_name))
        copy_files(test, os.path.join(test_dir, class_name))
