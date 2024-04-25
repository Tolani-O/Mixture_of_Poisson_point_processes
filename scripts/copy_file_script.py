import os
import shutil

target_dir = 'dataSeed1947067064'
output_dir = os.path.join(os.getcwd(), target_dir)
new_dir = os.path.join(os.getcwd(), 'likelihoods_folder')
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
train_count = 0
test_count = 0
for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.endswith('log_likelihoods_train.json'):
            print('here')
            old_file_path = os.path.join(root, file)
            new_file_path = os.path.join(new_dir, f'{train_count}_{file}')
            shutil.copy2(old_file_path, new_file_path)
            train_count += 1
        elif file.endswith('log_likelihoods_test.json'):
            old_file_path = os.path.join(root, file)
            new_file_path = os.path.join(new_dir, f'{test_count}_{file}')
            shutil.copy2(old_file_path, new_file_path)
            test_count += 1