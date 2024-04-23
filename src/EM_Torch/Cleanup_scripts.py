import os
from src.EM_Torch.general_functions import remove_minimums, remove_chunk

folder_name = 'GroundTruthInit+NoPenalty+Full_dataSeed1602141409_K60_R15_A3_C3_R15_tauBeta0_tauConfig0_tauSigma0_iters10000_BatchSizeAll_lr0.001'
output_dir = os.path.join(os.getcwd(), 'outputs', folder_name)
remove_minimums(-15, output_dir, 'Train', 'Likelihood', min_cutoff=300, write=1)

remove_chunk(-15, output_dir, 'Test', 'Likelihood', 770, -1, write=1)
remove_chunk(-15, output_dir, 'Train', 'Likelihood', 770, -1, write=1)
file_names = [
        'beta_MSE',
        'alpha_MSE',
        'theta_MSE',
        'pi_MSE',
        'configoffset_MSE',
        'ltri_MSE',
        'clusr_misses',
        'gains_MSE',
        'trialoffsets_MSE',
    ]
for file_name in file_names:
    remove_chunk(None, output_dir, 'Test', file_name, 770, -1, write=1)

file_names = [
        'clusr_misses',
        'gains_MSE',
        'trialoffsets_MSE',
    ]
for file_name in file_names:
    remove_chunk(None, output_dir, 'Train', file_name, 770, -1, write=1)



import os
import shutil

output_dir = os.path.join(os.getcwd(), 'outputs')
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