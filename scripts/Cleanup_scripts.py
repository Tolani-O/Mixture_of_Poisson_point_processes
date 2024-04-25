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
