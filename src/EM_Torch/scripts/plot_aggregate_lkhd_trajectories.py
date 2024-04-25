import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Set the global font size
plt.rcParams['font.size'] = 14

target_dir = os.path.join(os.getcwd(), 'likelihoods_folder')
# Get a list of all files in the target directory
files = os.listdir(target_dir)
# Initialize an empty list to hold all the lists from the JSON files
list_of_lists = []
length_of_lists = []
loaded_files = []

# Loop over each file in the directory
for file in files:
    # Check if it's a JSON file
    if file.endswith('test.json'):
        # Construct the full file path
        file_path = os.path.join(target_dir, file)
        # Open and load the content of the JSON file
        with open(file_path, 'r') as f:
            content = json.load(f)
        # Append the content to the list of lists
        list_of_lists.append(content)
        # Append the length of the content to the list of lengths
        length_of_lists.append(len(content))
        # Append the file name to the list of loaded files
        loaded_files.append(file)

n = 401
# Use a list comprehension to take the first n entries of each list
trimmed_lists = [lst[1:n] for lst in list_of_lists]
# Convert the list of lists into a 2D numpy array
array_2d = np.array(trimmed_lists)
# Create a new figure
plt.figure(figsize=(10, 6))
# Plot each row with a light color
for row in array_2d:
    plt.plot(row, color='lightgray')

plt.plot(row, color='lightgray', label='Training trajectories')
truth = -298.2
# Calculate the average across rows
average = np.mean(array_2d, axis=0)
# Plot the average with a dark color
plt.plot(average, color='black', label='Average')
# Plot the truth with a dashed line
plt.plot(truth*np.ones(n), color='orange', linestyle='dashed', label='Average truth')
plt.ylabel('Log likelihood')
plt.xlabel('Iteration (x100)')
plt.title('Training trajectories for multiple simulated datasets')
plt.legend()
plt.savefig(os.path.join(target_dir, 'Test_trajectoroes.png'))
plt.close()


indcs = np.where(array_2d < -320)
# Get the unique indices
unique_indices = np.unique(indcs[0])
# Get the corresponding file names
files_to_delete = [loaded_files[i] for i in unique_indices]
print(files_to_delete)
# Delete each file
for file in files_to_delete:
    file_path = os.path.join(target_dir, file)
    os.remove(file_path)
