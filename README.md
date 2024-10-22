# Mixture of Poisson point process model

## Description

This repository implements a model-based clustering technique to cluster samples from temporal Poisson point processes, based on their latent intensity functions. The method involves learning the intensity function for each cluster by
fitting a mixture of Poisson point processes (analogous to a mixture of Gaussians) to an observed sample, and assigning each observed process to a cluster accordingly. The mixture of Poisson process model is a latent variable model, and
can be fit using the Expectation maximization algorithm to maximize the expected log-likelihood of the observed sample. The observed processes do not need to be aligned to their "centroid", as the method with perform a group alignment
step at each iteration. Due to the model's analytical complexity, the maximizer for some variables does not exist in closed form, and thus these are learned using gradient descent, which is implemented in PyTorch. As such, this implementation
takes full advantage of GPU acceleration and parallelization. We applied the method to clustering neuron spike trains from the visual cortex of mice, which are assumed to follow a Poisson point process, and obtained excellent results.

## File Structure

## Getting Started

* ```src/```: Contains the main source code for the Poisson Point Process Mixture Model.
tests/: Includes unit tests for the model.
data/: Directory for storing any datasets used in the project.
notebooks/: Jupyter notebooks for exploratory data analysis and model visualization.

### Prerequisites

```
      - Python 3.8 or higher
      - allensdk==2.15.1
      - cython==0.29.35
      - distro==1.8.0
      - h5py==3.8.0
      - hdmf==3.4.7
      - matplotlib==3.4.2
      - numpy==1.23.5
      - pandas==1.5.3
      - scipy==1.10.1
      - seaborn==0.12.2
      - torch==2.2.1
```

### Installation

1. Clone the repository:

```
git clone https://github.com/Tolani-O/Poisson_point_process_mixture_model.git
cd Poisson_point_process_mixture_model
```

2. Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:

```
pip install -r requirements.txt
```

### Usage
* To run the Python scripts, you will need to have python and the AllenSDK installed. You can find instructions for installing the AllenSDK [here](https://allensdk.readthedocs.io/en/latest/install.html).
* To run the R scripts, you will need to have R installed. You can find instructions for installing R [here](https://www.r-project.org/).
* To run the R scripts, you will need to have RStan installed. You can find instructions for installing RStan [here](www.mc-stan.org/users/interfaces/rstan).
* First run the Python script `download_and_format_allen_data.py` to download the data from the Allen Brain Observatory Neuropixels Visual Coding dataset. This will create a folder called `RStudioProjects/rNeuroPixel/rDataset` in the home directory. For each mouse id specified in the script, it will create a folder called `units_data_[mouse_id]`, where for each stimulus configuration and visual region, it will save the mouses spike train data for all trials of each as a datatable, which is a format readable to the R script. All mouse ids are specified in the script.
* Next, run the R script `generate_single_mouse_data.R`. This can be run from commandline with a single command line argument being a mouse id. The script will read the mouse data from the corresponding `units_data_[mouse_id]` folder, perform subset preselection for each visual area, perform naive estimation of peak times, with standard errors, compute posterior estimates of peak times and their trial to trial correlations, and finally it will write posterior summaries of the peaktimes, correlations and partial correlaitons to their corresponding folders, together with the naive correlation estimates.     

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any feature additions or bug fixes.

## License

This project does not have a license at this moment.

## Contact

For any questions or issues, please open an issue on this repository or contact the repository owner at [Tolani-O](https://github.com/Tolani-O).

## Version History

* 0.1
    * Initial Release
