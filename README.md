# Mixture of Poisson point process model

## Description

This repository implements a model-based clustering technique to cluster samples from temporal Poisson point processes, based on their latent intensity functions. The method involves learning the intensity function for each cluster by fitting a mixture of Poisson point processes (analogous to a mixture of Gaussians) to an observed sample, and assigning each observed process to a cluster accordingly. The mixture of Poisson process model is a latent variable model, and can be fit using the Expectation maximization algorithm to maximize the expected log-likelihood of the observed sample. The observed processes do not need to be aligned to their "centroid", as the method with perform a group alignment step at each iteration, using time warping. Due to the model's analytical complexity, the maximizer for some variables does not exist in closed form, and thus these are learned using gradient descent, which is implemented in PyTorch. As such, this implementation takes full advantage of GPU acceleration and parallelization. We applied the method to clustering neuron spike trains from the visual cortex of mice, which are assumed to follow a Poisson point process. Our results are available on Arxiv.

## File Structure

## Getting Started

* ```LikelihoodELBOModel.py```: Contains the main source code for the Poisson Point Process Mixture Model, defined in the class ```LikelihoodELBOModel```.
* ```general_functions.py```: Includes utility and helper functions for the model, such as logging and plotting functions.
* ```simulate_data_multitrial.py```: Contains code to generate simulated data point process data on which to apply clustering, defined in the class ```DataAnalyzer```.
* ```Allen_data_torch.py```: Contains code to load point process data recorded from the visual cortex of mice from the Allen brain observatory, defined in the class ```EcephysAnalyzer```. The full Allen Brain Observatory Neuropixels Visual Coding dataset can be accessed via the [AllenSDK](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html), the [DANDI Archive](https://dandiarchive.org/dandiset/000021), and through [AWS Registry of Open Data](https://registry.opendata.aws/allen-brain-observatory/).
* ```main_em_torch.py```: Script to run the clustering algorithm on sample simulated data.
* ```load_trained_em_model.py```: Script to resume training of a previously run model on simulated data from a given checkpoint.
* ```main_load_allen_data.py```: Script to run the clustering algorithm on the Allen observatory data.
* ```load_trained_allen_model.py```: Script to resume training of a previously run model on Allen observatory data from a given checkpoint.

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

4. Install Allen SDK:

Instructions for installing the AllenSDK can be found [here](https://allensdk.readthedocs.io/en/latest/install.html).

### Usage

Simply run ```main_em_torch.py``` or ```main_load_allen_data.py```.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any feature additions or bug fixes.

## License

This project does not have a license at this moment.

## Contact

For any questions or issues, please open an issue on this repository or contact the repository owner at [Tolani-O](https://github.com/Tolani-O).

## Version History

* 0.1
    * Initial Release
