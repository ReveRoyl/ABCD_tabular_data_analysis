# ABCD Tabular Data Analysis

This repository contains code for deriving and evaluating low-dimensional representations of adolescent mental health symptoms using item-level questionnaire data from the ABCD Study.

The project compares autoencoder-based representations with conventional dimensionality-reduction methods, including exploratory factor analysis, principal component analysis, and kernel principal component analysis. It also includes scripts and notebooks for model training, parameter tuning, interpretability analysis, and external validation.

## Folder Guide

### `data/`

Local data folder used by the analysis workflow.

Raw ABCD data are not included in this repository because access is controlled by the NIMH Data Archive. Users need to obtain access to the ABCD data separately and place the required files in the expected local paths.

### `notebooks/`

Jupyter notebooks for running analysis steps, checking results, and generating manuscript-related outputs.

The notebooks are used for:

* preparing and checking analysis data;
* comparing dimensionality-reduction methods;
* summarizing model outputs;
* exploring latent dimensions;
* running validation analyses;
* generating figures and tables.

### `r_scripts/`

R scripts used for preprocessing and factor-analysis-related steps before or alongside the Python workflow. 
These scripts are used to prepare input data and support exploratory factor analysis steps.

### `scripts/`

Python scripts for model training, hyperparameter tuning, and cross-validation.
These scripts are used to train autoencoder models, tune model parameters, and run cross-validation procedures outside the notebook environment.

### `utils/`

Reusable Python modules shared across notebooks and scripts.

This folder includes code for:

* autoencoder model definitions;
* training utilities;
* validation functions;
* interpretability analysis;
* general helper functions.

## Installation

Clone the repository and install the package locally:

```bash
git clone https://github.com/ReveRoyl/ABCD_tabular_data_analysis.git
cd ABCD_tabular_data_analysis
pip install -e .
```

Required packages can also be installed with:

```bash
pip install -r requirements.txt
```

## Citation

If you use this code, please cite the associated manuscript:

Luo, L., Cummins, N., Michelini, G., Wise, T., & Iniesta, R.
Delineating the Transdiagnostic Structure of Adolescent Mental Health Using Non-Linear Unsupervised Learning.

## License

This repository is released under the MIT License.
