# ABCD_tabular_data_analysis

This repository contains code for factor analysis using autoencoder.

## Data pre-processing

For data preprocessing, run `notebooks/data_preparation.ipynb` first and run `r_scripts/remove_unrelated.r`

## Installation

To install this package, clone the repository and run `pip install -e .` from the root directory. It can then be used as a regular Python package, e.g. `from models import Autoencoder.train`.

## Examples

An example notebook is provided in the `notebooks` directory.

To Use nni for tunning, `config.yml` and `search_space.json` are provided, run `nnictl create --config config.yml` from the root directory.
