# dynamic_glmhmm

This package provides flexible and easy-to-use code for fitting Generalized Linear Models (GLMs), standard GLM-HMMs (also known as Input-Output HMMs or IO-HMMs), and **dynamic GLM-HMMs** (in which parameters are varying across sessions). The code currently allows fitting observations/output of Bernoulli distributions (binary choices), but could easily be adapted to fit observations/output with more than two classes. Inference is done using a variation of the Expectation Maximization (EM) algorithm. See bioRxiv paper for more information.

# Installation Instructions

For easy installation, download the code using the link above or type the following into a terminal window:

git clone https://github.com/lenca56/dynamic_glmhmm.git

For convenience, we recommend setting up a virtual environment before running the code, to avoid any unpleasant version control issues or interactions with other projects you're working on. See the env.yml file for configuration details. Note the package requires python 3.7 to run.

# Demo

**demo_data/dataset_two_state_simulated.npz**: demo dataset that contains the following arrays:
- trueW: true weights used to simulate data
- trueP: true transition matrix used to simulate data
- sessInd: start indices for each session
- trainX: simulated 2-dim inputs X (bias and stimulus) 
- trainY: simulated choice output Y (binary choices)
  
**Demo_simulated_data.ipynb**: demo jupyter notebook to fit the demo dataset with dynamic GLM-HMM


# Package Contents

**env.yml**: enivornment with all necessary packages

**dynamic_glmhmm.py**: containing class code for fitting standard or dynamic GLM-HMM

**analysis_utils.py**: containing functions for CV fitting models to find hyperparameters

**utils.py**: containing miscellaneous helper functions

**fit_cluster_GLM_all.py**: a script of 

data_IBL: a folder including pre-processed design matrix for all animals and choice behavior used in  **<cite paper>**
models_IBL: a folder showing best-fitting model parameters shown in **<cite paper>**

Each of the following jupyter notebooks recreates the plots from figures in **<cite paper>**

# Results on IBL dataset

To reproduce results, run in this order the following python scripts (on cluster, in parallel with slurm script):

fit_cluster_standardGLMHMM_all.py: Fits standard GLM-HMM with multiple random initialization, using the dataset with all mice data together


