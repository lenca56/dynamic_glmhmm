# dynamic_glmhmm

This package provides flexible and easy-to-use code for fitting Generalized Linear Models (GLMs), standard GLM-HMMs (also known as Input-Output HMMs or IO-HMMs), and **dynamic GLM-HMMs** (in which parameters are varying across sessions). The code currently allows fitting observations/output of Bernoulli distributions (binary choices), but can be adapted to fit observations/output with more than two classes. Inference is done using a variation of the Expectation Maximization (EM) algorithm. See bioRxiv paper for more information.

# Installation Instructions

For easy installation (it only takes a few minutes), download the code using the link above or type the following into a terminal window:

git clone https://github.com/lenca56/dynamic_glmhmm.git

Before running the code, we recommend setting up a virtual environment using Anaconda (add link). The "dglmhmm.yml" file contains all necessary packages for the environment. Note that the we used the 3.10 version of python. The exact versions of the other dependecies we used are found in "dglmhmm_exact.yml".

We have tested the code on both Mac OS (Sonoma 14.2) and Linux (Springdale 8) and expect it to work on all standard operating systems.

# Demo

**demo_data/dataset_two_state_simulated.npz**: demo dataset that contains the following arrays:
- trueW: true weights used to simulate data
- trueP: true transition matrix used to simulate data
- sessInd: start indices for each session
- trainX: simulated 2-dim inputs X (bias and stimulus) 
- trainY: simulated choice output Y (binary choices)
  
**Demo_simulated_data.ipynb**: demo jupyter notebook with instructions to fit the demo dataset using the dynamic GLM-HMM

The demo jupyter notebook demonstrates the same fitting procedure used for the experimental data in the paper: first fit a standard GLM-HMM to get the "mean" of the true time-varying parameters (since standard GLM-HMM has static parameters), then initialize from the best "standard" parameters and fit a partial GLM-HMM to obtain best fitting time-varying weights but constant transition matrix, and lastly initialize from the best "partial" parameters and fit a dynamic GLM-HMM to obtain the best fitting time-varying transition matrix and weights. The expectated output for the Demo jupyter notebook is for the fitted dynamic parameters to closely match the true one and for the accuracy in state 1 to increase gradually over sessions while the accuracy in state 2 to remain close to chance (50%). The expected run time for the whole notebook is 5-30 minutes.

# Package Contents

**dglmhmm.yml**: enivornment with all necessary packages

**dynamic_glmhmm.py**: contains class code for fitting standard, partial, or dynamic GLM-HMM

**analysis_utils.py**: contains functions for fitting with cross-validation to find optimal hyperparameters and for computing accuracy and occupancy in each state

**utils.py**: contains miscellaneous helper functions

**fit_cluster_GLM_all.py**: a script of 

data_IBL: a folder including pre-processed design matrix for all animals and choice behavior used in  **<cite paper>**
models_IBL: a folder showing best-fitting model parameters shown in **<cite paper>**

Each of the following jupyter notebooks recreates the plots from figures in **<cite paper>**

**dglmhmm.yml**: 

# Results on IBL dataset

To reproduce results, run in this order the following python scripts (on cluster, in parallel with slurm script):

fit_cluster_standardGLMHMM_all.py: Fits standard GLM-HMM with multiple random initialization, using the dataset with all mice data together


