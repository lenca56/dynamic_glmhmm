# dynamic_glmhmm

This package provides flexible and easy-to-use code for fitting Generalized Linear Models (GLMs), standard GLM-HMMs (also known as Input-Output HMMs or IO-HMMs), and **dynamic GLM-HMMs** (in which parameters are varying across sessions). The code currently allows fitting observations/output of Bernoulli distributions (binary choices), but can be adapted to fit observations/output with more than two classes. Inference is done using a variation of the Expectation Maximization (EM) algorithm. See bioRxiv paper for more information (https://www.biorxiv.org/content/10.1101/2024.11.30.626182v1.abstract).

# Installation Instructions

For installation (it only takes a few minutes), download the code using the link above or type the following into a terminal window:

```
git clone https://github.com/lenca56/dynamic_glmhmm.git
```

Before running the code, we recommend setting up a virtual environment using Anaconda (https://www.anaconda.com/download). The `dglmhmm.yml` file contains all necessary packages for the environment that should be used. Note that the we used the 3.10 version of python. The exact versions of all the packages we used are found in `dglmhmm_exact.yml`.

We have successfully tested the code on both Mac OS (Sonoma 14.2) and Linux (Springdale 8) and expect it to work on all standard operating systems.

# Demo

**data_demo/dataset_two_state_simulated.npz**: demo dataset that contains the following arrays:
- `trueW`: true weights used to simulate data
- `trueP`: true transition matrix used to simulate data
- `sessInd`: start indices for each session
- `trainX`: simulated 2-dim inputs X (bias and stimulus) 
- `trainY`: simulated choice output Y (binary choices)
  
**Demo_simulated_data.ipynb**: demo jupyter notebook with instructions to fit the demo dataset using the dynamic GLM-HMM

The demo jupyter notebook demonstrates the same fitting procedure used for the experimental data in the paper: first fit a standard GLM-HMM to get the "mean" of the true time-varying parameters (since standard GLM-HMM has static parameters), then initialize from the best "standard" parameters and fit a partial GLM-HMM to obtain best fitting time-varying weights but constant transition matrix, and lastly initialize from the best "partial" parameters to fit a dynamic GLM-HMM to obtain the best fitting time-varying transition matrix and weights. The expectated output for the Demo jupyter notebook is for the fitted dynamic parameters to closely match the true ones and for the accuracy in state 1 to increase gradually over sessions while the accuracy in state 2 to remain close to chance (50%). The expected run time for the whole notebook is 5-15 minutes.

# Package Contents

**`dglmhmm.yml`**: enivornment with all necessary packages

**`dglmhmm_exact.yml`**: exact enivornment we used for IBL results

**`dynamic_glmhmm.py`**: contains class code for fitting standard, partial, or dynamic GLM-HMM

**`analysis_utils.py`**: contains functions for fitting via cross-validation to find optimal hyperparameters, as well as functions for computing accuracy and occupancy in each state

**`plotting_utils.py`**: contains plotting functions

**`io_utils.py`**: contains IBL data handling functions

**`utils.py`**: contains miscellaneous helper functions

# Instructions for use

We recommend following a similar fitting procedure as used below on IBL dataset. More information can be found in Methods section of our bioRxiv paper.

# Results on IBL dataset

To reproduce results, run in this order the following python scripts (running multiple models in parallel with slurm scripts, preferrably on a cluster):

**`fit_cluster_GLM_all.py`** & **`job_GLM.slurm`**: a python script and slurm job for fitting a GLM (one state only) for IBL mice together, trying out models with different values 'pTanh', the parameter of tanh transformation of the stimulus contrast values

**`fit_cluster_standardGLMHMM_all.py`** & **`job_standardGLMHMM.slurm`**: a python script and slurm job for fitting a standard GLM-HMM for IBL mice together, trying out models with different numbers of states in the HMM and different random initializations

**`fit_cluster_partialGLMHMM.py`** & **`job_partialGLMHMM.slurm`**: a python script and slurm job for fitting a partial GLM-HMM for each IBL mouse individually, trying out models with different values of sigma, the hyperparameter governing the variability of the weights between consecutive sessions

**`fit_cluster_dynamicGLMHMM_all.py`** & **`job_dynamicGLMHMM.slurm`**: a python script and slurm job for fitting a dynamic GLM-HMM for each IBL mouse individually, trying out models with different values of alpha, the hyperparameter governing the variability of the transition matrix between consecutive sessions

To reproduce figures from our paper (https://www.biorxiv.org/content/10.1101/2024.11.30.626182v1.abstract), run the following jupyter notebooks:

**`Figure1.ipynb`**: Hypothetical learning trajectory under 2-state dynamic GLM-HMM

**`Figure2-3.ipynb`**: Fitted dynamic GLM-HMM results on example IBL mouse during training

**`Figure4-5.ipynb`**: Fitted dynamic GLM-HMM results on all 32 IBL mice analyzed during training

**`Figure6.ipynb`**: Post-training IBL mice results

**`Supplemental_Figure2.ipynb`**: Simulation of dynamic GLM-HMM flexibly using different number of states across sessions

**`Supplemental_Figure3.ipynb`**: Dynamic 3-state GLM-HMM performance comparison to other models



