# dynamic_glmhmm
This package provides flexible and easy-to-use code for fitting Generalized Linear Models (GLMs), standard GLM-HMMs (also known as Input-Output HMMs or IO-HMMs), and **dynamic GLM-HMMs** (in which parameters are varying across sessions). The code currently allows fitting observational data of Bernoulli distributions, but can easily be adapted to fit observational data with more than two classes. Inference is done using a variation of the Expectation Maximization (EM) algorithm. See bioRxiv paper for more information.

# Package Contents

----.yml: enivornment used **etc**
dynamic_glmhmm.py: Standard or Dynamic GLM-HMM class fitting code
analysis_utils.py: a script containing functions for CV fitting models to find hyperparameters
utils.py: a script containing miscellaneous helper functions

fit_cluster_GLM_all.py: a script of 


data_IBL: a folder including pre-processed design matrix for all animals and choice behavior used in  **<cite paper>**
models_IBL: a folder showing best-fitting model parameters shown in **<cite paper>**

Each of the following jupyter notebooks recreates the plots from figures in **<cite paper>**

Installation

For easy installation, download the code using the link above or type the following into a terminal window:

git clone https://github.com/lenca56/dynamic_glmhmm.git

For convenience, we recommend setting up a virtual environment before running the code, to avoid any unpleasant version control issues or interactions with other projects you're working on. See the env.yml file for configuration details. Note the package requires python 3.7 to run.
