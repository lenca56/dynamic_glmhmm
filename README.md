# dynamic_glmhmm
This package provides flexible and easy-to-use code for fitting Generalized Linear Models (GLMs), standard GLM-HMMs (also known as Input-Output HMMs or IO-HMMs), and **dynamic GLM-HMMs** (in which parameters are varying across sessions). The code currently allows fitting observational data of Bernoulli distributions, but can easily be adapted to fit observational data with more than two classes. Inference is done using a variation of the Expectation Maximization (EM) algorithm. See bioRxiv paper for more information.

# Package Contents

----.yml: enivornment used **etc**
dynamic_glmhmm.py: Standard or Dynamic GLM-HMM class fitting code
analysis_utils.py: a script containing functions for CV fitting models to find hyperparameters
utils.py: a script containing miscellaneous helper functions
analysis.py: a script containing post-fitting analysis functions used in Bolkan, Stone et al 2021
visualize.py: a script containing functions for plotting the figures seen in Bolkan, Stone et al 2021
examples

fit_cluster_GLM_all.py: a script of 

fit-glm.ipynb: a simple example of fitting a GLM to simulated data
fit-hmm.ipynb: a simple example of fitting an HMM to simulated data
fit-hmm-DAEM.ipynb: an example illustrating the benefits of deterministic annealing EM (DAEM)
fit-glm-hmm.ipynb: an example of fitting GLM-HMMs to simulated data
figures


data_IBL: a folder including pre-processed design matrix for all animals and choice behavior used in  **<cite paper>**
models_IBL: a folder showing best-fitting model parameters shown in **<cite paper>**

Each of the following jupyter notebooks recreates the plots from figures in **<cite paper>**

fig4.ipynb: fits a GLM to real data and interprets results
fig5.ipynb: compares model performance between a standard Bernoulli GLM and a 3-state GLM-HMM
fig6.ipynb: fits a 3-state GLM-HMM to real data and interpret results
fig7.ipynb: analyzes how the three states identified by the GLM-HMM manifest in the data
extdatafig7: describes model selection and control analyses
extdatafig9: shows how model simulations recapitulate characteristics of the real data
suppfig4: shows how individual mice occupy different states for each session of the task

Installation

For easy installation, download the code using the link above or type the following into a terminal window:

git clone https://github.com/lenca56/dynamic_glmhmm.git

For convenience, we recommend setting up a virtual environment before running the code, to avoid any unpleasant version control issues or interactions with other projects you're working on. See the env.yml file for configuration details. Note the package requires python 3.7 to run.
