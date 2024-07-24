from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# from oneibl.onelight import ONE # only used for downloading data
# import wget
from utils import *
from plotting_utils import *
from analysis_utils import *
import dynamic_glmhmm
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import sys
import os
from io_utils import *

ibl_data_path = '../data_IBL'
dfAll = pd.read_csv(ibl_data_path + '/Ibl_processed.csv')

# list of all animals
labChosen =  ['angelakilab','churchlandlab','wittenlab']
subjectsAll = []
for lab in labChosen:
    subjects = np.unique(dfAll[dfAll['lab'] == lab]['subject']).tolist()
    subjectsAll = subjectsAll + subjects
# removing missing or incomplete animals
if ('NYU-01' in subjectsAll):
    subjectsAll.remove('NYU-01')
if ('NYU-06' in subjectsAll):
    subjectsAll.remove('NYU-06')
if ('CSHL_007' in subjectsAll):
    subjectsAll.remove('CSHL_007')
if ('CSHL049' in subjectsAll):
    subjectsAll.remove('CSHL049')


inits = 20

df = pd.DataFrame(columns=['subject','K','signedStimulus']) # in total z=0,199 inclusively
z = 0
for subject in subjectsAll:
    for K in [1,2,3,4,5]:
        for signedStimulus in [False, True]:
            df.loc[z, 'subject'] = subject
            df.loc[z, 'K'] = K
            df.loc[z,'signedStimulus'] = signedStimulus
            z += 1
# read from cluster array in order to get parallelizations
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
subject = df.loc[idx,'subject']
K = df.loc[idx,'K']
signedStimulus = df.loc[idx,'signedStimulus']

# load data for particular animal
x, y, sessInd, correctSide = get_mouse_design(dfAll, subject=subject, sessStop=None, signedStimulus=signedStimulus, pTanh=5) # with tanh transformation


# setting hyperparameters
N = x.shape[0]
D = x.shape[1]
C = 2
sigmaList = [10**x for x in list(np.arange(-3,1,0.5,dtype=float))] + [10**x for x in list(np.arange(1,4,1,dtype=float))]
L2penaltyW = 1
priorDirP = [100,10]
maxiter = 300
splitFolds = 5
fit_init_states = False

model_type = 'partial' # fitting standard GLM-HMM

# initializing parameters and functions to save across all sigmas and folds
trainLl = np.zeros((splitFolds, len(sigmaList) + 1, maxiter))
testLl = np.zeros((splitFolds, len(sigmaList) + 1))
testAccuracy = np.zeros((splitFolds, len(sigmaList) + 1))
allP = np.zeros((splitFolds, len(sigmaList) + 1, K, K))
allW = np.zeros((splitFolds, len(sigmaList)+ 1, N, K, D, 2)) 


# initialize model from best fitting parameters of standard GLM-HMM
dataInit = np.load(f'../data_IBL/Best_standardGLMHMM_allAnimals_signedStimulus={signedStimulus}_{K}-state.npz')
initP = dataInit['P']
initpi = dataInit['pi']
initW = dataInit['W']



# fitting
for fold in range(0,splitFolds):    
    allP[fold], _, allW[fold], trainLl[fold], testLl[fold], testAccuracy[fold] = fit_eval_CV_multiple_sigmas(K, x, y, sessInd, presentTrain[fold], presentTest[fold], sigmaList=sigmaList, maxiter=maxiter, glmhmmW=glmhmmW, glmhmmP=glmhmmP, L2penaltyW=L2penaltyW, priorDirP=priorDirP, fit_init_states=fit_init_states)
    

dGLMHMM = dynamic_glmhmm.dynamic_GLMHMM(N,K,D,C)
present = np.ones((N)).astype(int) # using all data

# fit with cross-validation across multiple sigmas 


standardP, standardpi, standardW, _ = dGLMHMM.fit(x, y,  present, initP=initP, initpi=initpi, initW=initW, sigma=sigma, sessInd=sessInd, maxIter=maxiter, tol=1e-4, L2penaltyW=1, priorDirP=[10,1], model_type=model_type, fit_init_states=True) # fit the model
_, trainLl, trainAccuracy  = dGLMHMM.evaluate(x, y, sessInd, present, standardP, standardpi, standardW)

np.savez(f'../data_IBL/allAnimals_standardGLMHMM_{K}-state_init={init}_signedStimulus={signedStimulus}', P=standardP, pi=standardpi, W=standardW, trainLl=trainLl, trainAccuracy=trainAccuracy)

