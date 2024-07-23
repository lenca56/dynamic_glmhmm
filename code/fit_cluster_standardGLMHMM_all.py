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

inits = 20

df = pd.DataFrame(columns=['init','K','signedStimulus']) # in total z=0,199 inclusively
z = 0
for init in range(0,inits):
    for K in [1,2,3,4,5]:
        for signedStimulus in [False, True]:
            df.loc[z, 'init'] = init
            df.loc[z, 'K'] = K
            df.loc[z,'signedStimulus'] = signedStimulus
            z += 1
# read from cluster array in order to get parallelizations
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
init = df.loc[idx,'init']
K = df.loc[idx,'K']
signedStimulus = df.loc[idx,'signedStimulus']

x = np.load(f'../data_IBL/X_allAnimals_signedStimulus={signedStimulus}.npy')
y = np.load(f'../data_IBL/Y_allAnimals_signedStimulus={signedStimulus}.npy')
sessInd = np.load(f'../data_IBL/sessInd_allAnimals_signedStimulus={signedStimulus}.npy')

N = x.shape[0]
D = x.shape[1]
C = 2
maxiter = 250

model_type = 'standard' # fitting standard GLM-HMM

dGLMHMM = dynamic_glmhmm.dynamic_GLMHMM(N,K,D,C)
present = np.ones((N)).astype(int) # using all data
irrelevantSigma = np.ones((K,D))
initP, initpi, initW = dGLMHMM.generate_param(sessInd=sessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)], model_type=model_type) 
standardP, standardpi, standardW, _ = dGLMHMM.fit(x, y,  present, initP=initP, initpi=initpi, initW=initW, sigma=irrelevantSigma, sessInd=sessInd, maxIter=maxiter, tol=1e-4, L2penaltyW=1, priorDirP=[10,1], model_type=model_type, fit_init_states=True) # fit the model
_, trainLl, trainAccuracy  = dGLMHMM.evaluate(x, y, sessInd, present, standardP, standardpi, standardW)

np.savez(f'../data_IBL/allAnimals_standardGLMHMM_{K}-state_init={init}_signedStimulus={signedStimulus}', P=standardP, pi=standardpi, W=standardW, trainLl=trainLl, trainAccuracy=trainAccuracy)

