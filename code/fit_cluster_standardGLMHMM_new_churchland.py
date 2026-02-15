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
dfAll = pd.read_csv(ibl_data_path + '/IBL_processed_new_batch_churchland.csv')
labChosen =  ['churchlandlab']
subjectsAll = []
for lab in labChosen:
    subjects = np.unique(dfAll[dfAll['lab'] == lab]['subject']).tolist()
    subjectsAll = subjectsAll + subjects
print(f'total subjects {len(subjectsAll)}')

df = pd.DataFrame(columns=['init','K','signedStimulus', 'pTanh']) # in total z=0,399 inclusively
z = 0
for K in [3]:
    for signedStimulus in [True]:
        for pTanh in [5]:
            for subject_idx in range(0,len(subjectsAll)):
                df.loc[z, 'subject_idx'] = subject_idx
                df.loc[z, 'K'] = K
                df.loc[z, 'pTanh'] = pTanh
                df.loc[z,'signedStimulus'] = signedStimulus
                z += 1

# read from cluster array in order to get parallelizations
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
subject_idx = df.loc[idx,'subject_idx'].astype(int)
K = df.loc[idx,'K']
pTanh = df.loc[idx, 'pTanh']
signedStimulus = df.loc[idx,'signedStimulus']
subject = subjectsAll[subject_idx]

# create dataset with current pTanh transformation
x, y, sessInd, correctSide, responseTimes = get_mouse_design_new_batch(dfAll, subject=subject, sessStop=None, signedStimulus=signedStimulus, pTanh=pTanh) 

N = x.shape[0]
D = x.shape[1]
C = 2
maxiter = 300

model_type = 'standard' # fitting standard GLM-HMM

# initialize model from best fitting parameters of standard GLM-HMM on all animals (old batch)
dataInit = np.load(f'../data_IBL/all_animals/Best_allAnimals_standardGLMHMM_{K}-state_pTanh={pTanh}_signedStimulus={signedStimulus}.npz')

initP = np.zeros((N, K, K))
initpi = np.zeros((K))
initW = np.zeros((N, K, D, C)) 

initP[:] = dataInit['P'][0]
initpi[:] = dataInit['pi'][0]
initW[:] = dataInit['W'][0]

dGLMHMM = dynamic_glmhmm.dynamic_GLMHMM(N,K,D,C)
present = np.ones((N)).astype(int) # using all data
irrelevantSigma = np.ones((K,D))
standardP, standardpi, standardW, _ = dGLMHMM.fit(x, y,  present, initP=initP, initpi=initpi, initW=initW, sigma=irrelevantSigma, sessInd=sessInd, maxIter=maxiter, tol=1e-4, L2penaltyW=1, priorDirP=[10,1], model_type=model_type, fit_init_states=False) # fit the model
_, trainLl, trainAccuracy  = dGLMHMM.evaluate(x, y, sessInd, present, standardP, standardpi, standardW)

# saving parameters for each session to optimize memory
np.savez(f'../data_IBL/new_batch/New_batch_Churchland_standardGLMHMM_subject-{subject}_{K}-state_pTanh={pTanh}_signedStimulus={signedStimulus}', P=standardP[sessInd[:-1]], pi=standardpi, W=standardW[sessInd[:-1]], trainLl=trainLl, trainAccuracy=trainAccuracy)

