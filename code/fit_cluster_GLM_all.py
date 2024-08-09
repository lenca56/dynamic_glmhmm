from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
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

df = pd.DataFrame(columns=['pTanh','signedStimulus']) # in total z=0,29 inclusively
z = 0
for pTanh in [0.01, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]:
    for signedStimulus in [False, True]:
        df.loc[z, 'pTanh'] = pTanh
        df.loc[z,'signedStimulus'] = signedStimulus
        z += 1

# read from cluster array in order to get parallelizations
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
pTanh = df.loc[idx,'pTanh']
signedStimulus = df.loc[idx,'signedStimulus']


# create dataset with current pTanh transformation
firstSubject = 'ibl_witten_02'
x, y, sessInd, correctSide = get_mouse_design(dfAll, subject=firstSubject, sessStop=None, signedStimulus=signedStimulus, pTanh=pTanh) 
for subject in subjectsAll:
    if (subject != firstSubject):
        xTemp, yTemp, sessIndTemp, correctSideTemp = get_mouse_design(dfAll, subject=subject, sessStop=None, signedStimulus=signedStimulus, pTanh=pTanh) 
        # using all data
        x = np.concatenate((x,xTemp))
        y = np.concatenate((y,yTemp))
        sessInd = sessInd + [i + sessInd[-1] for i in sessIndTemp[1:]]

N = x.shape[0]
K = 1 # classic GLM
D = x.shape[1]
C = 2
maxiter = 250

model_type = 'standard' # fitting standard GLM-HMM

dGLMHMM = dynamic_glmhmm.dynamic_GLMHMM(N,K,D,C)
present = np.ones((N)).astype(int) # using all data
irrelevantSigma = np.ones((K,D))
initP, initpi, initW = dGLMHMM.generate_param(sessInd=sessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)], model_type=model_type) 
standardP, standardpi, standardW, _ = dGLMHMM.fit(x, y,  present, initP=initP, initpi=initpi, initW=initW, sigma=irrelevantSigma, sessInd=sessInd, maxIter=maxiter, tol=1e-4, L2penaltyW=0.5, priorDirP=[10,1], model_type=model_type, fit_init_states=False) # fit the model
_, trainLl, trainAccuracy  = dGLMHMM.evaluate(x, y, sessInd, present, standardP, standardpi, standardW)

np.savez(f'../data_IBL/allAnimals_pTanh={pTanh}_GLM_signedStimulus={signedStimulus}', P=standardP[sessInd[:-1]], pi=standardpi, W=standardW[sessInd[:-1]], trainLl=trainLl, trainAccuracy=trainAccuracy)

