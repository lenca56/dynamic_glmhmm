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
dfAll = pd.read_csv(ibl_data_path + '/IBL_processed_extra.csv')
splitFolds = 5

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

df = pd.DataFrame(columns=['subject','fold','K']) 
z = 0
for K in [3]:#[2,3,4]:
    for subject in subjectsAll:
            for fold in range(splitFolds):
                df.loc[z, 'subject'] = subject
                df.loc[z, 'K'] = K
                df.loc[z, 'fold'] = fold
                z += 1 

# read from cluster array in order to get parallelizations
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
subject = df.loc[idx,'subject']
K = df.loc[idx,'K']
fold = df.loc[idx,'fold']

# whether to have left and ride stimuli within one variable or two
signedStimulus = True

# load data for particular animal
pTanh = 5
x, y, sessInd, correctSide, responseTimes = get_mouse_design(dfAll, subject=subject, sessStop=None, signedStimulus=signedStimulus, pTanh=pTanh) # with tanh transformation
sess = len(sessInd) - 1

# setting hyperparameters
N = x.shape[0]
D = x.shape[1]
C = 2
presentTrain, presentTest = split_data(N, sessInd, folds=splitFolds, blocks=10, random_state=1)
alphaList = [0]+[2*(10**x) for x in list(np.arange(-1,6,1,dtype=float))]
L2penaltyW = 0
maxiter = 200
fit_init_states = False

# initialize model from best fitting parameters of standard GLM-HMM, checked in Figure4-5.ipynb
sigmaList = [10**x for x in list(np.arange(-3,1,0.5,dtype=float))] + [10**x for x in list(np.arange(1,4,1,dtype=float))]
bestSigmaInd = 8 
bestSigma = sigmaList[bestSigmaInd-1]

dataInit = data = np.load(f'../data_IBL/{subject}/{subject}_partialGLMHMM_CV_{K}-state_fold={fold}_pTanh={pTanh}_L2penaltyW={L2penaltyW}_signedStimulus={signedStimulus}.npz')
globalP = dataInit['allP'][bestSigmaInd,0]

data_dynamic = np.load(f'../data_IBL/{subject}/{subject}_dynamicGLMHMM_CV_{K}-state_fold={fold}_pTanh={pTanh}_L2penaltyW={L2penaltyW}_signedStimulus={signedStimulus}.npz')
bestAlphaInd = 2
truepi = np.ones((K))/K

W_dynamic_best, P_dynamic_best = reshape_parameters_session_to_trials(data_dynamic['allW'][bestAlphaInd], data_dynamic['allP'][bestAlphaInd], sessInd)

dGLM_HMM = dynamic_glmhmm.dynamic_GLMHMM(N,K,D,C)
P, pi, W, trainLl = dGLM_HMM.fit(x, y, presentTrain[fold], P_dynamic_best, truepi, W_dynamic_best, sigma=reshapeSigma(bestSigma, K, D), alpha=0, A=globalP, sessInd=sessInd, maxIter=maxiter, tol=1e-3, model_type='dynamic',  L2penaltyW=L2penaltyW, priorDirP = None, fit_init_states=fit_init_states)
# evaluate 
testLlSessions, testLl, testAccuracy = dGLM_HMM.evaluate(x, y, sessInd, presentTest[fold], P, pi, W)

# saving parameters (per-session to optimize memory)
np.savez(f'../data_IBL/{subject}/{subject}_dynamicGLMHMM_no-prior-P_CV_{K}-state_fold={fold}_pTanh={pTanh}_L2penaltyW={L2penaltyW}_signedStimulus={signedStimulus}', P=P[sessInd[:-1]], W=W[sessInd[:-1]], trainLl=trainLl, testLl=testLl, testLlSessions=testLlSessions, testAccuracy=testAccuracy)
