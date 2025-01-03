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

df = pd.DataFrame(columns=['subject','fold','K']) # in total z=0,159 inclusively per fixed state
z = 0
for K in [1,2]:
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
x, y, sessInd, correctSide = get_mouse_design(dfAll, subject=subject, sessStop=None, signedStimulus=signedStimulus, pTanh=pTanh) # with tanh transformation
sess = len(sessInd) - 1

# setting hyperparameters
N = x.shape[0]
D = x.shape[1]
C = 2
presentTrain, presentTest = split_data(N, sessInd, folds=splitFolds, blocks=10, random_state=1)
sigmaList = [10**x for x in list(np.arange(-3,1,0.5,dtype=float))] + [10**x for x in list(np.arange(1,4,1,dtype=float))]
L2penaltyW = 0
priorDirP = [10,1]
maxiter = 300
fit_init_states = False

# initialize model from best fitting parameters of standard GLM-HMM
dataInit = np.load(f'../data_IBL/all_animals/Best_allAnimals_standardGLMHMM_{K}-state_pTanh={pTanh}_signedStimulus={signedStimulus}.npz')
glmhmmP = dataInit['P']
glmhmmpi = dataInit['pi']
glmhmmW = dataInit['W']

# fitting
allP, _, allW, trainLl, testLlSessions, testLl, testAccuracy = fit_eval_CV_partial_model(K, x, y, sessInd, presentTrain[fold], presentTest[fold], sigmaList=sigmaList, maxiter=maxiter, glmhmmW=glmhmmW, glmhmmP=glmhmmP, glmhmmpi=glmhmmpi, L2penaltyW=L2penaltyW, priorDirP=priorDirP, fit_init_states=fit_init_states)

# saving parameters (per-session to optimize memory)
np.savez(f'../data_IBL/{subject}/{subject}_partialGLMHMM_CV_{K}-state_fold={fold}_pTanh={pTanh}_L2penaltyW={L2penaltyW}_signedStimulus={signedStimulus}', allP=allP[:,sessInd[:-1]], allW=allW[:,sessInd[:-1]], trainLl=trainLl, testLl=testLl, testLlSessions=testLlSessions, testAccuracy=testAccuracy)
