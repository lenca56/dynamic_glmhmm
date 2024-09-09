# importing packages and modules
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
from utils import *
from plotting_utils import *
import dynamic_glmhmm
from scipy.stats import multivariate_normal, norm
import seaborn as sns
from sklearn.model_selection import KFold
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '..', 'LC_PWM_GLM-HMM/code')))
import analysis_utils, plotting_utils
import warnings

def get_mouse_design(dfAll, subject, sessStop=None, signedStimulus=True, pTanh=None):
    ''' 
    function to process IBL data and form design matrix x and output vector y for a given subject 

    p_tanh = 5 used by Roy et al., 2021
    '''
    data = dfAll[dfAll['subject']==subject]   # Restrict data to the subject specified

    # sessions to keep
    if sessStop is not None:
        dateToKeep = np.unique(data['date'])[0:sessStop]
    else: 
        dateToKeep = np.unique(data['date'])
    dataTemp = pd.DataFrame(data.loc[data['date'].isin(list(dateToKeep))])

    # getting correct answer for each trial
    correctSide = np.array(dataTemp['correctSide'])
    
    # tanh transformation
    if pTanh is not None:
        cL = np.tanh(pTanh * data['contrastLeft']) / np.tanh(pTanh) # tanh transformation of left contrasts
        cR = np.tanh(pTanh * data['contrastRight']) / np.tanh(pTanh) # tanh transformation of right contrasts
    else:
        cL = data['contrastLeft']
        cR = data['contrastRight']

    # creating inputs and ouput arrays
    if signedStimulus == True:
        D = 4
        x = np.zeros((dataTemp.shape[0], D)) 
        y = np.array(dataTemp['choice'])

        x[:,0] = 1 # bias or offset is first column
        x[:,1] = cR - cL # 'stimulus contrast'
        x[:,1] = x[:,1] / np.std(x[:,1]) # z-scored   # note that mean should be around 0
        x[1:,2] = 2 * y[0:-1] - 1 # previous chioce as in Zoe's
        x[1:,3] = 2 * np.array(dataTemp['correctSide'])[0:-1] - 1 # previous reward as in Zoe's
    else:
        D = 5
        x = np.zeros((dataTemp.shape[0], D)) 
        y = np.array(dataTemp['choice'])

        x[:,0] = 1 # bias or offset is first column
        x[:,1] = cR # contrast right transformed 
        x[:,2] = cL # contrast left transformed
        # not taking into account first and last of each session 
        x[1:,3] = 2 * y[0:-1] - 1 # previous chioce as in Zoe's {-1,1}
        x[1:,4] = 2 * np.array(dataTemp['correctSide'])[0:-1] - 1 # previous reward as in Zoe's {-1,1}
        
    # session start indices
    sessInd = [0]
    for date in dateToKeep :
        d = dataTemp[dataTemp['date']==date]
        for sess in np.unique(d['session']):
            dTemp = d[d['session'] == sess] 
            dLength = len(dTemp.index.tolist())
            sessInd.append(sessInd[-1] + dLength)
    
    return x, y, sessInd, correctSide


def get_design_biased_blocks(dfAll, subject, sessInd, sessStop=None):
    ''' 
    function that gives biased block trials and sessions from the experimental setup
    '''
    data = dfAll[dfAll['subject']==subject]   # Restrict data to the subject specified
    
    # sessions to keep
    if sessStop is not None:
        dateToKeep = np.unique(data['date'])[0:sessStop]
    else: 
        dateToKeep = np.unique(data['date'])
    dataTemp = pd.DataFrame(data.loc[data['date'].isin(list(dateToKeep))]).reset_index(drop=True)

    pStimLeft = np.array(dataTemp['probabilityLeft'])
    biasedBlockSession = np.zeros((len(sessInd))) # sessions that have biased blocks are marked by 1
    biasedBlockStartInd = np.zeros((sessInd[-1])) # first indices of biased blocks are marked by -1,1 depending on bias
    biasedBlockTrials = np.zeros((sessInd[-1])) # trials within biased blocks are marked by -1,1 depending on bias
    for s in range(0, len(sessInd)-1):
        if (set(np.unique(pStimLeft[sessInd[s]:sessInd[s+1]])) == set([0.2,0.5,0.8])): # biased block sessions
            biasedBlockSession[s] = 1
            for t in range(sessInd[s]+1, sessInd[s+1]): 
                if (pStimLeft[t] != pStimLeft[t-1]): # shift in biased blocks
                    if (pStimLeft[t] == 0.2):
                        biasedBlockStartInd[t] = 1
                        biasedBlockTrials[t] = 1
                    elif (pStimLeft[t] == 0.8):
                        biasedBlockStartInd[t] = -1
                        biasedBlockTrials[t] = -1
                else:
                    biasedBlockTrials[t]=biasedBlockTrials[t-1]
    
    # finding first block session
    blockSessions = [x for [x] in np.argwhere(biasedBlockSession==1)]
    z = 0
    while 1>0:
        if (blockSessions == []):
            firstBlockSession = np.nan
            break
        if (blockSessions[z]+1 in blockSessions): # check for outlier single blocks
            firstBlockSession = blockSessions[z]
            break
        else:
            z += 1
    return biasedBlockTrials, biasedBlockStartInd, biasedBlockSession, firstBlockSession