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

colormap = ['tab:purple','tab:pink','tab:cyan','yellowgreen', 'olive']
colorsStates = ['tab:orange','tab:blue','tab:green','tab:purple', 'tab:brown']
myFeatures = [['bias','stimulus', 'previous choice', 'previous reward'],['bias','contrast left','contrast right', 'previous choice', 'previous reward']]
ibl_data_path = '../data_IBL'
# dfAll = pd.read_csv(ibl_data_path + '/Ibl_processed.csv')
dfAll = pd.read_csv(ibl_data_path + '/IBL_processed_extra.csv')

sessions = [25, 50]
trials = [100,200,400,800]
Nsamples = 20

df = pd.DataFrame(columns=['init','K','signedStimulus', 'pTanh']) # in total z=0,399 inclusively
z = 0
for Nsess in sessions:
    for Ntrial in trials:
        for Nsample in range(Nsamples):
            df.loc[z, 'Nsess'] = Nsess
            df.loc[z, 'Ntrial'] = Ntrial
            df.loc[z, 'Nsample'] = Nsample
            z += 1

# read from cluster array in order to get parallelizations
idx = 0 #int(os.environ["SLURM_ARRAY_TASK_ID"])
Nsess = int(df.loc[idx,'Nsess'])
Ntrial = int(df.loc[idx,'Ntrial'])
Nsample = int(df.loc[idx, 'Nsample'])

K = 3
D = 2
pTanh = 5
signedStimulus = True
sessStop = None

avg_model = np.load(f'../data_IBL/average_animals_fig4-5_best_parameters_dynamic.npz')
dynamicW = avg_model['bestAvgW']
dynamicP = avg_model['bestAvgP']
dynamicpi = np.ones((K))/K
truepi = np.ones((K))/K

subject = 'ibl_witten_15'
x, y, sessInd_old, correctSide, responseTimes = get_mouse_design(dfAll, subject, sessStop=None, signedStimulus=signedStimulus, pTanh=pTanh)
biasedBlockTrials, biasedBlockStartInd, biasedBlockSession, firstBlockSession = get_design_biased_blocks(dfAll, subject, sessInd_old, sessStop)
N = x.shape[0]
sess = len(sessInd_old)-1
x = x[:,:2] # only keeping bias and stimulus

standard_model = np.load(f'../data_IBL/all_animals/Best_allAnimals_standardGLMHMM_{K}-state_pTanh={pTanh}_signedStimulus={signedStimulus}.npz')
standardP = standard_model['P'][:50]
standardW = standard_model['W'][:50,:,:2]

# normalizing transition  matrix
for s in range(50):
    for j in range(K):
        dynamicP[s,j,:] = dynamicP[s,j,:] / dynamicP[s,j,:].sum()
        standardP[s,j,:] = standardP[s,j,:] / standardP[s,j,:].sum()

sigmaList = [10**x for x in list(np.arange(-3,1,0.5,dtype=float))] + [10**x for x in list(np.arange(1,4,1,dtype=float))]
bestSigmaInd = 8 
bestSigma = sigmaList[bestSigmaInd-1]
alphaList = [2*(10**x) for x in list(np.arange(-1,6,0.5,dtype=float))]
bestAlphaInd = 2  # Choosing best sigma index across animals
bestAlpha = alphaList[bestAlphaInd]
maxiter = 1 #250

rng = np.random.default_rng(42)

truex = np.ones((Nsess * Ntrial, 2))
sessInd= [i * Ntrial for i in range(Nsess+1)]
for s in range(Nsess):
    truex[Ntrial * s : Ntrial * s + Ntrial, 1] = rng.choice(x[sessInd_old[s]:sessInd_old[s+1],1], size=Ntrial, replace=True).flatten()
        
N = truex.shape[0]
dGLM_HMM = dynamic_glmhmm.dynamic_GLMHMM(N,K,D,2)
trueW, trueP = reshape_parameters_session_to_trials(dynamicW[:Nsess+1], dynamicP[:Nsess+1], sessInd)
initW, initP = reshape_parameters_session_to_trials(standardW[:Nsess+1], standardP[:Nsess+1], sessInd)

truey, truez = dGLM_HMM.simulate_data_given_x(truex, trueW, trueP, truepi, sessInd, seed=Nsample)

presentAll = np.ones((N))
fitP, _, fitW, trainLl = dGLM_HMM.fit(truex, truey, presentAll, initP, truepi, initW, sigma=reshapeSigma(bestSigma, K, D), alpha=bestAlpha, A=standardP[0], sessInd=sessInd, maxIter=maxiter, tol=1e-3, model_type='dynamic',  L2penaltyW=0, priorDirP = None, fit_init_states=False)
np.savez(f'../simulations/dynamicGLMHMM_simulation_NatComms_sample={Nsample}_Nsess={Nsess}_Ntrial={Ntrial}', P=fitP, allW=fitW, y=truey, z=truez, x=truex, trueW=trueW, trueP=trueP)

