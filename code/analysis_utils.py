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
# from pandas.errors import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def split_data(N, sessInd, folds=5, blocks=10, random_state=1):
    ''' 
    splitting data function for cross-validation by giving out indices of test and train
    splits each session into consecutive blocks that randomly go into train and test => each session appears in both train and test

    !Warning: each session must have at least (folds-1) * blocks  trials

    Parameters
    ----------
    x: n x d numpy array
        full design matrix
    y : n x 1 numpy vector 
        full vector of observations with values 0,1,..,C-1
    sessInd: list of int
        indices of each session start, together with last session end + 1
    folds: int
        number of folds to split the data in (test has 1/folds points of whole dataset)
    blocks: int (default = 10)
        blocks of trials to keep together when splitting data (to keep some time dependencies)
    random_state: int (default=1)
        random seed to always get the same split if unchanged

    Returns
    -------
    trainX: list of train_size[i] x d numpy arrays
        trainX[i] has input train data of i-th fold
    trainY: list of train_size[i] numpy arrays
        trainY[i] has output train data of i-th fold
    trainSessInd: list of lists
        trainSessInd[i] have session start indices for the i-th fold of the train data
    testX: // same for test
    '''
    if (sessInd[-1] != N):
        raise Exception("Number of datapoints does not match last sessInd index")
    
    # initializing
    presentTrain = [np.zeros((N)).astype(int) for i in range(0, folds)]
    presentTest = [np.zeros((N)).astype(int) for i in range(0, folds)]

    # split session indices into blocks and get session indices for train and test
    totalSess = len(sessInd) - 1
    for s in range(0, totalSess):
        ySessBlock = np.arange(0, (sessInd[s+1]-sessInd[s])/blocks)
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_state) # shuffle=True and random_state=int for random splitting, otherwise it's consecutive
        for i, (train_blocks, test_blocks) in enumerate(kf.split(ySessBlock)):
            train_indices = []
            test_indices = []
            for b in ySessBlock:
                if (b in train_blocks):
                    train_indices = train_indices + list(np.arange(sessInd[s] + b*blocks, min(sessInd[s] + (b+1) * blocks, sessInd[s+1])))
                elif(b in test_blocks):
                    test_indices = test_indices + list(np.arange(sessInd[s] + b*blocks, min(sessInd[s] + (b+1) * blocks, sessInd[s+1])))
                else:
                    raise Exception("Something wrong with session block splitting")
            
            presentTrain[i][np.array(train_indices).astype(int)] = 1 # part of training set for fold i
            presentTest[i][np.array(test_indices).astype(int)] = 1 # part of test set for fold i

    return presentTrain, presentTest

def fit_eval_CV_partial_model(K, x, y, sessInd, presentTrain, presentTest, sigmaList=[0.01, 0.1, 1, 10, 100], maxiter=300, glmhmmW=None, glmhmmP=None, glmhmmpi=None, L2penaltyW=1, priorDirP = None, fit_init_states=False):
    ''' 
    fitting function for multiple values of sigma with initializing from the previously found parameters with increasing order of fitting sigma
    first sigma is 0 and is the GLM-HMM fit
    each CV fold is fit individually

    Parameters
    ----------
    K: int
        number of latent states
    x: n x d numpy array
        full design matrix
    y : n x 1 numpy vector 
        full vector of observations with values 0,1,..,C-1
    sessInd: list of int
        indices of each session start, together with last session end + 1
    presentTrain: numpy vector
        list of indices in the train set for a single fold
    presentTest: numpy vector
        list of indices in the test set for a single fold
    sigmaList: list of positive numbers, starting with 0 (default =[0, 0.01, 0.1, 1, 10, 100])
        weight drifting hyperparameter list
    maxiter: int 
        maximum number of iterations before EM stopped (default=300)
    glmhmmW: Nsize x K x D x C numpy array 
        given weights from glm-hmm fit (default=None)
    glmhmmP=None: K x K numpy array 
        given transition matrix from glm-hmm fit (default=None)
    L2penaltyW: int
        positive value determinig strength of L2 penalty on weights when fitting (default=1)
    priorDirP : list of length 2
        first number is Dirichlet prior on diagonal, second number is the off-diagonal (default = [10,1])

    Returns
    ----------
    trainLl: list of length fitFolds 
        trainLl[i] is len(sigmaList) x maxiter numpy array of training log like for i'th fold
    testLl: list of length fitFolds 
        testLl[i] is len(sigmaList) numpy vector of normalized test log like for i'th fold
    allP: list of length fitFolds 
        allP[i] if len(sigmaList) x K x K numpy array of fit transition matrix for i'th fold
    allW: list of length fitFolds 
    '''
    N = x.shape[0]
    D = x.shape[1]
    C = 2 # only looking at binomial classes
    sess = len(sessInd) - 1

    trainLl = np.zeros((len(sigmaList)+1, maxiter))
    testLlSessions = np.zeros((len(sigmaList)+1, sess))
    testLl = np.zeros((len(sigmaList)+1))
    testAccuracy = np.zeros((len(sigmaList)+1))
    allP = np.zeros((len(sigmaList)+1, N, K, K))
    allpi = np.zeros((len(sigmaList)+1, K))
    allW = np.zeros((len(sigmaList)+1, N, K, D, C)) 

    dGLM_HMM = dynamic_glmhmm.dynamic_GLMHMM(N,K,D,C)
    
    # first one is standard GLM-HMM (sigma=0)
    if (glmhmmW is not None and glmhmmP is not None): # if parameters are given from standard GLM-HMM (constant across sessions)
        allpi[0] = np.ones((K)) / K
        allP[0, :] = glmhmmP[0]
        allW[0, :] = glmhmmW[0]
    else:
        initP, initpi, initW = dGLM_HMM.generate_param(sessInd=sessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)], model_type='standard') 
        # fit standard GLM-HMM 
        irrelevantSigma = np.ones((K,D))
        allP[0], allpi[0], allW[0], trainLl[0] = dGLM_HMM.fit(x, y, presentTrain, initP, initpi, initW, sigma=irrelevantSigma, sessInd=sessInd, maxIter=maxiter, tol=1e-3, L2penaltyW=L2penaltyW, priorDirP=priorDirP, model_type='standard', fit_init_states=fit_init_states) 
    
    testLlSessions[0], testLl[0], testAccuracy[0] = dGLM_HMM.evaluate(x, y, sessInd, presentTest, allP[0], allpi[0], allW[0])

    for indSigma in range(1,len(sigmaList)+1): 

        # initializing from previous fit
        initP = allP[indSigma-1] 
        initpi = allpi[indSigma-1] 
        initW = allW[indSigma-1] 

        # fitting dGLM-HMM
        allP[indSigma], allpi[indSigma], allW[indSigma], trainLl[indSigma] = dGLM_HMM.fit(x, y, presentTrain, initP, initpi, initW, sigma=reshapeSigma(sigmaList[indSigma-1], K, D), sessInd=sessInd, maxIter=maxiter, tol=1e-3, L2penaltyW=L2penaltyW, priorDirP=priorDirP, model_type='partial', fit_init_states=fit_init_states) 
        
        # evaluate 
        testLlSessions[indSigma], testLl[indSigma], testAccuracy[indSigma] = dGLM_HMM.evaluate(x, y, sessInd, presentTest, allP[indSigma], allpi[indSigma], allW[indSigma])

    return allP, allpi, allW, trainLl, testLlSessions, testLl, testAccuracy

def fit_eval_CV_multiple_alphas(K, x, y, sessInd, presentTrain, presentTest, alphaList=[0, 1, 10, 100, 1000, 10000], maxiter=200, dglmhmmW=None, globalP=None, bestSigma=None, L2penaltyW=1, fit_init_states=False):
    ''' 
    fitting function for multiple values of sigma with initializing from the previously found parameters with increasing order of fitting sigma
    first sigma is 0 and is the GLM-HMM fit
    each CV fold is fit individually

    Parameters
    ----------
    K: int
        number of latent states
    x: n x d numpy array
        full design matrix
    y : n x 1 numpy vector 
        full vector of observations with values 0,1,..,C-1
    sessInd: list of int
        indices of each session start, together with last session end + 1
    presentTrain: numpy vector
        list of indices in the train set for a single fold
    presentTest: numpy vector
        list of indices in the test set for a single fold
    alphaList: list of positive numbers, starting with 0 (default =[0, 0.01, 0.1, 1, 10, 100])
        weight drifting hyperparameter list
    maxiter: int 
        maximum number of iterations before EM stopped (default=300)
    glmhmmW: Nsize x K x D x C numpy array 
        given weights from glm-hmm fit (default=None)
    glmhmmP=None: K x K numpy array 
        given transition matrix from glm-hmm fit (default=None)
    L2penaltyW: int
        positive value determinig strength of L2 penalty on weights when fitting (default=1)
    priorDirP : list of length 2
        first number is Dirichlet prior on diagonal, second number is the off-diagonal (default = [10,1])

    Returns
    ----------
    trainLl: list of length fitFolds 
        trainLl[i] is len(sigmaList) x maxiter numpy array of training log like for i'th fold
    testLl: list of length fitFolds 
        testLl[i] is len(sigmaList) numpy vector of normalized test log like for i'th fold
    allP: list of length fitFolds 
        allP[i] if len(sigmaList) x K x K numpy array of fit transition matrix for i'th fold
    allW: list of length fitFolds 
    '''
    N = x.shape[0]
    D = x.shape[1]
    C = 2 # only looking at binomial classes

    trainLl = np.zeros((len(alphaList)+1, maxiter))
    testLl = np.zeros((len(alphaList)+1))
    testAccuracy = np.zeros((len(alphaList)+1))
    allP = np.zeros((len(alphaList)+1, N, K, K))
    allpi = np.zeros((len(alphaList)+1, K))
    allW = np.zeros((len(alphaList)+1, N,K,D,C)) 

    dGLM_HMM2 = dynamic_glmhmm.dGLM_HMM2(N,K,D,C)

    if (dglmhmmW is None or globalP is None): # fitting dGLM-HMM1 where only weights are varying
        raise Exception("dglmhmmW AND  globalP need to be given from dGLM-HMM1 parameter fitting of best sigma")
    if (bestSigma is None): # fitting dGLM-HMM1 where only weights are varying
        raise Exception("bestSigma need to be given from dGLM-HMM1 fitting of best sigma value")
    
    allP[len(alphaList)] = reshapeP_M1_to_M2(globalP, N)
    allpi[len(alphaList)] = np.ones((K))/K    
    allW[len(alphaList)] = np.copy(dglmhmmW)
    
    # evaluate dGLMHMM1 fit
    testLl[len(alphaList)], testAccuracy[len(alphaList)] = dGLM_HMM2.evaluate(x, y, sessInd, presentTest, allP[len(alphaList)], allpi[len(alphaList)], allW[len(alphaList)], sortStates=False)

    for indAlpha in range(len(alphaList)-1,-1,-1): 

        # initializing from previous fit which means higher alpha
        initP = allP[indAlpha+1] 
        initpi = allpi[indAlpha+1] 
        initW = allW[indAlpha+1] 
            
        # fitting dGLM-HMM
        allP[indAlpha], allpi[indAlpha], allW[indAlpha], trainLl[indAlpha] = dGLM_HMM2.fit(x, y, presentTrain, initP, initpi, initW, sigma=reshapeSigma(bestSigma, K, D), alpha=alphaList[indAlpha], globalP=globalP, sessInd=sessInd, maxIter=maxiter, tol=1e-3, L2penaltyW=L2penaltyW, fit_init_states=fit_init_states) 
   
        # evaluate 
        testLl[indAlpha], testAccuracy[indAlpha] = dGLM_HMM2.evaluate(x, y, sessInd, presentTest, allP[indAlpha], allpi[indAlpha], allW[indAlpha], sortStates=False)

    return allP, allpi, allW, trainLl, testLl, testAccuracy


def fit_eval_CV_2Dsigmas(K, x, y, sessInd, presentTrain, presentTest, sigmaList=[0.01, 0.1, 1, 10, 100], maxiter=300, glmhmmW=None, glmhmmP=None, L2penaltyW=1, priorDirP = [10,1], stimCol=None, fit_init_states=False):
    ''' 
    fitting 2D sigma matrix (one for stimulus and one for all others)
    initialized from best glm-hmm weights and sigmaList[0] =! 0

    Parameters
    ----------
    x: n x d numpy array
        full design matrix
    y : n x 1 numpy vector 
        full vector of observations with values 0,1,..,C-1
    sessInd: list of int
        indices of each session start, together with last session end + 1
    K: int
        number of latent states
    splitFolds: int
        number of folds to split the data in (test has approx 1/folds points of whole dataset)
    fitFolds: int 
        number of folds to actually train on (default=1)
    sigmaList: list of positive numbers, starting with 0 (default =[0, 0.01, 0.1, 1, 10, 100])
        weight drifting hyperparameter list
    maxiter: int 
        maximum number of iterations before EM stopped (default=300)
    glmhmmW: Nsize x K x D x C numpy array 
        given weights from glm-hmm fit (default=None)
    glmhmmP=None: K x K numpy array 
        given transition matrix from glm-hmm fit (default=None)
    L2penaltyW: int
        positive value determinig strength of L2 penalty on weights when fitting (default=1)
    priorDirP : list of length 2
        first number is Dirichlet prior on diagonal, second number is the off-diagonal (default = [10,1])

    Returns
    ----------
    trainLl: list of length fitFolds 
        trainLl[i] is len(sigmaList) x maxiter numpy array of training log like for i'th fold
    testLl: list of length fitFolds 
        testLl[i] is len(sigmaList) numpy vector of normalized test log like for i'th fold
    allP: list of length fitFolds 
        allP[i] if len(sigmaList) x K x K numpy array of fit transition matrix for i'th fold
    allW: list of length fitFolds 
    '''

    N = x.shape[0]
    D = x.shape[1]
    C = 2 # only looking at binomial classes
    oneSessInd = [0,N] # treating whole dataset as one session for normal GLM-HMM fitting

    dGLM_HMM = dglm_hmm1.dGLM_HMM1(N,K,D,C)
    trainLl = np.zeros((len(sigmaList), len(sigmaList), maxiter)) 
    testLl = np.zeros((len(sigmaList), len(sigmaList))) 
    testAccuracy = np.zeros((len(sigmaList), len(sigmaList))) 
    allP = np.zeros((len(sigmaList), len(sigmaList), K, K)) 
    allpi = np.zeros((len(sigmaList), len(sigmaList), K)) 
    allW = np.zeros((len(sigmaList), len(sigmaList), N,K,D,C)) 

    if (sigmaList[0] == 0):
        raise Exception('sigma 0 is given through glm-hmm W and P parameters instead, cant have sigmaList[0]=0')  

    if (stimCol is None):
        raise Exception('stimCol needs to specify which columns in the design matrix have stimuli info')

    if (glmhmmW is not None and glmhmmP is not None): # if parameters are given from standard GLM-HMM 
        oldSessInd = [0, glmhmmW.shape[0]] # assuming glmhmmW has constant weights
        initGlmHmmP = np.copy(glmhmmP) # K x K transition matrix
        initGlmHmmpi = np.ones((K))/K
        initGlmHmmW = reshapeWeights(glmhmmW, oldSessInd, oneSessInd, standardGLMHMM=True)
    else:
        presentAll = np.ones((N)).astype(int) # using all data for sigma=0 fit
        initP0, initpi0, initW0 = dGLM_HMM.generate_param(sessInd=oneSessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)]) 
        initGlmHmmP, initGlmHmmpi, initGlmHmmW, _ = dGLM_HMM.fit(x, y, presentAll, initP0, initpi0, initW0, sigma=reshapeSigma(0, K, D), sessInd=sessInd, maxIter=maxiter, tol=1e-3, L2penaltyW=L2penaltyW, priorDirP=priorDirP, fit_init_states=fit_init_states) 
            
    for indSigma1 in range(0,len(sigmaList)):
        for indSigma2 in range(0,len(sigmaList)): 
            # creating sigma with indSigma1 for delta stim and indSigma 2 for all other features
            sigma2D = [sigmaList[indSigma2] for i in range(0,D)]
            for i in stimCol:
                sigma2D[i] = sigmaList[indSigma1]
            sigma2D = np.array(sigma2D).reshape(1,D)
            sigma2D = reshapeSigma(sigma2D, K, D)

            if (indSigma1 == 0 and indSigma2 == 0): 
                # fitting dGLM-HMM
                allP[indSigma1, indSigma2], allpi[indSigma1, indSigma2], allW[indSigma1, indSigma2], trainLl[indSigma1, indSigma2] = dGLM_HMM.fit(x, y, presentTrain, initGlmHmmP, initGlmHmmpi, initGlmHmmW, sigma=sigma2D, sessInd=sessInd, maxIter=maxiter, tol=1e-3, L2penaltyW=L2penaltyW, priorDirP=priorDirP, fit_init_states=fit_init_states) # fit the model
                                                                                            
            elif (indSigma2 == 0):
                # initializing from previous fit on same indSigma2
                initP = allP[indSigma1-1, indSigma2] 
                initpi = allpi[indSigma1-1, indSigma2] 
                initW = allW[indSigma1-1, indSigma2]
                
                # fitting dGLM-HMM
                allP[indSigma1, indSigma2], allpi[indSigma1, indSigma2], allW[indSigma1, indSigma2], trainLl[indSigma1, indSigma2] = dGLM_HMM.fit(x, y, presentTrain, initP, initpi, initW, sigma=sigma2D, sessInd=sessInd, maxIter=maxiter, tol=1e-3, L2penaltyW=L2penaltyW, priorDirP=priorDirP, fit_init_states=fit_init_states) # fit the model
            else:
                # initializing from previous fit on same indSigma1
                initP = allP[indSigma1, indSigma2-1] 
                initpi = allpi[indSigma1, indSigma2-1]
                initW = allW[indSigma1, indSigma2-1] 
                
                # fitting dGLM-HMM
                allP[indSigma1, indSigma2], allpi[indSigma1, indSigma2], allW[indSigma1, indSigma2], trainLl[indSigma1, indSigma2] = dGLM_HMM.fit(x, y, presentTrain, initP, initpi, initW, sigma=sigma2D, sessInd=sessInd, maxIter=maxiter, tol=1e-3, L2penaltyW=L2penaltyW, priorDirP=priorDirP, fit_init_states=fit_init_states) # fit the model

            # evaluate 
            testLl[indSigma1, indSigma2], testAccuracy[indSigma1, indSigma2] = dGLM_HMM.evaluate(x, y, sessInd, presentTest, allP[indSigma1, indSigma2], allpi[indSigma1, indSigma2], allW[indSigma1, indSigma2], sortStates=False)

    return allP, allpi, allW, trainLl, testLl, testAccuracy

def find_top_init_plot_loglikelihoods(ll, maxdiff, ax=None,startix=5, plot=True):
    '''
    Function from Iris' GLM-HMM github with some alterations
    
    Plot the trajectory of the log-likelihoods for multiple fits, identify how many top fits (nearly) match, and 
    color those trajectories in the plot accordingly

    Parameters
    ----------

    Returns
    ----------

    '''

    # replacing the 0's by nan's
    lls = np.copy(ll)
    lls[lls==0] = np.nan

    # get the final ll for each fit
    final_lls = np.array([np.amax(lls[i,~np.isnan(lls[i,:])]) for i in range(lls.shape[0])])
    
    # get the index of the top ll
    bestInd = np.argmax(final_lls)
    
    # compute the difference between the top ll and all final lls
    ll_diffs = final_lls[bestInd] - final_lls
    
    # identify te fits where the difference from the top ll is less than maxdiff
    top_matching_lls = lls[ll_diffs < maxdiff,:]
    
    # plot
    if (plot == True):
        ax.plot(np.arange(startix,lls.shape[1]),lls.T[startix:], color='black')
        ax.plot(top_matching_lls.T[startix:], color='red')
        ax.set_xlabel('iterations of EM', fontsize=16)
        ax.set_ylabel('log-likelihood', fontsize=16)
    
    return bestInd, final_lls, np.where(ll_diffs < maxdiff)[0] # return indices of best (matching) fits

def IBL_performance(dfAll, subject, plot=False):
    data = dfAll[dfAll['subject']==subject]   # Restrict data to the subject specified

    # keeping first 40 sessions
    dateToKeep = np.unique(data['date'])
    df = pd.DataFrame(data.loc[data['date'].isin(list(dateToKeep))])

    contrastLeft=np.array(df['contrastLeft'])
    contrastRight=np.array(df['contrastRight'])
    correct=np.array(df['feedbackType'])
    dates=np.array(df['date'])

    easy_trials = (contrastLeft > 0.45).astype(int) | (contrastRight > 0.45).astype(int)
    easy_perf = []
    perf = []
    length = []
    for date in np.unique(dates):
        d = df[df['date']==date]
        for sess in np.unique(d['session']):
            session_trials = (np.array(df['session']==sess) * np.array(df['date']==date)).astype(int)
            inds = (session_trials * easy_trials).astype(bool)
            easy_perf += [np.average(correct[inds])]
            perf += [np.average(correct[session_trials.astype(bool)])]
    
    if (plot==True):
        fig, axes = plt.subplots(1, figsize = (13,5), sharex=True, dpi=400) 
        axes.plot(range(1,easy_perf.shape[0]+1), easy_perf, color="black", linewidth=3, label='easy trials') # only look at first 25 days
        axes.plot(range(1,perf.shape[0]+1), perf, color="gray", linewidth=3, label='all trials')
        axes.set_ylabel('task accuracy')
        axes.set_xlabel('session')
        axes.set_yticks([0.4,0.6,0.8,1.0])
        axes.set_ylim(0.2,1.0)
        axes.axhline(0.5, color="black", linestyle="--", lw=1, alpha=0.3, zorder=0)
        axes.set_xlim(0,perf.shape[0]+2)
        axes.spines[['right', 'top']].set_visible(False)
        axes.legend()
    return np.array(easy_perf), np.array(perf)

def accuracy_states_sessions(gamma, phi, y, correctSide, sessInd):
    '''   
    function that probabilistically computes accuracy for each state and overall (no hard assigning)

    P(y_t = correct choice | x_t) = sum over k of p(y_t=correct choice |x_t, z_t=k) * p(z_t=k)

    '''
    K = gamma.shape[1]
    N = y.shape[0]
    p_correct_states = np.zeros((N, K))
    p_correct = np.zeros(N)
    p_correct_states_sessions = np.zeros((len(sessInd)-1, K))
    p_correct_sessions = np.zeros((len(sessInd)-1))
    for session in range(0, len(sessInd)-1):
        for t in range(sessInd[session],sessInd[session+1]):
            for k in range(0,K):
                p_correct_states[t, k] = phi[t, k, correctSide[t]]
            p_correct[t] = p_correct_states[t, :] @ gamma[t]
        p_correct_states_sessions[session] = np.mean(p_correct_states[sessInd[session]:sessInd[session+1]], axis=0)
        p_correct_sessions[session] = np.mean(p_correct[sessInd[session]:sessInd[session+1]], axis=0)
    
    return 100 * p_correct, 100 * p_correct_states, 100 * p_correct_sessions, 100 * p_correct_states_sessions


def soft_occupancy_states_sessions(gamma, sessInd):
    K = gamma.shape[1]
    p_occ_states_sessions = np.zeros((len(sessInd)-1, K))
    for session in range(0, len(sessInd)-1):
        for k in range(0,K):
            p_occ_states_sessions[session, k] = np.mean(gamma[sessInd[session]:sessInd[session+1], k])    
   
    return p_occ_states_sessions


# # OLD SPLITTING DATA FUNCTION
# def old_split_data(x, y, sessInd, folds=4, blocks=10, random_state=1):
#     ''' 
#     function no longer used starting with September 1st 2023
    
#     splitting data function for cross-validation
#     currently does not balance trials for each session

#     !Warning: each session must have at least (folds-1) * blocks  trials

#     Parameters
#     ----------
#     x: n x d numpy array
#         full design matrix
#     y : n x 1 numpy vector 
#         full vector of observations with values 0,1,..,C-1
#     sessInd: list of int
#         indices of each session start, together with last session end + 1
#     folds: int
#         number of folds to split the data in (test has 1/folds points of whole dataset)
#     blocks: int (default = 10)
#         blocks of trials to keep together when splitting data (to keep some time dependencies)
#     random_state: int (default=1)
#         random seed to always get the same split if unchanged

#     Returns
#     -------
#     trainX: list of train_size[i] x d numpy arrays
#         trainX[i] has input train data of i-th fold
#     trainY: list of train_size[i] numpy arrays
#         trainY[i] has output train data of i-th fold
#     trainSessInd: list of lists
#         trainSessInd[i] have session start indices for the i-th fold of the train data
#     testX: // same for test
#     '''
#     N = x.shape[0]
#     D = x.shape[1]
    
#     # initializing session indices for each fold
#     trainSessInd = [[0] for i in range(0, folds)]
#     testSessInd = [[0] for i in range(0, folds)]

#     # split session indices into blocks and get session indices for train and test
#     totalSess = len(sessInd) - 1
#     for s in range(0, totalSess):
#         ySessBlock = np.arange(0, (sessInd[s+1]-sessInd[s])/blocks)
#         kf = KFold(n_splits=folds, shuffle=True, random_state=random_state) # shuffle=True and random_state=int for random splitting, otherwise it's consecutive
#         for i, (train_blocks, test_blocks) in enumerate(kf.split(ySessBlock)):
#             train_indices = []
#             test_indices = []
#             for b in ySessBlock:
#                 if (b in train_blocks):
#                     train_indices = train_indices + list(np.arange(sessInd[s] + b*blocks, min(sessInd[s] + (b+1) * blocks, sessInd[s+1])))
#                 elif(b in test_blocks):
#                     test_indices = test_indices + list(np.arange(sessInd[s] + b*blocks, min(sessInd[s] + (b+1) * blocks, sessInd[s+1])))
#                 else:
#                     raise Exception("Something wrong with session block splitting")

#             trainSessInd[i].append(len(train_indices)+ trainSessInd[i][-1])
#             testSessInd[i].append(len(test_indices) + testSessInd[i][-1])

#     # initializing input and output arrays for each folds
#     trainX = [np.zeros((trainSessInd[i][-1], D)) for i in range(0,folds)]
#     trainY = [np.zeros((trainSessInd[i][-1])).astype(int) for i in range(0,folds)]
#     testX = [np.zeros((testSessInd[i][-1], D)) for i in range(0,folds)]
#     testY = [np.zeros((testSessInd[i][-1])).astype(int) for i in range(0,folds)]

#     # same split as above but now get the actual data split
#     for s in range(0, totalSess):
#         ySessBlock = np.arange(0, (sessInd[s+1]-sessInd[s])/blocks)
#         kf = KFold(n_splits=folds, shuffle=True, random_state=random_state) # shuffle=True and random_state=int for random splitting, otherwise it's consecutive
#         for i, (train_blocks, test_blocks) in enumerate(kf.split(ySessBlock)):
#             train_indices = []
#             test_indices = []
#             for b in ySessBlock:
#                 if (b in train_blocks):
#                     train_indices = train_indices + list(np.arange(sessInd[s] + b*blocks, min(sessInd[s] + (b+1) * blocks, sessInd[s+1])))
#                 elif(b in test_blocks):
#                     test_indices = test_indices + list(np.arange(sessInd[s] + b*blocks, min(sessInd[s] + (b+1) * blocks, sessInd[s+1])))
#                 else:
#                     raise Exception("Something wrong with session block splitting")
#             trainX[i][trainSessInd[i][s]:trainSessInd[i][s+1]] = x[np.array(train_indices).astype(int)]
#             trainY[i][trainSessInd[i][s]:trainSessInd[i][s+1]] = y[np.array(train_indices).astype(int)]
#             testX[i][testSessInd[i][s]:testSessInd[i][s+1]] = x[np.array(test_indices).astype(int)]
#             testY[i][testSessInd[i][s]:testSessInd[i][s+1]] = y[np.array(test_indices).astype(int)]

#     return trainX, trainY, trainSessInd, testX, testY, testSessInd

# OLD FUNCTION FOR FITTING MULTIPLE SIGMAS ON SIMULATED DATA    
# def fit_multiple_sigmas_simulated(N, K, D, C, sessInd, sigmaList=[0.01,0.032,0.1,0.32,1,10,100], inits=1, maxiter=400, modelType='drift', save=False):
#     ''' 
#     fitting function for multiple values of sigma with initializing from the previously found parameters with increasing order of fitting sigma
#     '''

#     dGLM_HMM = dglm_hmm1.dGLM_HMM1(N,K,D,C)
#     simX = np.load(f'../data/N={N}_{K}_state_{modelType}_trainX.npy')
#     simY = np.load(f'../data/N={N}_{K}_state_{modelType}_trainY.npy')

#     allLl = np.zeros((inits, len(sigmaList), maxiter))
#     allP = np.zeros((inits, len(sigmaList), K,K))
#     allpi = np.zeros((inits, len(sigmaList), K))
#     allW = np.zeros((inits, len(sigmaList),N,K,D,C))

#     oneSessInd = [0,N] # treating whole dataset as one session for normal GLM-HMM fitting
 
#     for init in range(0,inits):
#         for indSigma in range(0,len(sigmaList)): 
#             print(indSigma)
#             if (indSigma == 0): 
#                 if(sigmaList[0] == 0):
#                     initP0, initpi0, initW0 = dGLM_HMM.generate_param(sessInd=oneSessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)]) 
#                     allP[init, indSigma], allpi[init, indSigma], allW[init, indSigma], allLl[init, indSigma] = dGLM_HMM.fit(simX, simY,  initP0, initpi0, initW0, sigma=reshapeSigma(1, K, D), sessInd=oneSessInd, maxIter=300, tol=1e-4) # sigma does not matter here
#                 else:
#                     initP, initpi, initW = dGLM_HMM.generate_param(sessInd=sessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)]) # initialize the model parameters
#             else:
#                 initP = allP[init, indSigma-1] 
#                 initW = allW[init, indSigma-1] 
#                 initpi = allpi[init, indSigma-1] 
            
#             if(sigmaList[indSigma] != 0):
#                 # fit on whole dataset
#                 allP[init, indSigma], allpi[init, indSigma], allW[init, indSigma], allLl[init, indSigma] = dGLM_HMM.fit(simX, simY,  initP, initpi, initW, sigma=reshapeSigma(sigmaList[indSigma], K, D), sessInd=sessInd, maxIter=maxiter, tol=1e-3) # fit the model
                
#     if(save==True):
#         np.save(f'../data/Ll_N={N}_{K}_state_{modelType}', allLl)
#         np.save(f'../data/P_N={N}_{K}_state_{modelType}', allP)
#         np.save(f'../data/pi_N={N}_{K}_state_{modelType}', allpi)
#         np.save(f'../data/W_N={N}_{K}_state_{modelType}', allW)

#     return allLl, allP, allpi, allW

# OLDEST SPLITTING DATA FUNCTION

# def split_data_per_session(x, y, sessInd, folds=10, random_state=1):
#     ''' 
#     splitting data function for cross-validation, splitting for each session into folds and then merging
#     currently does not balance number of trials for each session

#     Parameters
#     ----------
#     x: n x d numpy array
#         full design matrix
#         y : n x 1 numpy vector 
#             full vector of observations with values 0,1,..,C-1
#         sessInd: list of int
#             indices of each session start, together with last session end + 1

#         Returns
#         -------
#         trainX: folds x train_size x d numpy array
#             trainX[i] has train data of i-th fold
#         trainY: folds x train_size  numpy array
#             trainY[i] has train data of i-th fold
#         trainSessInd: list of lists
#             trainSessInd[i] have session start indices for the i-th fold of the train data
#         testX: folds x test_size x d numpy array
#             testX[i] has test data of i-th fold
#         testY: folds x test_size  numpy array
#             testY[i] has test data of i-th fold
#         testSessInd: list of lists
#             testSessInd[i] have session start indices for the i-th fold of the test data
#         '''
#     numberSessions = len(sessInd) - 1 # total number of sessions
#     D = x.shape[1]
#     N = x.shape[1]

#     # initializing test and train size based on number of folds
#     train_size = int(N - N/folds)
#     test_size = int(N/folds)

#     # initializing input and output arrays for each folds
#     trainX = [[] for i in range(0,folds)]
#     testX = [[] for i in range(0,folds)]
#     trainY = [[] for i in range(0,folds)]
#     testY = [[] for i in range(0,folds)]
#     # initializing session indices for each fold
#     trainSessInd = [[0] for i in range(0, folds)]
#     testSessInd = [[0] for i in range(0, folds)]

#     # splitting data for each fold for each session
#     for sess in range(0,numberSessions):
#         kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
#         for i, (train_index, test_index) in enumerate(kf.split(y[sessInd[sess]:sessInd[sess+1]])):
#             trainSessInd[i].append(trainSessInd[i][-1] + len(train_index))
#             testSessInd[i].append(testSessInd[i][-1] + len(test_index))
#             trainY[i].append(y[sessInd[sess] + train_index])
#             testY[i].append(y[sessInd[sess] + test_index])
#             trainX[i].append(x[sessInd[sess] + train_index])
#             testX[i].append(x[sessInd[sess] + test_index])
    
#     array_trainX =  [np.zeros((trainSessInd[i][-1],D)) for i in range(0,folds)]
#     array_testX = [np.zeros((testSessInd[i][-1],D)) for i in range(0,folds)]
#     array_trainY = [np.zeros((trainSessInd[i][-1])) for i in range(0,folds)]
#     array_testY = [np.zeros((testSessInd[i][-1])) for i in range(0,folds)]

#     for sess in range(0,numberSessions):
#         for i in range(0,folds):
#             array_trainX[i][trainSessInd[i][sess]:trainSessInd[i][sess+1],:] = trainX[i][sess]
#             array_testX[i][testSessInd[i][sess]:testSessInd[i][sess+1],:] = testX[i][sess]
#             array_trainY[i][trainSessInd[i][sess]:trainSessInd[i][sess+1]] = trainY[i][sess]
#             array_testY[i][testSessInd[i][sess]:testSessInd[i][sess+1]] = testY[i][sess]
            
#     return array_trainX, array_trainY, trainSessInd, array_testX, array_testY, testSessInd

# OLD FUNCTION - REPLACED BY ABOVE ONE

# def fit_eval_CV_multiple_sigmas_PWM(rat_id, stage_filter, K, folds=3, sigmaList=[0, 0.01, 0.1, 1, 10, 100], maxiter=300, glmhmmW=None, glmhmmP=None, L2penaltyW=1, path=None, save=False):
#     ''' 
#     fitting function for multiple values of sigma with initializing from the previously found parameters with increasing order of fitting sigma
#     first sigma is 0 and is the GLM-HMM fit
#     only suited for PWM data for now
#     '''
#     x, y = io_utils.prepare_design_matrices(rat_id=rat_id, path=path, psychometric=True, cutoff=10, stage_filter=stage_filter, overwrite=False)
#     sessInd = list(io_utils.session_start(rat_id=rat_id, path=path, psychometric=True, cutoff=10, stage_filter=stage_filter)) 
#     trainX, trainY, trainSessInd, testX, testY, testSessInd = split_data_per_session(x, y, sessInd, folds=folds, random_state=1)
#     D = trainX[0].shape[1]
#     C = 2 # only looking at binomial classes

#     trainLl = [np.zeros((len(sigmaList), maxiter)) for i in range(0,folds)] 
#     testLl = [np.zeros((len(sigmaList))) for i in range(0,folds)]
#     allP = [np.zeros((len(sigmaList), K,K)) for i in range(0,folds)] 
#     allW = [] 

#     for fold in [0]: # fitting single fold     # fittinng all folds -> range(0,folds): 
#         # initializing parameters for each fold
#         N = trainX[fold].shape[0]
#         oneSessInd = [0,N] # treating whole dataset as one session for normal GLM-HMM fitting
#         dGLM_HMM = dglm_hmm1.dGLM_HMM1(N,K,D,C)
#         allW.append(np.zeros((len(sigmaList), N,K,D,C)))
#         trainY[fold] = trainY[fold].astype(int)
#         testY[fold] = testY[fold].astype(int)

#         for indSigma in range(0,len(sigmaList)): 
#             print("Sigma Index " + str(indSigma))
#             if (indSigma == 0): 
#                 if(sigmaList[0] == 0):
#                     if (glmhmmW is not None and glmhmmP is not None):
#                         # best found glmhmm with multiple initializations - constant P and W
#                         print("GLM HMM BEST INIT")
#                         oldSessInd = [0, glmhmmW.shape[0]]
#                         initP0 = np.copy(glmhmmP) # K x K transition matrix
#                         initW0 = reshapeWeights(glmhmmW, oldSessInd, oneSessInd, standardGLMHMM=True)
#                     else:
#                         initP0, initW0 = dGLM_HMM.generate_param(sessInd=oneSessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)]) 
#                     # fitting for sigma = 0
#                     allP[fold][indSigma],  allW[fold][indSigma], trainLl[fold][indSigma] = dGLM_HMM.fit(trainX[fold], trainY[fold],  initP0, initW0, sigma=reshapeSigma(1, K, D), sessInd=oneSessInd, pi0=None, maxIter=maxiter, tol=1e-3, L2penaltyW=L2penaltyW) # sigma does not matter here
#                 else:
#                     raise Exception("First sigma of sigmaList should be 0, meaning standard GLM-HMM")
#             else:
#                 # initializing from previoous fit
#                 initP = allP[fold][indSigma-1] 
#                 initW = allW[fold][indSigma-1] 
            
#                 # fitting dGLM-HMM
#                 allP[fold][indSigma],  allW[fold][indSigma], trainLl[fold][indSigma] = dGLM_HMM.fit(trainX[fold], trainY[fold],  initP, initW, sigma=reshapeSigma(sigmaList[indSigma], K, D), sessInd=trainSessInd[fold], pi0=None, maxIter=maxiter, tol=1e-3, L2penaltyW=L2penaltyW) # fit the model
        
#             # evaluate
#             sess = len(trainSessInd[fold]) - 1 # number sessions
#             testPhi = dGLM_HMM.observation_probability(testX[fold], reshapeWeights(allW[fold][indSigma], trainSessInd[fold], testSessInd[fold]))
#             for s in range(0, sess):
#                 # evaluate on test data for each session separately
#                 _, _, temp = dGLM_HMM.forward_pass(testY[fold][testSessInd[fold][s]:testSessInd[fold][s+1]],allP[fold][indSigma],testPhi[testSessInd[fold][s]:testSessInd[fold][s+1]])
#                 testLl[fold][indSigma] += temp
    
#         testLl[fold] = testLl[fold] / testSessInd[fold][-1] # normalizing to the total number of trials in test dataset

#         if(save==True):
#             np.save(f'../data_PWM/trainLl_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}', trainLl[fold])
#             np.save(f'../data_PWM/testLl_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}', testLl[fold])
#             np.save(f'../data_PWM/P_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}', allP[fold])
#             np.save(f'../data_PWM/W_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}', allW[fold])
#             np.save(f'../data_PWM/trainSessInd_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}', np.array(trainSessInd[fold]))
#             np.save(f'../data_PWM/testSessInd_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas_L2penaltyW={L2penaltyW}', np.array(testSessInd[fold]))

#     return trainLl, testLl, allP, allW

