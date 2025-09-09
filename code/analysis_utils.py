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
import analysis_utils, plotting_utils
import warnings

def split_data(N, sessInd, folds=5, blocks=10, random_state=1):
    ''' 
    splitting data function for cross-validation by returning indices of test and train sets

    splits each session into consecutive blocks that randomly go into train and test => each session has trials in both train and test sets

    ! Warning: each session must have at least (folds-1) * blocks trials

    Parameters
    ----------
    N: int
        total number of trials
    sessInd: list of int
        indices of each session start, together with last session end + 1
    folds: int (default = 5)
        number of folds to split the data in (test has 1/folds points of whole dataset)
    blocks: int (default = 10)
        number of consecutive trials to keep together when splitting data (to keep some time dependencies)
    random_state: int (default=1)
        random seed to always get the same split if unchanged

    Returns
    -------
    presentTrain: list of N numpy vectors (length is given by number of folds)
        each vector corresponding to a different fold has 1's for indices of trials in train set and 0's otherwise
    presentTest: list of N numpy vectors (length is given by number of folds)
        each vector corresponding to a different fold has 1's for indices of trials in test set and 0's otherwise
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

def fit_eval_CV_partial_model(K, x, y, sessInd, presentTrain, presentTest, sigmaList=[0.01, 0.1, 1, 10, 100], maxiter=300, glmhmmW=None, glmhmmP=None, glmhmmpi=None, L2penaltyW=0, priorDirP = None, fit_init_states=False):
    ''' 
    fitting function for the "partial" models (time-varying weights and static transition matrix), 
    in increasing order of hyperparameter sigma that governs weight variability, where each models is initialized
    with the best parameters obtained from the previously fit model with a slightly smaller sigma

    first sigma is 0 which gives the standard GLM-HMM fit

    each CV fold is fit individually

    Parameters
    ----------
    K: int
        number of latent states
    x: N x D numpy array
        full design matrix
    y : N x 1 numpy vector 
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
        positive value determinig strength of L2 penalty on weights when fitting (default=0)
    priorDirP : list of length 2
        first number is Dirichlet prior on diagonal, second number is the off-diagonal (default = [10,1])
    fit_init_states: Boolean (default=False)
        whether to fit or not distribution of first latent in every session
        if not, it's assumed uniform

    Returns
    ----------
    allP: len(sigmaList)+1 x N x K x K numpy array
        allP[i] contains transition matrix across trials for i'th value of sigma in sigmaList
    allW: len(sigmaList)+1 x N x K x D x C numpy array
        allW[i] contains weight matrix across trials for i'th value of sigma in sigmaList
    trainLl: len(sigmaList)+1 x maxiter numpy array
        trainLl[i] contains total train set log-likelihood at every iteration for i'th value of sigma in sigmaList
    testLlSessions: len(sigmaList)+1 x sess numpy array
        testLlSessions[i] contains per-trial average test set log-likelihood for each session for i'th value of sigma in sigmaList
    testLl: len(sigmaList)+1 numpy vector
        testLl[i] contains per-trial average test set log-likelihood across sessions for i'th value of sigma in sigmaList
    testAccuracy: len(sigmaList)+1 numpy vector
        testAccuracy:[i] contains % correct predicted choies when hard assigning latent state and output for i'th value of sigma in sigmaList

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
        if glmhmmpi is not None:
                allpi[0] = np.copy(glmhmmpi)
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

def fit_eval_CV_dynamic_model(K, x, y, sessInd, presentTrain, presentTest, alphaList=[0, 1, 10, 100, 1000, 10000], maxiter=200, partial_glmhmmW=None, globalP=None, partial_glmhmmpi=None, bestSigma=None, L2penaltyW=0, fit_init_states=False):
    ''' 
    fitting function for the "dynamic" models (time-varying weights and time-varying transition matrix), 
    in decreasing order of hyperparameter alpha that governs transition matrix variability, where each models is initialized
    with the best parameters obtained from the previously fit model with a slightly larger alpha

    a very large alpha is equivalent to the parial GLM-HMM (static transition matrix)

    each CV fold is fit individually

    Parameters
    ----------
    K: int
        number of latent states
    x: N x D numpy array
        full design matrix
    y : N x 1 numpy vector 
        full vector of observations with values 0,1,..,C-1
    sessInd: list of int
        indices of each session start, together with last session end + 1
    presentTrain: numpy vector
        list of indices in the train set for a single fold
    presentTest: numpy vector
        list of indices in the test set for a single fold
    alphaList: list of positive numbers, starting with 0 (default = [0, 1, 10, 100, 1000, 10000])
        list of different hyperparameters alpha governing the rate of change of transition matrix
    maxiter: int 
        maximum number of iterations before EM stopped (default=300)
    partial_glmhmmW: Nsize x K x D x C numpy array 
        given weights from partial glm-hmm fit (default=None)
    globalP=None: K x K numpy array 
        fitted static transition matrix from standard/partial glm-hmm fit (default=None)
    bestSigma: float
        value used for sigma, rate of change of weights across sessions, found by CV of "partial" model with above function
    L2penaltyW: int
        positive value determinig strength of L2 penalty on weights when fitting (default=0)
    fit_init_states: Boolean (default=False)
        whether to fit or not distribution of first latent in every session
        if not, it's assumed uniform

    Returns
    ----------
    allP: len(alphaList)+1 x N x K x K numpy array
        allP[i] contains transition matrix across trials for i'th value of alpha in alphaList
    allW: len(alphaList)+1 x N x K x D x C numpy array
        allW[i] contains weight matrix across trials for i'th value of alpha in alphaList
    trainLl: len(alphaList)+1 x maxiter numpy array
        trainLl[i] contains total train set log-likelihood at every iteration for i'th value of alpha in alphaList
    testLlSessions: len(alphaList)+1 x sess numpy array
        testLlSessions[i] contains per-trial average test set log-likelihood for each session for i'th value of alpha in alphaList
    testLl: len(alphaList)+1 numpy vector
        testLl[i] contains per-trial average test set log-likelihood across sessions for i'th value of alpha in alphaList
    testAccuracy: len(alphaList)+1 numpy vector
        testAccuracy:[i] contains % correct predicted choies when hard assigning latent state and output for i'th value of alpha in alphaList

    '''
    N = x.shape[0]
    D = x.shape[1]
    C = 2 # only looking at binomial classes
    sess = len(sessInd) - 1

    trainLl = np.zeros((len(alphaList)+1, maxiter))
    testLl = np.zeros((len(alphaList)+1))
    testLlSessions = np.zeros((len(alphaList)+1, sess))
    testAccuracy = np.zeros((len(alphaList)+1))
    allP = np.zeros((len(alphaList)+1, N, K, K))
    allpi = np.zeros((len(alphaList)+1, K))
    allW = np.zeros((len(alphaList)+1, N,K,D,C)) 

    dGLM_HMM = dynamic_glmhmm.dynamic_GLMHMM(N,K,D,C)

    if (partial_glmhmmW is None or globalP is None): # fitting partial dynamic GLM-HMM, in which only weights are varying
        raise Exception("partial_glmhmmW AND globalP need to be given from partial dyanmic GLM-HMM parameter fitting of best sigma")
    if (bestSigma is None): # fitting dGLM-HMM1 where only weights are varying
        raise Exception("bestSigma need to be given from partial dyanmic GLM-HMM parameter fitting of best sigma")
    
    # model equivalent to alpha -> infinity
    allP[len(alphaList)] = reshapeP_M1_to_M2(globalP, N) 
    allW[len(alphaList)] = np.copy(partial_glmhmmW)

    if fit_init_states == False:
        allpi[len(alphaList)] = np.ones((K))/K 
    else:
        allpi[len(alphaList)] = partial_glmhmmpi
    
    # evaluate dGLMHMM1 fit
    testLlSessions[len(alphaList)], testLl[len(alphaList)], testAccuracy[len(alphaList)] = dGLM_HMM.evaluate(x, y, sessInd, presentTest, allP[len(alphaList)], allpi[len(alphaList)], allW[len(alphaList)])
    
    for indAlpha in range(len(alphaList)-1,-1,-1): 

        # initializing from previous fit which means higher alpha
        initP = allP[indAlpha+1] 
        initpi = allpi[indAlpha+1] 
        initW = allW[indAlpha+1] 
            
        # fitting dGLM-HMM
        allP[indAlpha], allpi[indAlpha], allW[indAlpha], trainLl[indAlpha] = dGLM_HMM.fit(x, y, presentTrain, initP, initpi, initW, sigma=reshapeSigma(bestSigma, K, D), alpha=alphaList[indAlpha], A=globalP, sessInd=sessInd, maxIter=maxiter, tol=1e-3, model_type='dynamic',  L2penaltyW=L2penaltyW, priorDirP = None, fit_init_states=fit_init_states)
   
        # evaluate 
        testLlSessions[indAlpha], testLl[indAlpha], testAccuracy[indAlpha] = dGLM_HMM.evaluate(x, y, sessInd, presentTest, allP[indAlpha], allpi[indAlpha], allW[indAlpha])

    return allP, allpi, allW, trainLl, testLlSessions, testLl, testAccuracy

def fit_eval_CV_2Dsigmas(K, x, y, sessInd, presentTrain, presentTest, sigmaList=[0.01, 0.1, 1, 10, 100], maxiter=300, glmhmmW=None, glmhmmP=None, L2penaltyW=1, priorDirP = [10,1], stimCol=None, fit_init_states=False):
    ''' 
    fitting 2D sigma matrix (one for stimulus and one for all others)
    initialized from best glm-hmm weights

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

def accuracy_states_sessions(gamma, phi, y, correctSide, sessInd):
    '''   
    function that probabilistically computes accuracy for each state and overall accuracy
    (no hard assigning of hidden states)

    P(y_t = correct choice | x_t) = sum over k of p(y_t=correct choice |x_t, z_t=k) * p(z_t=k)

    Parameters
    ----------
    gamma: N x k numpy array
        matrix of marginal posterior of latents given all observations p(z_t | y_1:T)
        = likelihood of being in each state given the data
    phi : N x k x c numppy array
        matrix of observation probabilities
    y : N x 1 numpy vector 
        vector of observations with values 0,1,..,C-1
    correctSide: T x 1 numpy vector
        vector of correct side for each trial
    sessInd: list of int
        indices of each session start, together with last session end + 1

    Returns
    ----------
    p_correct:  N x 1 numpy vector
        probability of correct choice for each trial
    p_correct_states: N x K numpy array
        probability of correct choice for each trial given a state
    p_correct_sessions: len(sessInd)-1 x 1 numpy vector
        mean fraction correct choices for each session
    p_correct_states_sessions: len(sessInd)-1 x K numpy array
        mean fraction correct choices for each session given a state
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
    ''' 
    function that probabilistically computes % occupancy in each state
    (no hard assigning of hidden states)

    Parameters
    ----------
    gamma: N x k numpy array
        matrix of marginal posterior of latents given all observations p(z_t | y_1:T)
        = likelihood of being in each state given the data
    sessInd: list of int
        indices of each session start, together with last session end + 1

    Returns
    ----------
    p_occ_states_sessions: len(sessInd)-1 x K numpy array
        = mean likelihood of trials spent in each state within a session
    '''
    
    K = gamma.shape[1]
    p_occ_states_sessions = np.zeros((len(sessInd)-1, K))
    for session in range(0, len(sessInd)-1):
        for k in range(0,K):
            p_occ_states_sessions[session, k] = np.mean(gamma[sessInd[session]:sessInd[session+1], k])    
   
    return p_occ_states_sessions
