# importing packages and modules
import pandas as pd 
import numpy as np
from pathlib import Path
import math
import matplotlib.pyplot as plt
import scipy.stats as stats

def reshapeObs(y):
    '''
    reshaping observation y from having C columns with values among 0 and 1 to having 1 column with values from 0 to C-1

    Parameters
    ----------
    y: T x C numpy array
        output / observation array

    Returns
    -------
    yNew: T numpy vector
        output / observation array
    '''
    
    yNew = np.empty((y.shape[0],))
    if(y.shape[1] > 1):
        yNew = np.array(np.where(y==1)[1]).reshape(y.shape[0],)
    return yNew

def reshapeSigma(sigma, K, D):
    ''' 
    changing variance parameter sigma to have shape K x D

    Parameters
    ----------
    sigma: nonnegative floats
        either scalar, Kx1, 1xD, or KxD numpy array
    K: int
        number of states
    D: int
        number of input features / task covariates

    Returns
    -------
    newSigma: KxD numpy array
        reshaped sigma matrix
    '''
    
    newSigma = np.empty((K,D))
    if (isinstance(sigma, float)  == True) or isinstance(sigma, int)  == True:
        newSigma.fill(sigma)
    elif (sigma.shape[0]==1 and sigma.shape[1]==D):
        newSigma = np.repeat(sigma, repeats = K, axis=0)
    elif (sigma.shape[0]==K and sigma.shape[1]==1):
        newSigma = np.repeat(sigma, repeats = D, axis=1)
    elif (sigma.shape[0]==K and sigma.shape[1]==D):
        newSigma = np.copy(sigma)
    else:
        raise Exception('sigma can only be scalar, Kx1, 1xD, or KxD numpy array')
    
    # check that it's all nonnegative elements
    if np.any(newSigma < 0):
        raise Exception('sigma can not have any negative elements')

    return newSigma

def reshape_parameters_session_to_trials(w, p, sessInd):
    ''' 
    reshaping weight and transition matrices from per-session to per-trial 
    (parameters are constant within session)

    Parameters
    ----------
    w: S x K x D x C numpy array
        weight matrix across S sessions
    p: S x K x K numpy array
        transition matrix across S sessions

    Returns
    -------
    w_new: T x K x D x C numpy array
        weight matrix across T trials spanning S sessions
    p_new: T x K x K numpy array
        transition matrix across T trials spanning S sessions
    
    '''

    w_new = np.zeros((sessInd[-1],w.shape[1],w.shape[2], w.shape[3]))
    p_new = np.zeros((sessInd[-1],p.shape[1],p.shape[2]))
    for sess in range(len(sessInd)-1):
        w_new[sessInd[sess]:sessInd[sess+1]] = w[sess]
        p_new[sessInd[sess]:sessInd[sess+1]] = p[sess]
    
    return w_new, p_new

def reshapeWeights(w, oldSessInd, newSessInd, standardGLMHMM=False):
    ''' 
    reshaping weights from session indices of oldSessInd to session indices of newSessInd

    Parameters
    ----------
    w: T x k x d x c numpy array
        weight matrix. for c=2, trueW[:,:,:,1] = 0 
    oldSessInd: list of int
        old indices of each session start, together with last session end + 1
    newSessInd: list of int
        new indices of each session start, together with last session end + 1
            
    Returns
    -------
    reshapedW: newT x k x d x c
        reshaped weight matrix
    '''
    T = w.shape[0]
    k = w.shape[1]
    d = w.shape[2]
    c = w.shape[3]
    
    if (T != oldSessInd[-1]):
        raise Exception ("Indices and weights do not match in size")
    if (standardGLMHMM == True):
        newT = newSessInd[-1]
        reshapedW = np.zeros((newT, k, d, c))
        reshapedW[:,:,:,1] = w[0,:,:,1]
    else:
        if (len(oldSessInd) != len(newSessInd) and standardGLMHMM==False):
            raise Exception ("old and new indices don't have the same number of sessions")
        
        newT = newSessInd[-1]
        reshapedW = np.zeros((newT, k, d, c))
        for sess in range(0,len(oldSessInd)-1):
            reshapedW[newSessInd[sess]:newSessInd[sess+1],:,:,1] = w[oldSessInd[sess],:,:,1]
        
    return reshapedW

def reshapeTransitionMatrix(p, oldSessInd, newSessInd):
    ''' 
    reshaping weights from session indices of oldSessInd to session indices of newSessInd

    Parameters
    ----------
    P: T x k x k numpy array
        transition matrix 
    oldSessInd: list of int
        old indices of each session start, together with last session end + 1
    newSessInd: list of int
        new indices of each session start, together with last session end + 1
            
    Returns
    -------
    reshapedP: newT x k x k
        reshaped transition matrix
    '''
    
    T = p.shape[0]
    k = p.shape[1]
    if (T != oldSessInd[-1]):
        raise Exception ("Indices and weights do not match in size")
    if (len(oldSessInd) != len(newSessInd)):
        raise Exception ("old and new indices don't have the same number of sessions")
        
    newT = newSessInd[-1]
    reshapedP = np.zeros((newT, k, k))
    for sess in range(0,len(oldSessInd)-1):
        reshapedP[newSessInd[sess]:newSessInd[sess+1]] = p[oldSessInd[sess]]
        
    return reshapedP

def get_states_order(w, sessInd, stimCol=[1]):
    ''' 
    returning states in decreasing order according to absolute value of sensory weights across consecutive sessions

    Parameters
    -------
    w: T x k x d x c numpy array
        weight matrix. for c=2, trueW[:,:,:,1] = 0 
    sessInd: list of int
        old indices of each session start, together with last session end + 1
    stimCol: list of int
        index of stimulus column to be examined for ordering

    Returns
    -------
    sortedInd: list of length k
        permutation of [0,1,..,k-1] in order described above
    '''

    k = w.shape[1]
    D = w.shape[2]
    sess = len(sessInd)-1
    driftState = np.zeros((k,))
    for s in range(0,sess):
        for i in range(0,k):
            for x in stimCol:
                driftState[i]+= abs(w[sessInd[s],i,x,1]) 
    sortedInd = list(np.argsort(driftState))
    sortedInd.reverse() # decreasing order
    
    return sortedInd

def softplus(x):
    '''   
    Softplus function computes  f(x) = log (1 + exp(x))

    Used for calculating log of observatin probabilities as -f(-x) = - log (1 + exp(-x)) = log( 1 / (1 + exp(-x))) 
    and - f(x) = log( 1 / (1 + exp(x))) = log( exp(-x) / (1 + exp(-x)))
    '''
    
    f = np.zeros((x.shape[0]))
    indUnder = (x < -20) # underflow
    indOver = (x > 200) # overflow
    indMiddle = (x >= - 20) & (x <= 200)

    # approx from Taylor expansion
    f[indUnder] = np.exp(x[indUnder])
    f[indOver] = x[indOver]
    f[indMiddle] = np.log(1 + np.exp(x[indMiddle]))

    return f

def softplus_deriv(x):
    '''
    Derivative of softplus function

    d/dx log (1 + exp(x)) = exp(x) / (1 + exp(x))
    
    '''

    # avoiding overflow by separating in two different cases
    if (x > 0):
        return 1/(1+math.exp(-x))
    else:
        return math.exp(x)/(1+math.exp(x))
    
def reshapeP_M1_to_M2(P, N):
    '''
    function reshaping transition matrix of shape (K,K) to shape (N,K,K) 
    
    '''
    return np.repeat(P[np.newaxis,...], N, axis=0)
