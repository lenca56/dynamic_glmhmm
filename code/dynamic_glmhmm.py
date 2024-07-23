# importing packages and modules
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from utils import *
from plotting_utils import *
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold
# from autograd import value_and_grad, hessian
# from jax import value_and_grad
# import jax.numpy as jnp

class dynamic_GLMHMM():
    """
    Class for fitting drifting GLM-HMM model (in which both weights and transition matrix can vary across sessions)
    Code just works for c=2 at the moment !
    Weights for class c=0 are always kept to 0 

    Notation: 
        N: number of data points
        K: number of states (states)
        D: number of features (inputs to design matrix)
        C: number of classes (possible observations)
        x: design matrix (n x d)
        y: observations (n x c) or (n x 1)
        w: weights mapping x to y (n x k x d x c)
    """

    def __init__(self, N, K, D, C):
            self.N, self.K, self.D, self.C  = N, K, D, C
    
    # Iris' fct has columns reverrsed
    def log_observation_probability(self, x, w):
        '''
        Calculating log of observation probabilities for part/all of design matrix x and weight matrix w
        C = 2 only

        Parameters
        ----------
        x: T x D numpy array
            input matrix
        w: T x K x D x C numpy array
            weight matrix

        Returns
        -------
        logphi: T x K x C numpy array
            log of observation probabilities matrix (already normalized inside log)
        '''
        T = x.shape[0] # not necessarily the full dataset

        # check dimensionality of weights and reshape if necessary
        if (w.ndim == 3): # it means K=1 state
            w = w.reshape((w.shape[0],1,w.shape[1],w.shape[2])) # reshape
            K = 1
        elif (w.ndim == 4): # K>=2
            K = w.shape[1] # number of states
        else:
            raise Exception("Weight matrix should have 3 or 4 dimensions (N X D x C or N x K x D x C)")

        logphi = np.zeros((T, K, self.C)) 
        for k in range(0, K): 
            logphi[:,k,1] = - softplus(-np.sum(w[:,k,:,1]*x,axis=1)) # p(y_t=1) = 1 / (1 + exp (-xw))
            logphi[:,k,0] = - softplus(np.sum(w[:,k,:,1]*x,axis=1)) # p(y_t=0) = exp (-xw) / (1 + exp (-xw))
            # logphi[:,k,1] = - softplus(np.sum(w[:,k,:,1]*x,axis=1)) # reversed (as in old version)
            # logphi[:,k,0] = - softplus(-np.sum(w[:,k,:,1]*x,axis=1)) # 
            
        return logphi

    def observation_probability(self, x, w):
        ''' 
        function that calculates observation probabilities by taking exp of log observation prob

        Parameters
        ----------
        x: Ncurrent x D numpy array
            input matrix
        w: Ncurrent x K x D x C numpy array
            weight matrix

        Returns
        -------
        phi: Ncurrent x K x C numpy array
            observation probabilities matrix (already normalized inside log)
        '''

        phi = self.log_observation_probability(x, w)
        phi = np.exp(phi)

        return phi  

    def simulate_data(self, trueW, trueP, truepi, sessInd):
        '''
        function that simulates X and Y data from true weights and true transition matrix

        Parameters
        ----------
        trueW: N x K x D x C numpy array
            true weight matrix. for C=2, trueW[:,:,:,1] = 0 
        trueP: N x K x K numpy array
            true probability transition matrix
        truepi: k x 1 numpy vector
            probabilities for first latent of each session
        sessInd: list of int
            indices of each session start, together with last session end + 1
        save: boolean
            whether to save out simulated data
        
        Returns
        -------
        x: n x d numpy array
            simulated design matrix
        y: n x 1 numpy array
            simulated observation vector
        z: n x 1 numpy array
            simulated hidden states vector

        '''
        # check that weight and transition matrices are consistent with model hyperparameters
        if (trueW.shape != (self.N, self.K, self.D, self.C)):
            raise Exception(f'Weights need to have shape ({self.N}, {self.K}, {self.D}, {self.C})')
        if (trueP.shape != (self.N, self.K, self.K)):
            raise Exception(f'Transition matrix needs to have shape ({self.N}, {self.K}, {self.C})')
        
        x = np.zeros((self.N, self.D))
        y = np.zeros((self.N, self.C)).astype(int)
        z = np.zeros((self.N,),dtype=int)

        # input data x
        x[:,0] = 1 # bias term
        x[:,1] = stats.uniform.rvs(loc=-16,scale=33,size=self.N).astype(int)
        # standardizing sensory info
        x[:,1] = x[:,1] - x[:,1].mean()
        x[:,1] = x[:,1] / x[:,1].std()

        if (self.K==1):
            z[:] = 0
        else:
            # latent variables z 
            for t in range(0, self.N):
                if (t in sessInd[:-1]): # beginning of session has a new draw for latent
                    z[t] = np.random.choice(range(0, self.K), p=truepi)
                else:
                    z[t] = np.random.choice(range(0, self.K), p=trueP[t,z[t-1],:])
        
        # observation probabilities
        phi = self.observation_probability(x, trueW)

        for t in range(0, self.N):
            y[t,int(np.random.binomial(n=1, p=phi[t,z[t],1]))]=1 
        
        y = reshapeObs(y) # reshaping from n x c to n x 1

        return x, y, z

    
    def forward_pass(self, y, present, P, pi, phi, startSessInd=[0]):
        '''
        Calculates alpha scaled as part of the forward-backward algorithm in E-step 
       
        Parameters
        ----------
        y : T x 1 numpy vector 
            vector of observations with values 0,1,..,C-1
        present: T x 1 numpy vector
            0 means missing data (p(y_t=missing)=1 and 1 means present
        P : n x k x k numpy array
            matrix of transition probabilities
        pi: k x 1 numpy vector
            p(z_1) for z_1 first latent of every session
        phi : T x k x  c numpy array
            matrix of observation probabilities
        startSessInd: list of int
            if [0], then it means it's a single sessioon, else it's start indices of multiple sessions
        
        Returns
        -------
        alpha : T x k numpy vector
            matrix of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
        ct : T x 1 numpy vector
            vector of the forward marginal likelihoods p(y_t | y_1:t-1)
        ll : float
            marginal log-likelihood of the data p(y)
        '''
        T = y.shape[0]
        
        alpha = np.zeros((T, self.K)).astype(float) # forward probabilities p(z_t | y_1:t)
        alpha_prior = np.zeros((T, self.K)).astype(float) # prior probabilities p(z_t | y_1:t-1)
        ct = np.zeros((T)).astype(float) # forward marginal likelihoods p(y_t | y_1:t-1)

        # forward pass calculations
        for t in range(0,T):
            if (t in startSessInd): # time point 0
                # prior of z_1 before any data 
                alpha_prior[t,:] = pi #np.ones((1,self.k))/self.k = uniform prior
            else:
                alpha_prior[t,:] = (alpha[t-1,:].T @ P[t]) # conditional p(z_t | y_1:t-1)

            if (present[t]==1): # NOT missing data
                pxz = np.multiply(phi[t,:,y[t]], alpha_prior[t,:]) # joint P(y_1:t, z_t)
                ct[t] = np.sum(pxz) # conditional p(y_t | y_1:t-1)
                alpha[t,:] = pxz/ct[t] # conditional p(z_t | y_1:t)
            elif (present[t]==0): # missing data -> all derivations come from p(y_t=missing)=1
                ct[t] = 1 # so np.log(ct[t])=0 likelihood term not included
                alpha[t,:] = alpha_prior[t,:]
            else:
                raise Exception('present vector can only have 0s and 1s')
        
        ll = np.sum(np.log(ct)) # marginal log likelihood p(y_1:T) as sum of log conditionals p(y_t | y_1:t-1) for present data
        
        return alpha, ct, ll
    
    def backward_pass(self, y, present, P, phi, ct, startSessInd=[0]):
        '''
        Calculates beta scaled as part of the forward-backward algorithm in E-step 

        Parameters
        ----------
        y : T x 1 numpy vector
            vector of observations with values 0,1,..,C-1
        p : n x k x k numpy array
            matrix of transition probabilities
        phi : T x k x c numppy array
            matrix of observation probabilities
        ct : T x 1 numpy vector 
            veector of forward marginal likelihoods p(y_t | y_1:t-1), calculated at forward_pass
        present: T numpy vector
            0 means missing data (p(y_t=missing)=1), 1 means present

        Returns
        -------
        beta: T x k numpy array 
            matrix of backward conditional probabilities p(y_t+1:T | z_t) / p(y_t+1:T | y_1:t)
        '''

        T = y.shape[0]
        
        beta = np.zeros((T, self.K)).astype(float) # backward conditional probabilities p(y_t+1:T | z_t) / p(y_t+1:T | y_1:t)

        # last time point
        beta[-1] = 1 # p(z_T=1)

        # backward pass calculations
        for t in np.arange(T-2,-1,-1):
            if (t+1 in startSessInd): # irresepective if missing or not
                beta[t,:] = 1 # last time point of a session
            else:
                if (present[t+1] == 1): # NOT missing
                    beta[t,:] = P[t+1] @ (np.multiply(beta[t+1,:],phi[t+1,:,y[t+1]]))
                    beta[t,:] = beta[t,:] / ct[t+1] # scaling factor
                elif (present[t+1] == 0): # missing data
                    if (ct[t+1] != 1): # c[t+1] = 1 already from forward pass
                        raise Exception("c[t] should already be 1 from forward pass -> present in backward might not be matching with forward")
                    beta[t,:] = P[t+1] @ beta[t+1,:] # CHECK THIS
                else:
                    raise Exception('present vector can only have 0s and 1s')
        
        return beta
    
    def posteriorLatents(self, y, present, p, phi, alpha, beta, ct, startSessInd=[0]):
        ''' 
        calculates marginal posterior of latents gamma(z_t) = p(z_t | y_1:T)
        and joint posterior of successive latens zeta(z_t, z_t+1) = p(z_t, z_t+1 | y_1:T)

        Parameters
        ----------
        y : T x 1 numpy vector 
            vector of observations with values 0,1,..,C-1
        p : n x k x k numpy array
            matrix of transition probabilities
        phi : T x k x c numppy array
            matrix of observation probabilities
        alpha : T x k numpy vector
            marix of the conditional probabilities p(z_t | x_1:t, y_1:t)
        beta: T x k numpy array 
            matrix of backward conditional probabilities p(y_t+1:T | z_t) / p(y_t+1:T | y_1:t)
        ct : T x 1 numpy vector
            vector of the forward marginal likelihoods p(y_t | y_1:t-1)
        
        Returns
        -------
        gamma: T x k numpy array
            matrix of marginal posterior of latents p(z_t | y_1:T)
        zeta: T-1 x k x k 
            matrix of joint posterior of successive latens p(z_t, z_t+1 | y_1:T)
        '''
        
        T = ct.shape[0]
        gamma = np.zeros((T, self.K)).astype(float) # marginal posterior of latents
        zeta = np.zeros((T-1, self.K, self.K)).astype(float) # joint posterior of successive latents

        # gamma(z_t) = alpha(z_t) * beta(z_t)
        gamma = np.multiply(alpha, beta) 

        # zeta(z_t, z_t+1) =  alpha(z_t) * beta(z_t+1) * p (z_t+1 | z_t) * p(y_t+1 | z_t+1) / c_t+1
        for t in range(0,T-1):
            if (t+1 not in startSessInd): # to include these terms or not?
                if (present[t+1] == 1): # NOT missing data
                    alpha_beta = alpha[t,:].reshape((self.K, 1)) @ beta[t+1,:].reshape((1, self.K))
                    zeta[t,:,:] = np.multiply(alpha_beta, p[t+1]) 
                    zeta[t,:,:] = np.multiply(zeta[t,:,:],phi[t+1,:,y[t+1]]) 
                    zeta[t,:,:] = zeta[t,:,:] / ct[t+1]
                elif (present[t+1] == 0):
                    if (ct[t+1] != 1): # c[t] = 1 already from forward pass
                        raise Exception("c[t+1] should already be 1 from forward pass -> present  might not be matching with forward & backward")
                    alpha_beta = alpha[t,:].reshape((self.K, 1)) @ beta[t+1,:].reshape((1, self.K))
                    zeta[t,:,:] = np.multiply(alpha_beta, p[t+1])
                else:
                    raise Exception('present vector can only have 0s and 1s')
            
        return gamma, zeta

    def get_states_in_time(self, x, y, w, p, pi, sessInd=None):
        ''' 
        function that gets the distribution of states across trials/time-points after assigning the states with respective maximum probabilities
        '''
        T = x.shape[0]

        if sessInd is None:
            sessInd = [0, T]
            sess = 1 # entire data set has one session
        else:
            sess = len(sessInd)-1 # total number of sessions 

        # calculate observation probabilities given theta_old
        phi = self.observation_probability(x, w)
        
        zStates = np.zeros((T)).astype(int)

        for s in range(0,sess):
        # E step - forward and backward passes given theta_old (= previous w and p)
            alphaSess, ctSess, _ = self.forward_pass(y[sessInd[s]:sessInd[s+1]], p[sessInd[s]:sessInd[s+1]], pi, phi[sessInd[s]:sessInd[s+1],:,:])
            betaSess = self.backward_pass(y[sessInd[s]:sessInd[s+1]], p[sessInd[s]:sessInd[s+1]], phi[sessInd[s]:sessInd[s+1],:,:], ctSess)
            gammaSess, _ = self.posteriorLatents(y[sessInd[s]:sessInd[s+1]], p[sessInd[s]:sessInd[s+1]], phi[sessInd[s]:sessInd[s+1],:,:], alphaSess, betaSess, ctSess)
            
            # assigning max probabiity state to trial
            zStates[sessInd[s]:sessInd[s+1]] = np.argmax(gammaSess, axis=1)
        
        return zStates
    
    def generate_param(self, sessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)], model_type='standard'):
        ''' 
        Function that generates parameters w and P and is used for initialization of parameters during fitting

        Parameters
        ----------
        sessInd: list of int
            indices of each session start, together with last session end + 1
        transitionDistribution: list of length 2
            first is str of distribution type, second is parameter tuple 
                dirichlet ditribution comes with (alphaDiagonal, alphaOther) the concentration values for either main diagonal or other locations
        weightDistribution: list of length 2
            first is str of distribution type, second is parameter tuple
                uniform distribution comes with (low, high) 
                normal distribution comes with (mean, std)

        Returns
        ----------
        p: T x k x k numpy array
            probability transition matrix
        w: T x k x d x c numpy array
            weight matrix. for c=2, trueW[:,:,:,1] = 0 
        
        '''
        T = int(sessInd[-1])

        sess = len(sessInd)-1 # number of total sessions

        # initialize weight and transitions
        p = np.zeros((T, self.K, self.K))
        w = np.zeros((T, self.K, self.D, self.C))
        pi = np.ones((1, self.K))/self.K 

        # generating transition matrix 
        if (transitionDistribution[0] == 'dirichlet'):
            (alphaDiag, alphaOther) = transitionDistribution[1]
            for k in range(0, self.K):
                alpha = np.full((self.K), alphaOther)
                alpha[k] = alphaDiag # concentration parameter of Dirichlet for row k
                
                if model_type in ['standard','partial']:
                    p[:,k,:] = np.random.dirichlet(alpha)
                elif model_type in ['dynamic']:
                    for s in range(0,sess):
                        p[sessInd[s]:sessInd[s+1],k,:] = np.random.dirichlet(alpha)
                else:
                    raise Exception("model_type can only be standard (static weights and transition matrix), partial (dynamic weights and static transition matrix), or dynamic (dynamic weights and transition matrix)")
        else:
            raise Exception("Transition distribution can only be dirichlet")
        
        # generating weight matrix
        if (weightDistribution[0] == 'uniform'):
            (low, high) = weightDistribution[1]
            if model_type == 'standard':
                rv = np.random.uniform(low, high, (self.K, self.D))
                w[:,:,:,1] = rv
            elif model_type in ['partial', 'dynamic']:
                for s in range(0,sess):
                    rv = np.random.uniform(low, high, (self.K, self.D))
                    w[sessInd[s]:sessInd[s+1],:,:,1] = rv
        elif (weightDistribution[0] == 'normal'):
            (mean, std) = weightDistribution[1]
            if model_type == 'standard':
                rv = np.random.normal(mean, std, (self.K, self.D))
                w[:,:,:,1] = rv
            elif model_type in ['partial', 'dynamic']:
                for s in range(0,sess):
                    rv = np.random.normal(mean, std, (self.K, self.D))
                    w[sessInd[s]:sessInd[s+1],:,:,1] = rv
        else:
            raise Exception("Weight distribution can only be uniform or normal")

        return p, pi, w 
    
    def value_weight_loss_function(self, currentW, x, y, present, gamma, prevW, nextW, sigma, model_type = 'standard', L2penaltyW=0):
        ''' 
        weight loss function to optimize the weights in M-step of fitting function is calculated as negative of weighted log likelihood + prior terms 
        coming from drifting wrt neighboring sessions

        it also returns the gradient of the above function to be used for faster optimization 

        for one state only

        L(currentW from state k) = sum_t gamma(z_t=k) * log p(y_t | z_t=k) + log P(currentW | prevW) + log P(currentW | nextW),
        where gamma matrix are fixed by old parameters but observation probabilities p(y_t | z_t=k) are updated with currentW


        Parameters
        ----------
        currentW: d numpy vector
            weights of current session for C=0 and one particular state
        x: T x d numpy array
            design matrix
        y : T numpy vector 
            vector of observations with values 0,1,..,C-1
        gamma: T numpy vector
            matrix of marginal posterior of latents p(z_t | y_1:T)
        prevW: d x c numpy array
            weights of previous session
        nextW: d x c numpy array
            weights of next session
        sigma: d numpy vector
            std parameters of normal distribution for each state and each feature
        
        Returns
        ----------
        -lf: float
            loss function for currentW to be minimized

        '''

        # number of datapoints
        T = x.shape[0]

        # check model type

        model_type = 'standard'

        sessW = np.zeros((T, 1, self.D, self.C)) # K=1
        for t in range(0,T):
            sessW[t,0,:,1] = currentW[:]

        # log observation probability
        logPhi = self.log_observation_probability(x, sessW) # N x K x C phi matrix calculated with current weights

        # weighted log likelihood term of loss function
        lf = 0
        for t in range(0, T):
            if (present[t] == 1): # only for present data (not missing)
                lf += gamma[t] * logPhi[t,0,y[t]]

        if model_type == 'standard': # standard GLM-HMM (static weights and transition matrix across sessions)
            
            # penalty term for size of weights for the standard model
            lf+= L2penaltyW * -1/2 * currentW[:].T @ currentW[:]

        elif model_type in ['partial','dynamic']: # time-varying weights in dynamic GLM-HMM

            if (sigma.sum() != 0): # sigma does not have any 0s
                # inverse of covariance matrix
                invSigma = np.square(1/sigma[:])
                det = np.prod(invSigma)
                invCov = np.diag(invSigma)
            else:
                raise Exception("Sigma should not contain any 0s if model_type is not standard, but values very close to 0 are allowed")

            if (prevW is not None):
                # logpdf of multivariate normal (ignoring pi constant)
                lf +=  -1/2 * np.log(det) - 1/2 * (currentW[:] - prevW[:]).T @ invCov @ (currentW[:] - prevW[:])
            if (nextW is not None):
                # logpdf of multivariate normal (ignoring pi constant)
                lf += -1/2 * np.log(det) - 1/2 * (currentW[:] - nextW[:]).T @ invCov @ (currentW[:] - nextW[:])
        else:
            raise Exception("model_type can only be standard (static weights and transition matrix), partial (dynamic weights and static transition matrix), or dynamic (dynamic weights and transition matrix)")

        return -lf 

    def grad_weight_loss_function(self, currentW, x, y, present, gamma, prevW, nextW, sigma, model_type='standard', L2penaltyW=0):
        ''' 
        weight loss function to optimize the weights in M-step of fitting function is calculated as negative of weighted log likelihood + prior terms 
        coming from drifting wrt neighboring sessions

        it also returns the gradient of the above function to be used for faster optimization 

        for one state only

        L(currentW from state k) = sum_t gamma(z_t=k) * log p(y_t | z_t=k) + log P(currentW | prevW) + log P(currentW | nextW),
        where gamma matrix are fixed by old parameters but observation probabilities p(y_t | z_t=k) are updated with currentW


        Parameters
        ----------
        currentW: d numpy vector
            weights of current session for C=0 and one particular state
        x: T x d numpy array
            design matrix
        y : T x 1 numpy vector 
            vector of observations with values 0,1,..,C-1
        gamma: T x k numpy array
            matrix of marginal posterior of latents p(z_t | y_1:T)
        prevW: k x d x c numpy array
            weights of previous session
        nextW: k x d x c numpy array
            weights of next session
        sigma: k x d numpy array
            std parameters of normal distribution for each state and each feature
        
        Returns
        ----------
        -lf: float
            loss function for currentW to be minimized

        '''

        # number of datapoints
        T = x.shape[0]

        sessW = np.zeros((T, 1, self.D, self.C)) # K=1
        for t in range(0,T):
            sessW[t,0,:,1] = currentW[:]

        # weighted log likelihood term of loss function
        grad = np.zeros((self.D))
        for t in range(0, T):
            if (present[t] == 1): # only for present data (not missing)
                grad += gamma[t] * (softplus_deriv(-x[t] @ currentW) - (1 - y[t])) * x[t]

        if model_type == 'standard': # standard GLM-HMM (static weights and transition matrix across sessions)

            # penalty term for size of weights for the standard model
            grad += L2penaltyW * - currentW[:]

        elif model_type in ['partial','dynamic']: # time-varying weights in dynamic GLM-HMM

            if (sigma.sum() != 0): # sigma does not have any 0s
                # inverse of covariance matrix
                invSigma = np.square(1/sigma[:])
            else:
                raise Exception("Sigma should not contain any 0s if model_type is not standard, but values very close to 0 are allowed")

            if (prevW is not None):
                # gradient logpdf of multivariate normal (ignoring pi constant)
                grad += - np.multiply(invSigma, currentW[:] - prevW[:])
            if (nextW is not None):
                # gradient logpdf of multivariate normal (ignoring pi constant)
                grad += - np.multiply(invSigma, currentW[:] - nextW[:])
        else:
            raise Exception("model_type can only be standard (static weights and transition matrix), partial (dynamic weights and static transition matrix), or dynamic (dynamic weights and transition matrix)")

        return -grad
    
    def fit(self, x, y, present, initP, initpi, initW, sigma=0, alpha=0, A=None, sessInd=None, maxIter=250, tol=1e-3, model_type='standard',  L2penaltyW=0, priorDirP = [10,1], fit_init_states=False):
        '''
        Fitting function based on EM algorithm. Algorithm: observation probabilities are calculated with old weights for all sessions, then 
        forward and backward passes are done for each session, weights are optimized for one particular session (phi stays the same),
        then after all weights are optimized (in consecutive increasing order), the transition matrix is updated with the old zetas that
        were calculated before weights were optimized

        Parameters
        ----------
        x: T x d numpy array
            design matrix
        y : T x 1 numpy vector 
            vector of observations with values 0,1,..,C-1
        initP :k x k numpy array
            initial matrix of transition probabilities
        initW: n x k x d x c numpy array
            initial weight matrix
        sigma: k x d numpy array
            st dev of normal distr for weights drifting over sessions
            if one is 0, then all are 0 (=standard GLM-HMM)
        sessInd: list of int
            indices of each session start, together with last session end + 1
        pi0 : k x 1 numpy vector
            initial k x 1 vector of state probabilities for t=1.
        maxiter : int
            The maximum number of iterations of EM to allow. The default is 300.
        tol : float
            The tolerance value for the loglikelihood to allow early stopping of EM. The default is 1e-3.
        priorDirP: list of length 2
            diagonal and off diagonal terms for Dirichlet prior (+1) on transition matrix 
        
        Returns
        -------
        p: k x k numpy array
            fitted probability transition matrix
        w: T x k x d x c numpy array
            fitteed weight matrix
        ll: float
            marginal log-likelihood of the data p(y)
        '''
        # number of datapoints
        T = x.shape[0]

        # initialize weights and transition matrix
        w = np.copy(initW)
        p = np.copy(initP)
        pi = np.copy(initpi)

        # initialize zeta = joint posterior of successive latents 
        zeta = np.zeros((T-1, self.K, self.K)).astype(float) 
        # initialize gamma postierior of latents
        gamma = np.zeros((T, self.K)).astype(float)
        # initialize marginal log likelihood p(y)
        ll = np.zeros((maxIter)).astype(float) 

        if model_type == 'dynamic':
            # checking that A (only used for dynamic model) is a transition matrix
            if (A.shape != (self.K, self.K)):
                raise Exception("Global P should be an array of shape (k,k)")
            else:
                for i in range(0, self.K):
                    if (abs(A[i,:].sum() - 1) > 1e-6):
                        raise Exception(f'Global P row {i} does not sum up to 1')

        if (fit_init_states==True and len(sessInd)-1<50):
            raise Exception("Should not fit init states when less than 20 sessions due to high uncertainty")

        sess = len(sessInd)-1 # total number of sessions
        startSessInd = sessInd[:-1] # only the first trial of each session

        for iter in range(maxIter):
            if (iter%100==0):
                print(iter)
                
            # calculate observation probabilities given theta_old
            phi = self.observation_probability(x, w)

            # E-step for each session 
            for s in range(0,sess):
                    
                # E step - forward and backward passes given theta_old (= previous w and p)
                alphaSess, ctSess, llSess = self.forward_pass(y[sessInd[s]:sessInd[s+1]], present, p[sessInd[s]:sessInd[s+1]], pi, phi[sessInd[s]:sessInd[s+1],:,:], startSessInd)
                betaSess = self.backward_pass(y[sessInd[s]:sessInd[s+1]], present, p[sessInd[s]:sessInd[s+1]], phi[sessInd[s]:sessInd[s+1],:,:], ctSess, startSessInd)
                gammaSess, zetaSess = self.posteriorLatents(y[sessInd[s]:sessInd[s+1]], present, p[sessInd[s]:sessInd[s+1]], phi[sessInd[s]:sessInd[s+1],:,:], alphaSess, betaSess, ctSess, startSessInd)
                    
                # merging zeta for all sessions 
                zeta[sessInd[s]:sessInd[s+1]-1,:,:] = zetaSess[:,:,:] 
                gamma[sessInd[s]:sessInd[s+1]] = gammaSess
                ll[iter] += llSess

                if model_type in ['partial','dynamic']: # models with weights varying across sessions

                    for k in range(0,self.K):
                        prevW = w[sessInd[s-1],k,:,1] if s!=0 else None #  d x c matrix of previous session weights
                        nextW = w[sessInd[s+1],k,:,1] if s!=sess-1 else None #  d x c matrix of next session weights
                        w_flat = np.ndarray.flatten(w[sessInd[s],k,:,1]) # flatten weights for optimization 
                        opt_val = lambda w: self.value_weight_loss_function(w, x[sessInd[s]:sessInd[s+1]], y[sessInd[s]:sessInd[s+1]], present, gammaSess[:,k], prevW, nextW, sigma[k,:], model_type=model_type, L2penaltyW=L2penaltyW)
                        opt_grad = lambda w: self.grad_weight_loss_function(w, x[sessInd[s]:sessInd[s+1]], y[sessInd[s]:sessInd[s+1]], present, gammaSess[:,k], prevW, nextW, sigma[k,:], model_type=model_type, L2penaltyW=L2penaltyW)
                        optimized = minimize(opt_val, w_flat, jac=opt_grad, method='L-BFGS-B')
                        w[sessInd[s]:sessInd[s+1],k,:,1] = optimized.x # updating weight w for current session

                # M-step for transition matrix P for each session - closed form update (dynamic model)
                if model_type == 'dynamic':
                    for i in range(0, self.K):
                        for j in range(0, self.K):
                            p[sessInd[s]:sessInd[s+1],i,j] = (zetaSess[:,i,j].sum() + alpha * A[i,j])/(zetaSess[:,i,:].sum() + alpha) 
                
            if model_type == 'standard':
                for k in range(0,self.K):
                    prevW = None #  d x c matrix of previous session weights
                    nextW = None #  d x c matrix of next session weights
                    w_flat = np.ndarray.flatten(w[0,k,:,1]) # flatten weights for optimization 
                    opt_val = lambda w: self.value_weight_loss_function(w, x, y, present, gamma[:,k], prevW, nextW, sigma[k,:], model_type=model_type, L2penaltyW=L2penaltyW)
                    opt_grad = lambda w: self.grad_weight_loss_function(w, x, y, present, gamma[:,k], prevW, nextW, sigma[k,:], model_type=model_type, L2penaltyW=L2penaltyW)
                    optimized = minimize(opt_val, w_flat, jac=opt_grad, method='L-BFGS-B')
                    w[:,k,:,1] = optimized.x # updating weights w

            # M-step for transition matrix P (when static across session) - closed form update (standard and partial model)
            if model_type in ['standard','partial']:          
                for i in range(0, self.K):
                    for j in range(0, self.K):
                        if priorDirP is not None:
                            p[:,i,j] = (zeta[:,i,j].sum() + priorDirP[i!=j]) / (zeta[:,i,:].sum() + priorDirP[i!=j] + priorDirP[i==j] * (self.K-1) ) # added prior of Dirichlet(11,2,2..,2)
                        else:
                            p[:,i,j] = zeta[:,i,j].sum() / zeta[:,i,:].sum()
                
            if (fit_init_states==True):
                # initial distribution of latents
                pi = gamma[sessInd[:-1],:].sum(axis=0) 
                pi = pi / pi.sum() # normalize
                    
            # check if stopping early
            if (iter >= 10 and ll[iter] - ll[iter-1] < tol):
                break

        return p, pi.reshape((self.K)), w, ll

    def evaluate(self, x, y, sessInd, presentTest, p, pi, w):
        ''' 
        function that gives per session test log-like and test accuracy with forward pass using all data
        '''

        N = x.shape[0]

        present = np.ones((N))
        phi = self.observation_probability(x=x, w=w)

        alpha, ct, ll = self.forward_pass(y, present, p, pi, phi, sessInd[:-1])
        beta = self.backward_pass(y, present, p, phi, ct, sessInd[:-1])
        gamma, _ = self.posteriorLatents(y, present, p, phi, alpha, beta, ct, sessInd[:-1])

        sess = len(sessInd) - 1
        llTest_per_session = np.zeros((sess))
        for s in range(sess):
            ct_session = ct[sessInd[s]:sessInd[s+1]]
            presentTest_session = presentTest[sessInd[s]:sessInd[s+1]]
            Ntest_session = presentTest_session.sum() # number of trials in test in this session
            llTest_per_session[s] = np.sum(np.log(ct_session[np.argwhere(presentTest_session==1)])) / Ntest_session # average test log-like per session

        Ntest = presentTest.sum() # number of trials in test     
        llTest = np.sum(np.log(ct[np.argwhere(presentTest==1)])) / Ntest # average test log-likelihood per trial

        gammaTest = gamma[np.argwhere(presentTest==1)].reshape((Ntest, self.K))
        phiTest = phi[np.argwhere(presentTest==1)].reshape((Ntest, self.K, self.C))
        yTest = y[np.argwhere(presentTest==1)].reshape((Ntest,))
        pChoice = np.zeros((gammaTest.shape[0], self.C)) # p(y_t | x_t, w_t)
        pChoice[:,1] = np.sum(np.multiply(gammaTest, phiTest[:,:,1]), axis=1) # p(y_t | x_t, w_t) = sum over k of gamma(z_t=k) * p (y_t|z_t=k, z_t, w_t)
        pChoice[:,0] = 1 - pChoice[:,1]
        choiceHard = np.argmax(pChoice, axis=1) # predicted choice of animal

        Nwrong = np.logical_xor(choiceHard, yTest).sum()
        accuracyTest = (Ntest - Nwrong) / Ntest * 100 # correct predictions on observed y
            
        return llTest_per_session, llTest, accuracyTest 

    def get_posterior_latent(self, p, pi, w, x, y, present, sessInd, sortedStateInd=None):
        if (sortedStateInd is not None):
        # permute states
            w = w[:,sortedStateInd,:,:]
            p = p[:,sortedStateInd,:][:,:,sortedStateInd]

        T = x.shape[0]

        if sessInd is None:
            sessInd = [0, T]
            sess = 1 # equivalent to saying the entire data set has one session
        else:
            sess = len(sessInd)-1 # total number of sessions 

        gamma = np.zeros((T, self.K)).astype(float) 
        phi = self.observation_probability(x, w)

        for s in range(0,sess):
            # E step - forward and backward passes given theta_old (= previous w and p)
            alphaSess, ctSess, llSess = self.forward_pass(y[sessInd[s]:sessInd[s+1]], present, p[sessInd[s]:sessInd[s+1]], pi, phi[sessInd[s]:sessInd[s+1],:,:])
            betaSess = self.backward_pass(y[sessInd[s]:sessInd[s+1]], present, p[sessInd[s]:sessInd[s+1]], phi[sessInd[s]:sessInd[s+1],:,:], ctSess)
            gammaSess, _ = self.posteriorLatents(y[sessInd[s]:sessInd[s+1]], present, p[sessInd[s]:sessInd[s+1]], phi[sessInd[s]:sessInd[s+1],:,:], alphaSess, betaSess, ctSess)
            
            # concatenating across sessions
            gamma[sessInd[s]:sessInd[s+1],:] = gammaSess[:,:] 
        
        return gamma
    
  

        
