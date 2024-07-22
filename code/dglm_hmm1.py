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

class dGLM_HMM1():
    """
    Class for fitting driftinig GLM-HMM model 1 in which weights are constant within session but vary across sessions
    Code just works for c=2 at the moment!!!
    Weights for class c=0 are always kept to 0 (so then emission probability becomes 1/(1+exp(-wTx)))
    X columns represent [bias, sensory] in this order

    Notation: 
        n: number of data points
        k: number of states (states)
        d: number of features (inputs to design matrix)
        c: number of classes (possible observations)
        X: design matrix (n x d)
        Y: observations (n x c) or (n x 1)
        w: weights mapping x to y (n x k x d x c)
    """

    def __init__(self, n, k, d, c):
            self.n, self.k, self.d, self.c  = n, k, d, c
    
    # Iris' fct has columns reverrsed
    def log_observation_probability(self, x, w):
        '''
        Calculating observation probabilities for given design matrix x and weight matrix w
        C = 2 only

        Parameters
        ----------
        x: Ncurrent x D numpy array
            input matrix
        w: Ncurrent x K x D x C numpy array
            weight matrix

        Returns
        -------
        logphi: Ncurrent x K x C numpy array
                log of observation probabilities matrix (already normalized inside log)
            '''
        C = 2
        Ncurrent = x.shape[0]

        if (w.ndim == 3): # it means K=1
            w = w.reshape((w.shape[0],1,w.shape[1],w.shape[2]))
            K = 1
        elif (w.ndim == 4): # K>=2
            K = w.shape[1]
        else:
            raise Exception("Weight matrix should have 3 or 4 dimensions (N X D x C or N x K x D x C)")

        logphi = np.zeros((Ncurrent, K, self.c)) 
        for k in range(0, K):
            # be careful with soft plus function in relation to obs prob!! as the two cases for c are different
            logphi[:,k,1] = - softplus(np.sum(w[:,k,:,1]*x,axis=1))
            logphi[:,k,0] = - softplus(-np.sum(w[:,k,:,1]*x,axis=1))
            
        return logphi

    def observation_probability(self, x, w):
        phi = self.log_observation_probability(x, w)
        phi = np.exp(phi)
        return phi  

    def simulate_data(self, trueW, trueP, truepi, sessInd, save=False, title='sim'):
        '''
        function that simulates X and Y data from true weights and true transition matrix

        Parameters
        ----------
        trueW: n x k x d x c numpy array
            true weight matrix. for c=2, trueW[:,:,:,1] = 0 
        trueP: T x k x k numpy array
            true probability transition matrix
        priorZstart: int
            0.5 probability of starting a session with state 0 (works for C=2)
        sessInd: list of int
            indices of each session start, together with last session end + 1
        save: boolean
            whether to save out simulated data
        pi0: float
            constant between 0 and 1, representing probability that first latent in a session is state 0
            
        Returns
        -------
        x: n x d numpy array
            simulated design matrix
        y: n x 1 numpy array
            simulated observation vector
        z: n x 1 numpy array
            simulated hidden states vector

        '''
        # check that weight and transition matrices are valid options
        if (trueW.shape != (self.n, self.k, self.d, self.c)):
            raise Exception(f'Weights need to have shape ({self.n}, {self.k}, {self.d}, {self.c})')
        
        if (trueP.shape != (self.k, self.k)):
            raise Exception(f'Transition matrix needs to have shape ({self.k}, {self.k})')
        
        x = np.zeros((self.n, self.d))
        y = np.zeros((self.n, self.c)).astype(int)
        z = np.zeros((self.n,),dtype=int)

        # input data x
        x[:,0] = 1 # bias term
        x[:,1] = stats.uniform.rvs(loc=-16,scale=33,size=self.n).astype(int)
        # standardizing sensory info
        x[:,1] = x[:,1] - x[:,1].mean()
        x[:,1] = x[:,1] / x[:,1].std()

        # TRY normal distribution for x[:,1]

        if (self.k==1):
            z[:] = 0
        else:
            # latent variables z 
            for t in range(0, self.n):
                if (t in sessInd[:-1]): # beginning of session has a new draw for latent
                    z[t] = np.random.choice(range(0, self.k), p=truepi)
                else:
                    z[t] = np.random.choice(range(0, self.k), p=trueP[z[t-1],:])
        
        # observation probabilities
        phi = self.observation_probability(x, trueW)

        for t in range(0, self.n):
            y[t,int(np.random.binomial(n=1, p=phi[t,z[t],1]))]=1
        
        y = reshapeObs(y) # reshaping from n x c to n x 1

        if (save==True):
            np.save(f'../data_M1/{title}x', x)
            np.save(f'../data_M1/{title}y', y)
            np.save(f'../data_M1/{title}z', z)

        return x, y, z

    # already checked with Iris' function that it is correct
    def forward_pass(self, y, present, P, pi, phi, startSessInd=[0]):
        '''
        Calculates alpha scaled as part of the forward-backward algorithm in E-step 
       
        Parameters
        ----------
        y : T x 1 numpy vector 
            vector of observations with values 0,1,..,C-1
        P : k x k numpy array 
            matrix of transition probabilities
        pi: k x 1 numpy vector
            p(z_1) for z_1 first latent of every session
        phi : T x k x  c numpy array
            matrix of observation probabilities
        present: T numpy vector
            0 means missing data (p(y_t=missing)=1), 1 means present
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
        
        alpha = np.zeros((T, self.k)).astype(float) # forward probabilities p(z_t | y_1:t)
        alpha_prior = np.zeros((T, self.k)).astype(float) # prior probabilities p(z_t | y_1:t-1)
        ct = np.zeros((T)).astype(float) # forward marginal likelihoods p(y_t | y_1:t-1)

        # forward pass calculations
        for t in range(0,T):
            if (t in startSessInd): # time point 0
                # prior of z_1 before any data 
                alpha_prior[t,:] = pi #np.ones((1,self.k))/self.k = uniform prior
            else:
                alpha_prior[t,:] = (alpha[t-1,:].T @ P) # conditional p(z_t | y_1:t-1)

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
    
    # already checked with Iris' function that it is correct
    def backward_pass(self, y, present, P, phi, ct, startSessInd=[0]):
        '''
        Calculates beta scaled as part of the forward-backward algorithm in E-step 

        Parameters
        ----------
        y : T x 1 numpy vector
            vector of observations with values 0,1,..,C-1
        p : k x k numpy array
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
        
        beta = np.zeros((T, self.k)).astype(float) # backward conditional probabilities p(y_t+1:T | z_t) / p(y_t+1:T | y_1:t)

        # last time point
        beta[-1] = 1 # p(z_T=1)

        # backward pass calculations
        for t in np.arange(T-2,-1,-1):
            if (t+1 in startSessInd): # irresepective if missing or not
                beta[t,:] = 1 # last time point of a session
            else:
                if (present[t+1] == 1): # NOT missing
                    beta[t,:] = P @ (np.multiply(beta[t+1,:],phi[t+1,:,y[t+1]]))
                    beta[t,:] = beta[t,:] / ct[t+1] # scaling factor
                elif (present[t+1] == 0): # missing data
                    if (ct[t+1] != 1): # c[t+1] = 1 already from forward pass
                        raise Exception("c[t] should already be 1 from forward pass -> present in backward might not be matching with forward")
                    beta[t,:] = P @ beta[t+1,:] # CHECK THIS
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
        p : k x k numpy array
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
        gamma = np.zeros((T, self.k)).astype(float) # marginal posterior of latents
        zeta = np.zeros((T-1, self.k, self.k)).astype(float) # joint posterior of successive latents

        # gamma(z_t) = alpha(z_t) * beta(z_t)
        gamma = np.multiply(alpha, beta) 

        # zeta(z_t, z_t+1) =  alpha(z_t) * beta(z_t+1) * p (z_t+1 | z_t) * p(y_t+1 | z_t+1) / c_t+1
        for t in range(0,T-1):
            if (t+1 not in startSessInd): # to include these terms or not?
                if (present[t+1] == 1): # NOT missing data
                    alpha_beta = alpha[t,:].reshape((self.k, 1)) @ beta[t+1,:].reshape((1, self.k))
                    zeta[t,:,:] = np.multiply(alpha_beta,p) 
                    zeta[t,:,:] = np.multiply(zeta[t,:,:],phi[t+1,:,y[t+1]]) # Iris has index t at phi instead
                    zeta[t,:,:] = zeta[t,:,:] / ct[t+1]
                elif (present[t+1] == 0):
                    if (ct[t+1] != 1): # c[t] = 1 already from forward pass
                        raise Exception("c[t+1] should already be 1 from forward pass -> present  might not be matching with forward & backward")
                    alpha_beta = alpha[t,:].reshape((self.k, 1)) @ beta[t+1,:].reshape((1, self.k))
                    zeta[t,:,:] = np.multiply(alpha_beta,p)
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
            alphaSess, ctSess, _ = self.forward_pass(y[sessInd[s]:sessInd[s+1]], p, pi, phi[sessInd[s]:sessInd[s+1],:,:])
            betaSess = self.backward_pass(y[sessInd[s]:sessInd[s+1]], p, phi[sessInd[s]:sessInd[s+1],:,:], ctSess)
            gammaSess, _ = self.posteriorLatents(y[sessInd[s]:sessInd[s+1]], p, phi[sessInd[s]:sessInd[s+1],:,:], alphaSess, betaSess, ctSess)
            
            # assigning max probabiity state to trial
            zStates[sessInd[s]:sessInd[s+1]] = np.argmax(gammaSess, axis=1)
        
        return zStates
    
    def generate_param(self, sessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)]):
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
        p: k x k numpy array
            probability transition matrix
        w: T x k x d x c numpy array
            weight matrix. for c=2, trueW[:,:,:,1] = 0 
        
        '''
        T = int(sessInd[-1])

        sess = len(sessInd)-1 # number of total sessions

        # initialize weight and transitions
        p = np.zeros((self.k, self.k))
        w = np.zeros((T, self.k, self.d, self.c))
        pi = np.ones((1, self.k))/self.k 

        # generating transition matrix 
        if (transitionDistribution[0] == 'dirichlet'):
            (alphaDiag, alphaOther) = transitionDistribution[1]
            for k in range(0, self.k):
                alpha = np.full((self.k), alphaOther)
                alpha[k] = alphaDiag # concentration parameter of Dirichlet for row k
                p[k,:] = np.random.dirichlet(alpha)
        else:
            raise Exception("Transition distribution can only be dirichlet")
        
        # generating weight matrix
        if (weightDistribution[0] == 'uniform'):
            (low, high) = weightDistribution[1]
            for s in range(0,sess):
                rv = np.random.uniform(low, high, (self.k, self.d))
                w[sessInd[s]:sessInd[s+1],:,:,1] = rv
        elif (weightDistribution[0] == 'normal'):
            (mean, std) = weightDistribution[1]
            for s in range(0,sess):
                rv = np.random.normal(mean, std, (self.k, self.d))
                w[sessInd[s]:sessInd[s+1],:,:,1] = rv
        else:
            raise Exception("Weight distribution can only be uniform or normal")

        return p, pi, w 
    
    def value_weight_loss_function(self, currentW, x, y, present, gamma, prevW, nextW, sigma, L2penaltyW=1):
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

        sessW = np.zeros((T, 1, self.d, self.c)) # K=1
        for t in range(0,T):
            sessW[t,0,:,1] = currentW[:]

        # log observation probability
        logPhi = self.log_observation_probability(x, sessW) # N x K x C phi matrix calculated with currentW

        # weighted log likelihood term of loss function
        lf = 0
        for t in range(0, T):
            if (present[t] == 1): # only for present data (not missing)
                lf += gamma[t]*logPhi[t,0,y[t]]

        if (sigma.sum() != 0): # sigma does not have any 0s
            # inverse of covariance matrix
            invSigma = np.square(1/sigma[:])
            det = np.prod(invSigma)
            invCov = np.diag(invSigma)

        if (prevW is not None):
            # logpdf of multivariate normal (ignoring pi constant)
            lf +=  -1/2 * np.log(det) - 1/2 * (currentW[:] - prevW[:,1]).T @ invCov @ (currentW[:] - prevW[:,1])
        if (nextW is not None):
            # logpdf of multivariate normal (ignoring pi constant)
            lf += -1/2 * np.log(det) - 1/2 * (currentW[:] - nextW[:,1]).T @ invCov @ (currentW[:] - nextW[:,1])
                    
        # penalty term for size of weights - NOT NECESSARY FOR NOW
        lf += L2penaltyW * -1/2 * currentW[:].T @ currentW[:]

        return -lf 

    def grad_weight_loss_function(self, currentW, x, y, present, gamma, prevW, nextW, sigma, L2penaltyW=1):
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

        sessW = np.zeros((T, 1, self.d, self.c)) # K=1
        for t in range(0,T):
            sessW[t,0,:,1] = currentW[:]

        # weighted log likelihood term of loss function
        grad = np.zeros((self.d))
        for t in range(0, T):
            if (present[t] == 1): # only for present data (not missing)
                grad += gamma[t] * (softplus_deriv(-x[t] @ currentW) - y[t]) * x[t]

        if (sigma.sum() != 0): # sigma does not have any 0s
            # inverse of covariance matrix
            invSigma = np.square(1/sigma[:])

        if (prevW is not None): # previous session
            # gradient of logpdf of multivariate normal (ignoring pi constant)
            grad += - np.multiply(invSigma, currentW[:] - prevW[:,1])
        if (nextW is not None): # next session
            # gradient of logpdf of multivariate normal (ignoring pi constant)
            grad += - np.multiply(invSigma, currentW[:] - nextW[:,1])
                    
        # penalty term for size of weights - NOT NECESSARY FOR NOW
        grad += L2penaltyW * - currentW[:]

        return -grad
    
    def fit(self, x, y, present, initP, initpi, initW, sigma, sessInd=None, maxIter=250, tol=1e-3, L2penaltyW=1, priorDirP = [10,1], fit_init_states=False):
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

        if sessInd is None:
            sessInd = [0, T]

        # initialize weights and transition matrix
        w = np.copy(initW)
        p = np.copy(initP)
        pi = np.copy(initpi)

        # initialize zeta = joint posterior of successive latents 
        zeta = np.zeros((T-1, self.k, self.k)).astype(float) 
        # initialize gamma postierior of latents
        gamma = np.zeros((T, self.k)).astype(float)
        # initialize marginal log likelihood p(y)
        ll = np.zeros((maxIter)).astype(float) 

        if (fit_init_states==True and len(sessInd)-1<50):
            raise Exception("Should not fit init states when less than 50 sessions due to high uncertainty")

        # dealing with case sigma=0
        if (np.sum(sigma) == 0):
            startSessInd = sessInd[:-1]
            sessInd = [0, T]
        else:
            for k in range(0, self.k):
                for d in range(0, self.d):
                    if (sigma[k,d] == 0):
                        raise Exception ('All or no elemenets of sigma are 0')

            startSessInd = [0]
        
        sess = len(sessInd)-1

        # prior coefficients on transition matrix P
        priorP = np.zeros((self.k, self.k))
        if (priorDirP != None):
            for i in range(0, self.k):
                for j in range(0, self.k):
                    if (i==j):
                        priorP[i, j] = priorDirP[0] # diagonal term
                    else:
                        priorP[i, j] = priorDirP[1] # off diagonal term

        for iter in range(maxIter):
            # if (iter%100==0):
            #     print(iter)
            
            # calculate observation probabilities given theta_old
            phi = self.observation_probability(x, w)

            # EM step for each session independently 
            for s in range(0,sess):
                
                # E step - forward and backward passes given theta_old (= previous w and p)
                alphaSess, ctSess, llSess = self.forward_pass(y[sessInd[s]:sessInd[s+1]], present, p, pi, phi[sessInd[s]:sessInd[s+1],:,:], startSessInd)
                betaSess = self.backward_pass(y[sessInd[s]:sessInd[s+1]], present, p, phi[sessInd[s]:sessInd[s+1],:,:], ctSess, startSessInd)
                gammaSess, zetaSess = self.posteriorLatents(y[sessInd[s]:sessInd[s+1]], present, p, phi[sessInd[s]:sessInd[s+1],:,:], alphaSess, betaSess, ctSess, startSessInd)
                
                # merging zeta for all sessions 
                zeta[sessInd[s]:sessInd[s+1]-1,:,:] = zetaSess[:,:,:] 
                gamma[sessInd[s]:sessInd[s+1]] = gammaSess
                ll[iter] += llSess
                
                for k in range(0,self.k):
                    prevW = w[sessInd[s-1],k,:,:] if s!=0 else None #  d x c matrix of previous session weights
                    nextW = w[sessInd[s+1],k,:,:] if s!=sess-1 else None #  d x c matrix of next session weights
                    w_flat = np.ndarray.flatten(w[sessInd[s],k,:,1]) # flatten weights for optimization 
                    #optimized = minimize(self.value_weight_loss_function, w_flat, args=(x[sessInd[s]:sessInd[s+1]], y[sessInd[s]:sessInd[s+1]], gammaSess[:,k], prevW, nextW, sigma[k,:]))
                    opt_val = lambda w: self.value_weight_loss_function(w, x[sessInd[s]:sessInd[s+1]], y[sessInd[s]:sessInd[s+1]], present, gammaSess[:,k], prevW, nextW, sigma[k,:], L2penaltyW=L2penaltyW)
                    opt_grad = lambda w: self.grad_weight_loss_function(w, x[sessInd[s]:sessInd[s+1]], y[sessInd[s]:sessInd[s+1]], present, gammaSess[:,k], prevW, nextW, sigma[k,:], L2penaltyW=L2penaltyW)
                    optimized = minimize(opt_val, w_flat, jac=opt_grad, method='L-BFGS-B')
                    w[sessInd[s]:sessInd[s+1],k,:,1] = optimized.x # updating weight w for current session

            # M-step for transition matrix p - for all sessions together
            for i in range(0, self.k):
                for j in range(0, self.k):
                    p[i,j] = (zeta[:,i,j].sum() + priorP[i,j]) / (zeta[:,i,:].sum() + priorP[i,:].sum()) # closed form update
            
            if (fit_init_states==True):
                # M-step for initial distribution of latents
                if (np.sum(sigma) == 0):
                    pi = gamma[startSessInd,:].sum(axis=0) 
                else:
                    # initial distribution of latents
                    pi = gamma[sessInd[:-1],:].sum(axis=0) 
                pi = pi / pi.sum() # normalize
        
            # check if stopping early 
            if (iter >= 10 and ll[iter] - ll[iter-1] < tol):
                break

        return p, pi.reshape((self.k)), w, ll

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
            Ntest_session = presentTest_session.sum() # number of trials in test
            llTest_per_session[s] = np.sum(np.log(ct_session[np.argwhere(presentTest_session==1)])) / Ntest_session # average test log-like per session

        Ntest = presentTest.sum() # number of trials in test        
        llTest = np.sum(np.log(ct[np.argwhere(presentTest==1)])) / Ntest # average test log-likelihood per trial

        gammaTest = gamma[np.argwhere(presentTest==1)].reshape((Ntest, self.k))
        phiTest = phi[np.argwhere(presentTest==1)].reshape((Ntest, self.k, self.c))
        yTest = y[np.argwhere(presentTest==1)].reshape((Ntest,))
        pChoice = np.zeros((gammaTest.shape[0], self.c)) # p(y_t | x_t, w_t)
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
            p = p[sortedStateInd,:][:,sortedStateInd]

        T = x.shape[0]

        if sessInd is None:
            sessInd = [0, T]
            sess = 1 # equivalent to saying the entire data set has one session
        else:
            sess = len(sessInd)-1 # total number of sessions 

        gamma = np.zeros((T, self.k)).astype(float) 
        phi = self.observation_probability(x, w)

        for s in range(0,sess):
            # E step - forward and backward passes given theta_old (= previous w and p)
            alphaSess, ctSess, llSess = self.forward_pass(y[sessInd[s]:sessInd[s+1]], present, p, pi, phi[sessInd[s]:sessInd[s+1],:,:])
            betaSess = self.backward_pass(y[sessInd[s]:sessInd[s+1]], present, p, phi[sessInd[s]:sessInd[s+1],:,:], ctSess)
            gammaSess, _ = self.posteriorLatents(y[sessInd[s]:sessInd[s+1]], present, p, phi[sessInd[s]:sessInd[s+1],:,:], alphaSess, betaSess, ctSess)
            
            # concatenating across sessions
            gamma[sessInd[s]:sessInd[s+1],:] = gammaSess[:,:] 
        
        return gamma
    
    # def all_states_weight_loss_function(self, currentW, x, y, gamma, prevW, nextW, sigma):
    #     '''
    #     weight loss function to optimize the weight in M-step of fitting function is calculated as negative of weighted log likelihood + prior terms 
    #     coming from drifting wrt neighboring sessions

    #     L(currentW) = sum_t sum_k gamma(z_t=k) * log p(y_t | z_t=k) + log P(currentW | prevW) + log P(currentW | nextW),
    #     where gamma matrix are fixed by old parameters but observation probabilities p(y_t | z_t=k) are updated with currentW

    #     Parameters
    #     ----------
    #     currentW: k x d numpy array
    #         weights of current session for C=0
    #     x: T x d numpy array
    #         design matrix
    #     y : T x 1 numpy vector 
    #         vector of observations with values 0,1,..,C-1
    #     gamma: T x k numpy array
    #         matrix of marginal posterior of latents p(z_t | y_1:T)
    #     prevW: k x d x c numpy array
    #         weights of previous session
    #     nextW: k x d x c numpy array
    #         weights of next session
    #     sigma: k x d numpy array
    #         std parameters of normal distribution for each state and each feature
        
    #     Returns
    #     ----------
    #     -lf: float
    #         loss function for currentW to be minimized
    #     '''
    #     # number of datapoints
    #     T = x.shape[0]

    #     # reshaping current session weights from flat to (T, k, d, c)
    #     # currentW = currentW._value.reshape((self.k, self.d))
    #     currentW = currentW.reshape((self.k, self.d))
    #     sessW = np.zeros((T, self.k, self.d, self.c))
    #     for t in range(0,T):
    #         sessW[t,:,:,1] = currentW[:,:]

    #     # log observation probability
    #     logPhi = self.log_observation_probability(x, sessW) # N x K x C phi matrix calculated with currentW
        
    #     # weighted log likelihood term of loss function
    #     lf = 0
    #     for t in range(0, T):
    #         lf += np.multiply(gamma[t,:],logPhi[t,:,y[t]]).sum()
        
    #     for k in range(0, self.k):
    #         # sigma=0 together with session indices [0,N] means usual GLM-HMM
    #         # inverse of covariance matrix
    #         invSigma = np.square(1/sigma[k,:])
    #         det = np.prod(invSigma)
    #         invCov = np.diag(invSigma)

    #         if (prevW is not None):
    #             # logpdf of multivariate normal (ignoring pi constant)
    #             lf +=  -1/2 * np.log(det) - 1/2 * (currentW[k,:] - prevW[k,:,1]).T @ invCov @ (currentW[k,:] - prevW[k,:,1])
    #         if (nextW is not None):
    #             # logpdf of multivariate normal (ignoring pi constant)
    #             lf += -1/2 * np.log(det) - 1/2 * (currentW[k,:] - nextW[k,:,1]).T @ invCov @ (currentW[k,:] - nextW[k,:,1])
                   
    #         # penalty term for size of weights - NOT NECESSARY FOR NOW
    #         #lf -= 1/2 * currentW[k,:].T @ currentW[k,:]

    #     return -lf


    # OLD SPLIT DATA FUNCTION
    # def split_data(self, x, y, sessInd, folds=10, random_state=1):
    #     ''' 
    #     splitting data function for cross-validation
    #     currently does not balance trials for each session

    #     Parameters
    #     ----------
    #     x: n x d numpy array
    #         full design matrix
    #     y : n x 1 numpy vector 
    #         full vector of observations with values 0,1,..,C-1
    #     sessInd: list of int
    #         indices of each session start, together with last session end + 1

    #     Returns
    #     -------
    #     trainX: folds x train_size x d numpy array
    #         trainX[i] has train data of i-th fold
    #     trainY: folds x train_size  numpy array
    #         trainY[i] has train data of i-th fold
    #     trainSessInd: list of lists
    #         trainSessInd[i] have session start indices for the i-th fold of the train data
    #     testX: folds x test_size x d numpy array
    #         testX[i] has test data of i-th fold
    #     testY: folds x test_size  numpy array
    #         testY[i] has test data of i-th fold
    #     testSessInd: list of lists
    #         testSessInd[i] have session start indices for the i-th fold of the test data
    #     '''
    #     # initializing test and train size based on number of folds
    #     train_size = int(self.n - self.n/folds)
    #     test_size = int(self.n/folds)

    #     # initializing input and output arrays for each folds
    #     trainY = np.zeros((folds, train_size)).astype(int)
    #     testY = np.zeros((folds, test_size)).astype(int)
    #     trainX = np.zeros((folds, train_size, self.d))
    #     testX = np.zeros((folds, test_size, self.d))

    #     # splitting data for each fold
    #     kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    #     for i, (train_index, test_index) in enumerate(kf.split(y)):
    #         trainY[i,:], testY[i,:] = y[train_index], y[test_index]
    #         trainX[i,:,:], testX[i,:,:] = x[train_index], x[test_index]
        
    #     # initializing session indices for each fold
    #     trainSessInd = [[0] for i in range(0, folds)]
    #     testSessInd = [[0] for i in range(0, folds)]

    #     # getting sesssion start indices for each fold
    #     for i, (train_index, test_index) in enumerate(kf.split(y)):
    #         for sess in range(1,len(sessInd)-1):
    #             testSessInd[i].append(np.argmin(test_index < sessInd[sess]))
    #             trainSessInd[i].append(np.argmin(train_index < sessInd[sess]))
    #         testSessInd[i].append(test_index.shape[0])
    #         trainSessInd[i].append(train_index.shape[0])
        
    #     return trainX, trainY, trainSessInd, testX, testY, testSessInd

  
# VERSION WITH JAX NUMPY

# # importing packages and modules
# import jax.numpy as jnp
# import numpy as np
# import scipy.stats as stats
# from scipy.optimize import minimize
# from utils import *
# from plotting_utils import *
# from scipy.stats import multivariate_normal
# from sklearn.model_selection import KFold
# # from autograd import value_and_grad, hessian
# from jax import value_and_grad, jit
# from warnings import simplefilter

# class dGLM_HMM1():
#     """
#     Class for fitting driftinig GLM-HMM model 1 in which weights are constant within session but vary across sessions
#     Code just works for c=2 at the moment
#     Weights for class c=1 are always kept to 0 (so then emission probability becomes 1/(1+exp(-wTx)))
#     X columns represent [bias, sensory] in this order

#     Notation: 
#         n: number of data points
#         k: number of states (states)
#         d: number of features (inputs to design matrix)
#         c: number of classes (possible observations)
#         X: design matrix (n x d)
#         Y: observations (n x c) or (n x 1)
#         w: weights mapping x to y (n x k x d x c)
#     """

#     def __init__(self, n, k, d, c):
#             self.n, self.k, self.d, self.c  = n, k, d, c
    
#     # Iris' code has reversed columns
#     def observation_probability(self, x, w):
#         '''
#         Calculating observation probabilities for given design matrix x and weight matrix w

#         Parameters
#         ----------
#         x: Ncurrent x D numpy array
#             input matrix
#         w: Ncurrent x K x D x C numpy array
#             weight matrix

#         Returns
#         -------
#         phi: Ncurrent x K x C numpy array
#             observation probabilities matrix
#         '''
        
#         Ncurrent = x.shape[0]

#         phi = jnp.empty((Ncurrent, self.k, self.c)) # probability that it is state 1
#         for k in range(0, self.k):
#             for c in range(0, self.c):
#                 phi = phi.at[:,k,c].set(jnp.exp(-jnp.sum(w[:,k,:,c]*x,axis=1)))
#             phi = phi.at[:,k,:].set(jnp.divide((phi[:,k,:]).T,jnp.sum(phi[:,k,:],axis=1)).T)     

#         return phi
    
#     def simulate_data(self, trueW, trueP, sessInd, save=False, title='sim', pi0=0.5):
#         '''
#         function that simulates X and Y data from true weights and true transition matrix

#         Parameters
#         ----------
#         trueW: n x k x d x c numpy array
#             true weight matrix. for c=2, trueW[:,:,:,1] = 0 
#         trueP: k x k numpy array
#             true probability transition matrix
#         priorZstart: int
#             0.5 probability of starting a session with state 0 (works for C=2)
#         sessInd: list of int
#             indices of each session start, together with last session end + 1
#         save: boolean
#             whether to save out simulated data
#         pi0: float
#             constant between 0 and 1, representing probability that first latent in a session is state 0
            
#         Returns
#         -------
#         x: n x d numpy array
#             simulated design matrix
#         y: n x 1 numpy array
#             simulated observation vector
#         z: n x 1 numpy array
#             simulated hidden states vector

#         '''
#         # check that weight and transition matrices are valid options
#         if (trueW.shape != (self.n, self.k, self.d, self.c)):
#             raise Exception(f'Weights need to have shape ({self.n}, {self.k}, {self.d}, {self.c})')
        
#         if (trueP.shape != (self.k, self.k)):
#             raise Exception(f'Transition matrix needs to have shape ({self.k}, {self.k})')
        
#         x = np.empty((self.n, self.d))
#         y = np.zeros((self.n, self.c)).astype(int)
#         z = np.empty((self.n,),dtype=int)

#         # input data x
#         x[:,0] = 1 # bias term
#         x[:,1] = stats.uniform.rvs(loc=-16,scale=33,size=self.n).astype(int)
#         # standardizing sensory info
#         x[:,1] = x[:,1] - x[:,1].mean()
#         x[:,1] = x[:,1] / x[:,1].std()

#         # TRY ormal distribution for x[:,1]

#         if (self.k==1):
#             z[:] = 0
#         elif (self.k ==2):
#             # latent variables z 
#             for t in range(0, self.n):
#                 if (t in sessInd[:-1]): # beginning of session has a new draw for latent
#                     z[t] = np.random.binomial(n=1,p=1-pi0)
#                 else:
#                     z[t] = np.random.binomial(n=1, p=trueP[z[t-1],1])
#         elif (self.k >=3):
#             raise Exception("simulate data does not support k>=3")
        
#         # observation probabilities
#         phi = self.observation_probability(x, trueW)

#         for t in range(0, self.n):
#             y[t,int(np.random.binomial(n=1,p=phi[t,z[t],1]))]=1
        
#         y = reshapeObs(y) # reshaping from n x c to n x 1

#         if (save==True):
#             np.save(f'../data/{title}X', x)
#             np.save(f'../data/{title}Y', y)
#             np.save(f'../data/{title}Z', z)

#         return x, y, z

#     # already checked with Iris' function that it is correct
#     def forward_pass(self, y, P, phi, pi0=None):
#         '''
#         Calculates alpha scaled as part of the forward-backward algorithm in E-step 
       
#         Parameters
#         ----------
#         y : T x 1 numpy vector 
#             vector of observations with values 0,1,..,C-1
#         P : k x k numpy array 
#             matrix of transition probabilities
#         phi : T x k x  c numpy array
#             matrix of observation probabilities
#         pi0: k x 1 numpy vector
#             distribution of first state before it has sesn any data 
#         Returns
#         -------
#         alpha : T x k numpy vector
#             matrix of the conditional probabilities p(z_t|x_{1:t},y_{1:t})
#         ct : T x 1 numpy vector
#             vector of the forward marginal likelihoods p(y_t | y_1:t-1)
#         ll : float
#             marginal log-likelihood of the data p(y)
#         '''
#         T = y.shape[0]
        
#         alpha = jnp.zeros((T, self.k)) # forward probabilities p(z_t | y_1:t)
#         alpha_prior = jnp.zeros((T, self.k)) # prior probabilities p(z_t | y_1:t-1)
#         lt = jnp.zeros((T, self.k)) # likelihood of data p(y_t|z_t)
#         ct = jnp.zeros(T) # forward marginal likelihoods p(y_t | y_1:t-1)

#         # forward pass calculations
#         for t in range(0,T):
#             lt = lt.at[t,:].set(phi[t,:,y[t]]) # likelihood p(y_t | z_t)
#             if (t==0): # time point 0
#                 # prior of z_0 before any data 
#                 if (pi0==None):
#                     alpha_prior = alpha_prior.at[0,:].set(jnp.divide(jnp.ones((self.k)),self.k)) # uniform prior
#                 else:
#                     alpha_prior = alpha_prior.at[0,:].set(pi0)
#             else:
#                 alpha_prior =alpha_prior.at[t,:].set(alpha[t-1,:].T @ P) # conditional p(z_t | y_1:t-1)
#             pxz = jnp.multiply(lt[t],alpha_prior[t,:]) # joint P(y_1:t, z_t)
#             ct = ct.at[t].set(jnp.sum(pxz)) # conditional p(y_t | y_1:t-1)
#             alpha = alpha.at[t,:].set(pxz/ct[t]) # conditional p(z_t | y_1:t)
        
#         ll = jnp.sum(jnp.log(ct)) # marginal log likelihood p(y_1:T) as sum of log conditionals p(y_t | y_1:t-1) 
        
#         return alpha, ct, ll
    
#     # already checked with Iris' function that it is correct
#     def backward_pass(self, y, P, phi, ct, pi0=None):
#         '''
#         Calculates beta scaled as part of the forward-backward algorithm in E-step 

#         Parameters
#         ----------
#         y : T x 1 numpy vector
#             vector of observations with values 0,1,..,C-1
#         p : k x k numpy array
#             matrix of transition probabilities
#         phi : T x k x c numppy array
#             matrix of observation probabilities
#         ct : T x 1 numpy vector 
#             veector of forward marginal likelihoods p(y_t | y_1:t-1), calculated at forward_pass
            
#         Returns
#         -------
#         beta: T x k numpy array 
#             matrix of backward conditional probabilities p(y_t+1:T | z_t) / p(y_t+1:T | y_1:t)
#         '''

#         T = y.shape[0]
        
#         beta = jnp.zeros((T, self.k)) # backward conditional probabilities p(y_t+1:T | z_t) / p(y_t+1:T | y_1:t)
#         lt = jnp.zeros((T, self.k)) # likelihood of data p(y_t|z_t)

#         # last time point
#         beta = beta.at[-1].set(1) # p(z_T=1)

#         # backward pass calculations
#         for t in jnp.arange(T-2,-1,-1):
#             lt = lt.at[t+1,:].set(phi[t+1,:,y[t+1]]) 
#             beta = beta.at[t,:].set(P @ (np.multiply(beta[t+1,:],lt[t+1,:])))
#             beta = beta.at[t,:].set(beta[t,:] / ct[t+1]) # scaling factor
        
#         return beta
    
#     def posteriorLatents(self, y, p, phi, alpha, beta, ct):
#         ''' 
#         calculates marginal posterior of latents gamma(z_t) = p(z_t | y_1:T)
#         and joint posterior of successive latens zeta(z_t, z_t+1) = p(z_t, z_t+1 | y_1:T)

#         Parameters
#         ----------
#         y : T x 1 numpy vector 
#             vector of observations with values 0,1,..,C-1
#         p : k x k numpy array
#             matrix of transition probabilities
#         phi : T x k x c numppy array
#             matrix of observation probabilities
#         alpha : T x k numpy vector
#             marix of the conditional probabilities p(z_t | x_1:t, y_1:t)
#         beta: T x k numpy array 
#             matrix of backward conditional probabilities p(y_t+1:T | z_t) / p(y_t+1:T | y_1:t)
#         ct : T x 1 numpy vector
#             vector of the forward marginal likelihoods p(y_t | y_1:t-1)
        
#         Returns
#         -------
#         gamma: T x k numpy array
#             matrix of marginal posterior of latents p(z_t | y_1:T)
#         zeta: T-1 x k x k 
#             matrix of joint posterior of successive latens p(z_t, z_t+1 | y_1:T)
#         '''
        
#         T = ct.shape[0]
#         gamma = jnp.empty((T, self.k)).astype(float) # marginal posterior of latents
#         zeta = jnp.empty((T-1, self.k, self.k)).astype(float) # joint posterior of successive latents

#         gamma = jnp.multiply(alpha, beta) # gamma(z_t) = alpha(z_t) * beta(z_t)

#         # zeta(z_t, z_t+1) =  alpha(z_t) * beta(z_t+1) * p (z_t+1 | z_t) * p(y_t+1 | z_t+1) / c_t+1
#         for t in range(0,T-1):
#             alpha_beta = alpha[t,:].reshape((self.k, 1)) @ beta[t+1,:].reshape((1, self.k))
#             zeta = zeta.at[t,:,:].set(np.multiply(alpha_beta,p)) 
#             zeta = zeta.at[t,:,:].set(np.multiply(zeta[t,:,:],phi[t+1,:,y[t+1]])) # change t+1 to t in phi to match Iris'
#             zeta = zeta.at[t,:,:].set(zeta[t,:,:] / ct[t+1])
            
#         return gamma, zeta

#     def get_states_in_time(self, x, y, w, p, sessInd=None):
#         ''' 
#         function that gets the distribution of states across trials/time-points after assigning the states with respective maximum probabilities
#         '''
#         T = x.shape[0]

#         if sessInd is None:
#             sessInd = [0, T]
#             sess = 1 # equivalent to saying the entire data set has one session
#         else:
#             sess = len(sessInd)-1 # total number of sessions 

#         # calculate observation probabilities given theta_old
#         phi = self.observation_probability(x, w)
        
#         zStates = np.empty((T)).astype(int)

#         for s in range(0,sess):
#         # E step - forward and backward passes given theta_old (= previous w and p)
#             alphaSess, ctSess, _ = self.forward_pass(y[sessInd[s]:sessInd[s+1]], p, phi[sessInd[s]:sessInd[s+1],:,:])
#             betaSess = self.backward_pass(y[sessInd[s]:sessInd[s+1]], p, phi[sessInd[s]:sessInd[s+1],:,:], ctSess)
#             gammaSess, _ = self.posteriorLatents(y[sessInd[s]:sessInd[s+1]], p, phi[sessInd[s]:sessInd[s+1],:,:], alphaSess, betaSess, ctSess)
            
#             # assigning max probabiity state to trial
#             zStates[sessInd[s]:sessInd[s+1]] = np.argmax(gammaSess,axis=1)
        
#         return zStates
    
#     def generate_param(self, sessInd, transitionDistribution=['dirichlet', (5, 1)], weightDistribution=['uniform', (-2,2)]):
#         ''' 
#         Function that generates parameters w and P and is used for initialization of parameters during fitting

#         Parameters
#         ----------
#         sessInd: list of int
#             indices of each session start, together with last session end + 1
#         transitionDistribution: list of length 2
#             first is str of distribution type, second is parameter tuple 
#                 dirichlet ditribution comes with (alphaDiagonal, alphaOther) the concentration values for either main diagonal or other locations
#         weightDistribution: list of length 2
#             first is str of distribution type, second is parameter tuple
#                 uniform distribution comes with (low, high) 
#                 normal distribution comes with (mean, std)

#         Returns
#         ----------
#         p: k x k numpy array
#             probability transition matrix
#         w: T x k x d x c numpy array
#             weight matrix. for c=2, trueW[:,:,:,1] = 0 
        
#         '''
#         T = int(sessInd[-1])

#         sess = len(sessInd)-1 # number of total sessions

#         # initialize weight and transitions
#         p = jnp.empty((self.k, self.k))
#         w = jnp.zeros((T, self.k, self.d, self.c))

#         # generating transition matrix 
#         if (transitionDistribution[0] == 'dirichlet'):
#             (alphaDiag, alphaOther) = transitionDistribution[1]
#             for k in range(0, self.k):
#                 alpha = jnp.full((self.k), alphaOther)
#                 alpha = alpha.at[k].set(alphaDiag) # concentration parameter of Dirichlet for row k
#                 p = p.at[k,:].set(np.random.dirichlet(alpha))
#         else:
#             raise Exception("Transition distribution can only be dirichlet")
        
#         # generating weight matrix
#         if (weightDistribution[0] == 'uniform'):
#             (low, high) = weightDistribution[1]
#             for s in range(0,sess):
#                 rv = np.random.uniform(low, high, (self.k, self.d))
#                 w = w.at[sessInd[s]:sessInd[s+1],:,:,0].set(rv)
#         elif (weightDistribution[0] == 'normal'):
#             (mean, std) = weightDistribution[1]
#             for s in range(0,sess):
#                 rv = np.random.normal(mean, std, (self.k, self.d))
#                 w = w.at[sessInd[s]:sessInd[s+1],:,:,0].set(rv)
#         else:
#             raise Exception("Weight distribution can only be uniform or normal")

#         return p, w 
    
#     def weight_loss_function(self, currentW, x, y, gamma, prevW, nextW, sigma):
#         '''
#         weight loss function to optimize the weight in M-step of fitting function is calculated as negative of weighted log likelihood + prior terms 
#         coming from drifting wrt neighboring sessions

#         L(currentW) = sum_t sum_k gamma(z_t=k) * log p(y_t | z_t=k) + log P(currentW | prevW) + log P(currentW | nextW),
#         where gamma matrix are fixed by old parameters but observation probabilities p(y_t | z_t=k) are updated with currentW

#         Parameters
#         ----------
#         currentW: k x d numpy array
#             weights of current session for C=0
#         x: T x d numpy array
#             design matrix
#         y : T x 1 numpy vector 
#             vector of observations with values 0,1,..,C-1
#         gamma: T x k numpy array
#             matrix of marginal posterior of latents p(z_t | y_1:T)
#         prevW: k x d x c numpy array
#             weights of previous session
#         nextW: k x d x c numpy array
#             weights of next session
#         sigma: k x d numpy array
#             std parameters of normal distribution for each state and each feature
        
#         Returns
#         ----------
#         -lf: float
#             loss function for currentW to be minimized
#         '''
#         # number of datapoints
#         T = x.shape[0]

#         # reshaping current session weights from flat to (T, k, d, c)
#         # currentW = currentW._value.reshape((self.k, self.d))
#         currentW = currentW.reshape((self.k, self.d))
#         sessW = jnp.zeros((T, self.k, self.d, self.c))
#         for t in range(0,T):
#             #sessW[t,:,:,0] = currentW[:,:]     # original numpy approach
#             sessW = sessW.at[t,:,:,0].set(currentW[:,:])      # jax numpy approach


#         phi = self.observation_probability(x, sessW) # N x K x C phi matrix calculated with currentW
#         logPhi = jnp.log(phi) # natural log of observation probabilities

#         # weighted log likelihood term of loss function
#         lf = 0
#         for t in range(0, T):
#             lf += jnp.multiply(gamma[t,:],logPhi[t,:,y[t]]).sum()
        
#         for k in range(0, self.k):
#             # sigma=0 together with session indices [0,N] means usual GLM-HMM
#             # inverse of covariance matrix
#             invSigma = jnp.square(1/sigma[k,:])
#             det = jnp.prod(invSigma)
#             invCov = jnp.diag(invSigma)

#             if (prevW is not None):
#                 # logpdf of multivariate normal (ignoring pi constant)
#                 lf +=  -1/2 * jnp.log(det) - 1/2 * (currentW[k,:] - prevW[k,:,0]).T @ invCov @ (currentW[k,:] - prevW[k,:,0])
#             if (nextW is not None):
#                 # logpdf of multivariate normal (ignoring pi constant)
#                 lf += -1/2 * jnp.log(det) - 1/2 * (currentW[k,:] - nextW[k,:,0]).T @ invCov @ (currentW[k,:] - nextW[k,:,0])
                   
#             # penalty term for size of weights - NOT NECESSARY FOR NOW
#             #lf -= 1/2 * currentW[k,:].T @ currentW[k,:]

#         return -lf

#     # def weight_loss_function_one_state(self, currentW, x, y, gamma, prevW, nextW, sigma):
#     #     '''
#     #     weight loss function to optimize the weight in M-step of fitting function is calculated as negative of weighted log likelihood + prior terms 
#     #     coming from drifting wrt neighboring sessions
        
#     #     just for one state

#     #     L(currentW) = sum_t sum_k gamma(z_t=k) * log p(y_t | z_t=k) + log P(currentW | prevW) + log P(currentW | nextW),
#     #     where gamma matrix are fixed by old parameters but observation probabilities p(y_t | z_t=k) are updated with currentW

#     #     Parameters
#     #     ----------
#     #     currentW: 1 x d numpy array
#     #         weights of current session for C=0
#     #     x: T x d numpy array
#     #         design matrix
#     #     y : T x 1 numpy vector 
#     #         vector of observations with values 0,1,..,C-1
#     #     gamma: T x 1 numpy array
#     #         matrix of marginal posterior of latents p(z_t | y_1:T)
#     #     prevW: 1 x d x c numpy array
#     #         weights of previous session
#     #     nextW: 1 x d x c numpy array
#     #         weights of next session
#     #     sigma: 1 x d numpy array
#     #         std parameters of normal distribution for each state and each feature
        
#     #     Returns
#     #     ----------
#     #     -lf: float
#     #         loss function for currentW to be minimized
#     #     '''
#     #     # number of datapoints
#     #     T = x.shape[0]

#     #     # reshaping current session weights from flat to (T, k, d, c)
#     #     # currentW = currentW._value.reshape((self.k, self.d))
#     #     sessW = np.zeros((T, 1, self.d, self.c))
#     #     for t in range(0,T):
#     #         sessW[t,:,:,0] = currentW[:,:]

#     #     phi = self.observation_probability(x, sessW) # N x K x C phi matrix calculated with currentW
#     #     logPhi = np.log(phi) # natural log of observation probabilities

#     #     # weighted log likelihood term of loss function
#     #     lf = 0
#     #     for t in range(0, T):
#     #         lf += np.multiply(gamma[t,:],logPhi[t,:,y[t]]).sum()
        
#     #     # sigma=0 together with session indices [0,N] means usual GLM-HMM
#     #     # inverse of covariance matrix
#     #     invSigma = np.square(1/sigma[1,:])
#     #     det = np.prod(invSigma)
#     #     invCov = np.diag(invSigma)

#     #     if (prevW is not None):
#     #         # logpdf of multivariate normal (ignoring pi constant)
#     #         lf +=  -1/2 * np.log(det) - 1/2 * (currentW[1,:] - prevW[1,:,0]).T @ invCov @ (currentW[1,:] - prevW[1,:,0])
#     #     if (nextW is not None):
#     #         # logpdf of multivariate normal (ignoring pi constant)
#     #         lf += -1/2 * np.log(det) - 1/2 * (currentW[1,:] - nextW[1,:,0]).T @ invCov @ (currentW[1,:] - nextW[1,:,0])
                   
#     #     # penalty term for size of weights - NOT NECESSARY FOR NOW
#     #      #lf -= 1/2 * currentW[k,:].T @ currentW[k,:]

#     #     return -lf
    
#     def fit(self, x, y,  initP, initW, sigma, sessInd=None, pi0=None, maxIter=250, tol=1e-3):
#         '''
#         Fitting function based on EM algorithm. Algorithm: observation probabilities are calculated with old weights for all sessions, then 
#         forward and backward passes are done for each session, weights are optimized for one particular session (phi stays the same),
#         then after all weights are optimized (in consecutive increasing order), the transition matrix is updated with the old zetas that
#         were calculated before weights were optimized

#         Parameters
#         ----------
#         x: T x d numpy array
#             design matrix
#         y : T x 1 numpy vector 
#             vector of observations with values 0,1,..,C-1
#         initP :k x k numpy array
#             initial matrix of transition probabilities
#         initW: n x k x d x c numpy array
#             initial weight matrix
#         sigma: k x d numpy array
#             st dev of normal distr for weights drifting over sessions
#         sessInd: list of int
#             indices of each session start, together with last session end + 1
#         pi0 : k x 1 numpy vector
#             initial k x 1 vector of state probabilities for t=1.
#         maxiter : int
#             The maximum number of iterations of EM to allow. The default is 300.
#         tol : float
#             The tolerance value for the loglikelihood to allow early stopping of EM. The default is 1e-3.
        
#         Returns
#         -------
#         p: k x k numpy array
#             fitted probability transition matrix
#         w: T x k x d x c numpy array
#             fitteed weight matrix
#         ll: float
#             marginal log-likelihood of the data p(y)
#         '''
#         # number of datapoints
#         T = x.shape[0]

#         if sessInd is None:
#             sessInd = [0, T]
#             sess = 1 # equivalent to saying the entire data set has one session
#         else:
#             sess = len(sessInd)-1 # total number of sessions 

#         # initialize weights and transition matrix
#         w = jnp.copy(initW)
#         p = jnp.copy(initP)

#         # initialize zeta = joint posterior of successive latents 
#         zeta = jnp.zeros((T-1, self.k, self.k)).astype(float) 
#         # initialize marginal log likelihood p(y)
#         ll = jnp.zeros((maxIter)).astype(float) 

#         #plotting_weights(initW, sessInd, 'initial weights')

#         for iter in range(maxIter):
            
#             print(iter)
            
#             # calculate observation probabilities given theta_old
#             phi = self.observation_probability(x, w)

#             # EM step for each session independently 
#             for s in range(0,sess):
                
#                 # E step - forward and backward passes given theta_old (= previous w and p)
#                 alphaSess, ctSess, llSess = self.forward_pass(y[sessInd[s]:sessInd[s+1]], p, phi[sessInd[s]:sessInd[s+1],:,:], pi0=pi0)
#                 betaSess = self.backward_pass(y[sessInd[s]:sessInd[s+1]], p, phi[sessInd[s]:sessInd[s+1],:,:], ctSess, pi0=pi0)
#                 gammaSess, zetaSess = self.posteriorLatents(y[sessInd[s]:sessInd[s+1]], p, phi[sessInd[s]:sessInd[s+1],:,:], alphaSess, betaSess, ctSess)
                
#                 # merging zeta for all sessions 
#                 zeta = zeta.at[sessInd[s]:sessInd[s+1]-1,:,:].set(zetaSess[:,:,:]) 
#                 ll = ll.at[iter].set(ll[iter] + llSess)
                
#                 # M step for weights - weights are updated for each session individually (as neighboring session weights have to be fixed)
#                 prevW = w[sessInd[s-1]] if s!=0 else None # k x d x c matrix of previous session weights
#                 nextW = w[sessInd[s+1]] if s!=sess-1 else None # k x d x c matrix of next session weights
                
#                 #w_flat = jnp.ndarray.flatten(w[sessInd[s],:,:,0]) # flatten weights for optimization 
#                 # optimized = minimize(self.weight_loss_function, w_flat, args=(x[sessInd[s]:sessInd[s+1]], y[sessInd[s]:sessInd[s+1]], gammaSess, prevW, nextW, sigma))
               
#                 # optimize loglikelihood given weights
#                 w_flat = jnp.ravel(w[sessInd[s],:,:,0])
#                 opt_log = lambda w: self.weight_loss_function(w,x[sessInd[s]:sessInd[s+1]], y[sessInd[s]:sessInd[s+1]], gammaSess, prevW, nextW, sigma) # calculate log likelihood 
#                 simplefilter(action='ignore', category=FutureWarning) # ignore FutureWarning generated by scipy
#                 optimized = minimize(jit(value_and_grad(opt_log)), w_flat, jac=True, method = "BFGS")
#                 optimizedW = jnp.reshape(optimized.x,(self.k, self.d)) # reshape optimized weights
#                 w = w.at[sessInd[s]:sessInd[s+1],:,:,0].set(optimizedW) # updating weight w for current session
            
#                 # optimizedW = np.zeros((self.k,self.d,self.c))
#                 # # prevW = w[sessInd[s-1]] if s!=0 else None # k x d x c matrix of previous session weights
#                 # # nextW = w[sessInd[s+1]] if s!=sess-1 else None # k x d x c matrix of next session weights
#                 # # optimized = minimize(self.weight_loss_function, w_flat, args=(x[sessInd[s]:sessInd[s+1]], y[sessInd[s]:sessInd[s+1]], gammaSess, prevW, nextW, sigma))
#                 # for k in range(0, self.k):
#                 #     prevW = w[sessInd[s-1],k,:,:] if s!=0 else None # k x d x c matrix of previous session weights
#                 #     nextW = w[sessInd[s+1],k,:,:] if s!=sess-1 else None # k x d x c matrix of next session weights
#                 #     w_flat = np.ndarray.flatten(w[sessInd[s],k,:,0]) # flatten weights for optimization 
#                 #     opt_log = lambda w: self.weight_loss_function_one_state(w, x[sessInd[s]:sessInd[s+1]], y[sessInd[s]:sessInd[s+1]], gammaSess[:,k], prevW, nextW, sigma[k]) # calculate log likelihood 
#                 #     optimized = minimize(value_and_grad(opt_log), w_flat) # , jac = "True", method = "L-BFGS-B")
#                 #     optimizedW[k,:,0] = np.reshape(optimized.x,(1, self.d)) # reshape optimized weights
#                 # w[sessInd[s]:sessInd[s+1],:,:,0] = optimizedW # updating weight w for current session
            
#             #plotting_weights(w, sessInd, f'iter {iter} optimized')
#             # print(w[sessInd[:-1]])

#             # M-step for transition matrix p - for all sessions together
#             for i in range(0, self.k):
#                 for j in range(0, self.k):
#                     p = p.at[i,j].set(zeta[:,i,j].sum()/zeta[:,i,:].sum()) # closed form update
        
#             # check if stopping early 
#             if (iter >= 10 and ll[iter] - ll[iter-1] < tol):
#                 break

#         return p, w, ll
    
#     def split_data(self, x, y, sessInd, folds=10, random_state=1):
#         ''' 
#         splitting data function for cross-validation
#         currently does not balance trials for each session

#         Parameters
#         ----------
#         x: n x d numpy array
#             full design matrix
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
#         # initializing test and train size based on number of folds
#         train_size = int(self.n - self.n/folds)
#         test_size = int(self.n/folds)

#         # initializing input and output arrays for each folds
#         trainY = np.zeros((folds, train_size)).astype(int)
#         testY = np.zeros((folds, test_size)).astype(int)
#         trainX = np.zeros((folds, train_size, self.d))
#         testX = np.zeros((folds, test_size, self.d))

#         # splitting data for each fold
#         kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
#         for i, (train_index, test_index) in enumerate(kf.split(y)):
#             trainY[i,:], testY[i,:] = y[train_index], y[test_index]
#             trainX[i,:,:], testX[i,:,:] = x[train_index], x[test_index]
        
#         # initializing session indices for each fold
#         trainSessInd = [[0] for i in range(0, folds)]
#         testSessInd = [[0] for i in range(0, folds)]

#         # getting sesssion start indices for each fold
#         for i, (train_index, test_index) in enumerate(kf.split(y)):
#             for sess in range(1,len(sessInd)-1):
#                 testSessInd[i].append(np.argmin(test_index < sessInd[sess]))
#                 trainSessInd[i].append(np.argmin(train_index < sessInd[sess]))
#             testSessInd[i].append(test_index.shape[0])
#             trainSessInd[i].append(train_index.shape[0])
        
#         return trainX, trainY, trainSessInd, testX, testY, testSessInd

  

        
