import pandas as pd 
import numpy as np
from pathlib import Path
import math
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from scipy.optimize import minimize
from utils import *
from scipy.stats import multivariate_normal
import dynamic_glmhmm


colormap = sns.color_palette("viridis")
colors_dark = ['darkblue','darkred','darkgreen','darkgoldenrod']
colors_light = ['royalblue','indianred','limegreen','gold']
colorsFeatures = [['#FAA61A','indigo','#99CC66','#59C3C3','#9593D9'],['#FAA61A',"#2369BD","#A9373B",'#99CC66','#59C3C3','#9593D9']]
colorsStates = ['tab:orange','tab:blue', 'tab:green','tab:red']
# colorsStates = ["#2369BD",'#FAA61A',"#A9373B",'#99CC66','#59C3C3','#9593D9']
myFeatures = [['bias','delta stimulus', 'previous choice', 'previous reward'],['bias','contrast left','contrast right', 'previous choice', 'previous reward']]

def plotting_weights(w, sessInd, axes, trueW=None, title='', colors=colorsStates, sortedStateInd=None, size=24):
    ''' 
    Parameters
    __________
    w: N x K x D x C numpy array
        weight matrix (weights are fixed within one session)
    sessInd:
    title: str
        title of the plot
    Returns
    ________
    '''
    # permute weights 
    if (sortedStateInd is not None):
        w = w[:,sortedStateInd,:,:]

    sess = len(sessInd)-1
    for i in range(0,w.shape[1]):
        axes.plot(range(1,sess+1),w[sessInd[:-1],i,1,1],color=colorsStates[i],marker='o',label=f'state {i+1} sensory')
        axes.plot(range(1,sess+1),w[sessInd[:-1],i,0,1],color=colorsStates[i], marker='s',label=f'state {i+1} bias')

    axes.set_title(title, size=size)
    axes.set_xticks(range(1,sess+1))
    axes.set_ylabel("weights", size=size-2)
    axes.set_xlabel('session', size=size-2)
    axes.legend()

    if(trueW is not None):
        for i in range(0,trueW.shape[1]):
            axes.plot(range(1,sess+1),trueW[sessInd[:-1],i,1,1],color=colorsStates[i],marker='o',linestyle='dashed', label=f'true sensory {i+1}')
            axes.plot(range(1,sess+1),trueW[sessInd[:-1],i,0,1],color=colorsStates[i], marker='s',linestyle='dashed', label=f'true bias {i+1}')

def plotting_self_transition_probabilities(p, sessInd, axes, linewidth=5, linestyle='-o', title='', colorsStates=colorsStates, labels=[f'state {i+1}' for i in range(0,5)], sortedStateInd=None, size=24):
    ''' 
    
    Parameters
    __________
    
    Returns
    ________
    '''

    if (sortedStateInd is not None):
        p = p[:,sortedStateInd,:][:,:,sortedStateInd]

    sess = len(sessInd)-1
    for i in range(0,p.shape[1]):
        axes.plot(range(1,sess+1),p[sessInd[:-1],i,i],linestyle, linewidth=linewidth, color=colorsStates[i], label=labels[i], zorder=p.shape[1]-i)
        
    axes.set_title(title)
    axes.set_ylabel("self-transition probabilities", size=size-2)
    axes.set_ylim(0.6,1)
    axes.set_xlabel('session', size=size-2)

def plot_testLl_CV_sigma(testLl, sigmaList, label, color, axes, linestyle='-o', alpha=1, size=24, linewidth=1):
    '''  
    function that plots test LL as a function of sigma 

    Parameters
    ----------
    testLl: len(sigmaList) x 1 numpy array
        per trial log-likelihood on test data
    sigmaList: list

    '''
    sigmaListEven = [sigmaList[ind] for ind in range(len(sigmaList)) if ind%2==0]
    sigmaListOdd = [sigmaList[ind] for ind in range(len(sigmaList)) if ind%2==1] #+ [sigmaList[ind] for ind in range(11,len(sigmaList))]
    sigmaList3 = [sigmaList[ind] for ind in range(len(sigmaList)) if ind%3==1]
    # sigmaListOdd = [sigmaList[ind] for ind in range(11) if ind%2==1] #+ [sigmaList[ind] for ind in range(11,len(sigmaList))]
    axes.plot(np.log(sigmaList[1:]), testLl[1:], linestyle, color=color, label=label, alpha=alpha, linewidth=linewidth)
    if(sigmaList[0]==0):
        axes.scatter(-2 + np.log(sigmaList[1]), testLl[0], color=color, alpha=alpha)
        axes.set_xticks([-2 + np.log(sigmaList[1])]+list(np.log(sigmaList3)),['GLM-HMM'] + [f'{np.round(sigma,3)}' for sigma in sigmaList3])
    else:
        axes.scatter(np.log(sigmaList[0]), testLl[0], color=color, alpha=alpha)
        axes.set_xticks([np.log(sigmaListEven)],[f'{np.round(sigma, 1)}' for sigma in sigmaListEven])
    axes.set_ylabel("Test LL (per trial)", size=size-2)
    axes.set_xlabel("sigma", size=size-2)
    if (label is not None):
        axes.legend(loc='lower right')

def plot_testLl_CV_alpha(testLl, alphaList, label, color, axes, linestyle='-o', alpha=1, size=24):
    '''  
    function that plots test LL as a function of sigma 

    Parameters
    ----------
    testLl: len(sigmaList) x 1 numpy array
        per trial log-likelihood on test data
    sigmaList: list

    '''
    axes.plot(np.log10(alphaList), testLl[:-1], linestyle, color=color, label=label, alpha=alpha)
    alphaListEven = [alphaList[ind] for ind in range(0,len(alphaList),2)]
    axes.set_xticks(np.log10(alphaListEven),[f'{np.round(alpha,1)}' for alpha in alphaListEven])
    axes.set_ylabel('Test log-like (per trial)', size=size-2)
    axes.set_xlabel('alpha', size=size-2)
    axes.set_title('dGLM-HMM with varying transition matrix P', size=size)
    if (label is not None):
        axes.legend(loc='upper right')

# def plotting_weights_IBL(w, sessInd, axes, yLim, colors=None, labels=None, linewidth=5, linestyle='-', legend=True, sortedStateInd=None, size=24):

#     # permute weights 
#     if (sortedStateInd is not None):
#         w = w[:,sortedStateInd,:,:]

#     D = w.shape[2]

#     K = w.shape[1]
#     sess = len(sessInd)-1

#     if (K==1):
#         axes.axhline(0, alpha=0.3, color='black',linestyle='-')
#         for d in range(0, D):
#             axes.plot(range(1,sess+1),w[sessInd[:-1],0,d,1],color=colors[d],linewidth=linewidth,label=labels[d], alpha=0.8, linestyle=linestyle)
#         axes.set_ylabel("weights", size=size-2)
#         axes.set_xlabel('session', size=size-2)
#         axes.set_ylim(yLim)
#         axes.set_title(f'State 1', size=size)
#         axes.legend()
#     else:
#         for k in range(0,K):
#             axes[k].axhline(0, alpha=0.2, color='black',linestyle='-')
#             for d in range(0, D):
#                 axes[k].plot(range(1,sess+1),w[sessInd[:-1],k,d,1],color=colors[d],linewidth=linewidth,label=labels[d], alpha=0.8, linestyle=linestyle)
#             axes[k].set_ylim(yLim)
#             axes[k].set_ylabel("weights", size=size-2)
#             axes[k].set_title(f'State {k+1}', size=size)
#             if (legend==True):
#                 axes[k].legend()
#         axes[K-1].set_xlabel('session')

def plotting_weights_per_feature(w, sessInd, axes, trueW=None, yLim=[[-2.2,2.2],[-6.2,6.2]], colors=colorsStates, labels=myFeatures, linewidth=5, linestyle='-', alpha=0.9, legend=True, sortedStateInd=None, size=24):
    
    # permute weights 
    if (sortedStateInd is not None):
        w = w[:,sortedStateInd,:,:]

    D = w.shape[2]
    K = w.shape[1]
    sess = len(sessInd)-1

    for d in range(0,D):
        axes[d].axhline(0, alpha=0.2, color='gray',linestyle='-', linewidth=0.5)
        for k in range(0, K):
            if (legend==True):
                axes[d].plot(range(1,sess+1),w[sessInd[:-1],k,d,1],color=colors[k],linewidth=linewidth,label=f'state {k+1}', alpha=alpha, linestyle=linestyle)
            else:
                axes[d].plot(range(1,sess+1),w[sessInd[:-1],k,d,1],color=colors[k],linewidth=linewidth,label=None, alpha=alpha, linestyle=linestyle)
            if trueW is not None:
                axes[d].plot(range(1,sess+1),trueW[sessInd[:-1],k,d,1],color=colors[k],linewidth=linewidth,label=f'state {k+1}', alpha=alpha, linestyle='--')
            
        axes[d].set_ylim(yLim[d])
        axes[d].set_ylabel("weights", size=size-2)
        axes[d].set_title(f'{labels[d]}')
        if (legend==True):
            axes[d].legend(loc = 'center left', bbox_to_anchor=(0.99, 0.4))
    axes[D-1].set_xlabel('session', size=size-2)

def plot_constant_weights(w, axes, labels, colors):
    C = 2
    if (w.ndim == 3): # it means N=1 
        w = w.reshape((1,w.shape[0],w.shape[1],w.shape[2]))
        K = w.shape[1]
    elif (w.ndim == 4): # 
        K = w.shape[1]
    else:
        raise Exception("Weight matrix should have 3 or 4 dimensions (N X D x C or N x K x D x C)")
    
    for k in range(K):
        axes.plot(w[0, k, :, 1], marker='o', color=colors[k], label=f'state {k+1}', linewidth=2)
    
    axes.plot(labels,np.zeros((len(labels),1)),'k--')
    axes.set_xticks(np.arange(0,len(labels)))
    axes.set_xticklabels(labels,rotation=90)

def plot_transition_matrix(P, title='Recovered transition matrix', sortedStateInd=None):
    ''' 
    function that plots heatmap of transition matrix (assumed constant)

    Parameters
    ----------
    P: K x K numpy array
        transition matrix to be plotted 

    Returns
    ----------
    '''
    if (sortedStateInd is not None):
        P = P[sortedStateInd,:][:,sortedStateInd]

    plt.figure(figsize=(5, 4), dpi=400)
    K = P.shape[0] # 
    s = sns.heatmap(np.round(P,3),annot=True, vmin=-0.6, vmax=1,cmap='bone', fmt='g', linewidths=1, linecolor='black',clip_on=False, cbar=False)
    s.set(xlabel='state at time t+1', ylabel='state at time t', title=f'{title}', xticklabels=range(1,K+1), yticklabels=range(1,K+1))
    

def plot_posteior_latent(gamma, sessInd, axes, sessions = [10,20,30], linewidth=1, size=24):
    s = len(sessions)
    K = gamma.shape[1]
    for i in range(0,s):
        axes[i].set_title(f'session {sessions[i]+1}', size=size)
        axes[-1].set_xlabel('trials', size=size-2)
        axes[i].set_ylabel('posterior latent', size=size-2)
        for k in range(0,K):
            axes[i].plot(np.arange(sessInd[sessions[i]+1]-sessInd[sessions[i]]), gamma[sessInd[sessions[i]]:sessInd[sessions[i]+1],k], color=colorsStates[k], label=f'state {k+1}', linewidth=linewidth)
        axes[i].legend(loc = 'center left', bbox_to_anchor=(0.99, 0.4))

def plotting_psychometric(w, sessInd, session, axes, colorsStates, signedStimulus=True, title=f'session', linestyle='-', size=24, linewidth=1):
    ''' 
    '''
    N = w.shape[0]
    K = w.shape[1]
    D = w.shape[2]
    C = w.shape[3]

    if signedStimulus == True: # signed stimulus contrast = contrast Right - contrast Left
        d = 2
        x = np.ones((N, d)) # bias and delta stimulus only
        x[:,1] = np.linspace(-2,2,N)

        dGLMHMM = dynamic_glmhmm.dynamic_GLMHMM(N,K,D,C)
        phi = dGLMHMM.observation_probability(x, np.repeat(w[sessInd[session]][np.newaxis], N, axis=0)[:,:,:d,:])
        for k in range(K-1,-1,-1):
            axes.plot(np.linspace(-2,2,N), phi[:,k,1], color=colorsStates[k], linewidth=linewidth, label=f'state {k+1}', linestyle=linestyle)


    elif signedStimulus == False:
        d = 3 # bias, contrast right, contrast left
        x = np.zeros((N, d)) # bias and delta stimulus only
        x[:,0] = 1
        x[int(N/2)-1::-1,2] = np.linspace(0,2,int(N/2))
        x[int(N/2):,1] = np.linspace(0,2,N-int(N/2))
    
        dGLMHMM = dynamic_glmhmm.dynamic_GLMHMM(N,K,D,C)
        phi = dGLMHMM.observation_probability(x, np.repeat(w[sessInd[session]][np.newaxis], N, axis=0)[:,:,:d,:])
        for k in range(K-1,-1,-1):
            axes.plot(np.linspace(-2,2,N), phi[:,k,1], color=colorsStates[k], linewidth=linewidth, label=f'state {k+1}', linestyle=linestyle)

    axes.set_title(title, size=size)
    axes.set_ylim(-0.01,1.01)
    axes.set_ylabel('P(Right)', size=size-2)
    axes.set_xlabel('stimulus', size=size-2)
    axes.legend(loc='lower right')

def plot_state_occupancy_sessions(gamma, sessInd, axes, colors=colorsStates, linewidth=3, size=24):
    ''' 
    funcion that plots percentage of trials in each state across sessions
    '''
    K = gamma.shape[1]
    choiceHard = np.argmax(gamma, axis=1)
    count = np.zeros((len(sessInd)-1,K))
    for sess in range(0,len(sessInd)-1):
        for k in range(0,K):
            count[sess,k] = np.where(choiceHard[sessInd[sess]:sessInd[sess+1]] == k)[0].shape[0]/(sessInd[sess+1]-sessInd[sess]) * 100
    for k in range(0,K):
        axes.plot(range(1,len(sessInd)), count[:,k], color=colors[k], linewidth=linewidth, label=f'state {k+1}')
    axes.set_ylabel('% trial occupancy', size=size-2)
    axes.set_xlabel('session', size=size-2)
    axes.set_ylim(0,100)
    axes.legend(loc='upper right')

    return count

def plot_task_accuracy_states_sessions(gamma, y, correctSide, sessInd, axes, firstBlockSession=None, colors=colorsStates, linewidth=3):
    '''   
    function that plotts animal's task accuracy within each state across sessions, by hard assigning states and counting correct for animal
    '''
    K = gamma.shape[1]
    stateHard = np.argmax(gamma, axis=1)
    correct = np.zeros((len(sessInd)-1, K))
    for session in range(0, len(sessInd)-1):
        for t in range(sessInd[session],sessInd[session+1]):
            if (correctSide[t] == y[t]):
                correct[session, stateHard[t]] += 1
        for k in range(0,K):
            if (np.where(stateHard[sessInd[session]:sessInd[session+1]] == k)[0].shape[0] == 0):
                correct[session, k] = np.nan
            else:
                correct[session, k] = correct[session, k] / np.where(stateHard[sessInd[session]:sessInd[session+1]] == k)[0].shape[0] * 100
    for k in range(0,K):
        axes.plot(range(1,len(sessInd)), correct[:,k], color=colors[k], linewidth=linewidth, label=f'state {k+1}')
    axes.set_ylim(0,100)
    axes.set_ylabel('% task accuracy ')
    axes.set_xlabel('session')
    axes.legend(loc='lower right')
    if (firstBlockSession is not None):
        axes.axvline(firstBlockSession+1, color='gray', zorder=0)

    return correct

def barplot_task_accuracy(gamma, y, correctSide, sessInd, axes, session='all', colors=colorsStates):
    ''' 
    accuracy in each state by hard asssigning states and counting times correct for animal
    '''
    K = gamma.shape[1]
    stateHard = np.argmax(gamma, axis=1)
    correct = np.zeros((len(sessInd)-1, K))
    for sess in range(0, len(sessInd)-1):
        for t in range(sessInd[sess],sessInd[sess+1]):
            if (correctSide[t] == y[t]):
                correct[sess, stateHard[t]] += 1
        for k in range(0,K):
            correct[sess, k] = correct[sess, k] / np.where(stateHard[sessInd[sess]:sessInd[sess+1]] == k)[0].shape[0] * 100
    if (session=='all'): # task accuracy averaged across sessions
        axes.bar(['state 1','state 2','state 3'], np.nanmean(correct,axis=0), color=colorsStates) # mean calculation ignores nans
    else: 
        axes.bar(['state 1','state 2','state 3'], np.nan_to_num(correct[session]), color=colorsStates) # replacing nans with 0
    axes.set_ylim(45,100)
    axes.set_ylabel('% task accuracy ')

def plot_aligned_fraction_blocks_state(gamma, sessInd, biasedBlockTrials, biasedBlockSession, axes):
    '''  
    plotting for each biased state, the fraction of trials in that state in bias-aligned biased blocks vs bias-opposite
    '''
    stateHard = np.argmax(gamma, axis=1)
    blocksStateRight = np.zeros((len(sessInd)-1)) # fraction of right block given right state
    blocksStateLeft = np.zeros((len(sessInd)-1))
    for sess in range(0,len(sessInd)-1):
        # getting indices for right/left blocks and right/left trials for the current session
        rightBlocks = np.argwhere(biasedBlockTrials[sessInd[sess]:sessInd[sess+1]] == 1)
        rightBlocks = set([x for [x] in rightBlocks])
        rightStateTrials = np.argwhere(stateHard[sessInd[sess]:sessInd[sess+1]] == 1)
        rightStateTrials = set([x for [x] in rightStateTrials])
        leftBlocks = np.argwhere(biasedBlockTrials[sessInd[sess]:sessInd[sess+1]] == -1)
        leftBlocks = set([x for [x] in leftBlocks])
        leftStateTrials = np.argwhere(stateHard[sessInd[sess]:sessInd[sess+1]] == 2)
        leftStateTrials = set([x for [x] in leftStateTrials])

        if (biasedBlockSession[sess] == 1): # only for sessions with biased blocks 
            if (len(rightStateTrials.intersection(rightBlocks)) + len(rightStateTrials.intersection(leftBlocks)) > 0):
                blocksStateRight[sess] = len(rightStateTrials.intersection(rightBlocks)) / (len(rightStateTrials.intersection(rightBlocks)) + len(rightStateTrials.intersection(leftBlocks)))
            if (len(leftStateTrials.intersection(leftBlocks)) + len(leftStateTrials.intersection(rightBlocks)) > 0):
                blocksStateLeft[sess] = len(leftStateTrials.intersection(leftBlocks)) / (len(leftStateTrials.intersection(rightBlocks)) + len(leftStateTrials.intersection(leftBlocks)))
    blocksStateRight[np.argwhere(blocksStateRight==0)] = np.nan
    blocksStateLeft[np.argwhere(blocksStateLeft==0)] = np.nan
    axes.set_ylabel('fraction bias-aligned blocks')
    axes.set_xlabel('sessions')
    axes.set_ylim(0,1)
    axes.plot(range(1,len(sessInd)), blocksStateRight, '-o', color='forestgreen', label='state 2 - bias right')
    axes.plot(range(1,len(sessInd)), blocksStateLeft, '-o', color='gold', label = 'state 3 - bias left')
    axes.plot(range(1,len(sessInd)), np.nanmean(np.array([blocksStateRight,blocksStateLeft]),axis=0), color='black',label='mean')
    axes.legend()
    axes.set_xlim(0,len(sessInd)+1)
    axes.axhline(0.5,color='gray',linestyle='dashed')

    return blocksStateRight, blocksStateLeft

def distribution_most_likely_state(gamma, sessInd, axes, linewidth=2):
    maxState = gamma.argmax(axis=1)
    probMaxSession = np.zeros((len(sessInd)-1))
    for sess in range(0,len(sessInd)-1):
        for t in range(sessInd[sess],sessInd[sess+1]):
            probMaxSession[sess] += gamma[t,maxState[t]]
        probMaxSession[sess] = probMaxSession[sess] / (sessInd[sess+1]-sessInd[sess])
    axes.plot(range(1,len(sessInd)), probMaxSession, color='black', linewidth=linewidth)
    return probMaxSession

from datetime import date, datetime, timedelta
def IBL_plot_performance(dfAll, subject, axes, sessStop=-1):
    # code from Psytrack
    p = 5
    df = dfAll[dfAll['subject']==subject]   # Restrict data to the subject specified
    cL = np.tanh(p*df['contrastLeft'])/np.tanh(p)   # tanh transformation of left contrasts
    cR = np.tanh(p*df['contrastRight'])/np.tanh(p)  # tanh transformation of right contrasts
    inputs = dict(cL = np.array(cL)[:, None], cR = np.array(cR)[:, None])

    outData = dict(
        subject=subject,
        lab=np.unique(df["lab"])[0],
        contrastLeft=np.array(df['contrastLeft']),
        contrastRight=np.array(df['contrastRight']),
        date=np.array(df['date']),
        dayLength=np.array(df.groupby(['date','session']).size()),
        correct=np.array(df['feedbackType']),
        correctSide=np.array(df['correctSide']),
        probL=np.array(df['probabilityLeft']),
        inputs = inputs,
        y = np.array(df['choice'])
    )

    easy_trials = (outData['contrastLeft'] > 0.45).astype(int) | (outData['contrastRight'] > 0.45).astype(int)
    perf = []
    length = []
    for d in np.unique(outData['date']):
        date_trials = (outData['date'] == d).astype(int)
        inds = (date_trials * easy_trials).astype(bool)
        perf += [np.average(outData['correct'][inds])]
        length.append((outData['date'] == d).sum())

    dates = np.unique([datetime.strptime(i, "%Y-%m-%d") for i in outData['date']])
    dates = np.arange(len(dates)) + 1

    # My plotting function

    l1, = axes[0].plot(dates[:sessStop], perf[:sessStop], color="black", linewidth=1.5, zorder=2) # only look at first 25 days
    l2, = axes[1].plot(dates[:sessStop], length[:sessStop], color='gray', linestyle='--')
    # plt.scatter(dates[9], perf[9], c="white", s=30, edgecolors="black", linestyle="--", lw=0.75, zorder=5, alpha=1) # first session >50% accuracy has circle

    axes[0].axhline(0.5, color="black", linestyle="-", lw=1, alpha=0.3, zorder=0)

    # axes[0].set_xticks(np.arange(0,sessStop+1,5))
    axes[0].set_yticks([0.4,0.6,0.8,1.0])
    axes[0].set_ylim(0.2,1.0)
    axes[1].set_ylim(100,1500)
    # axes[0].set_xlim(1, sessStop + .5)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.title('Accuracy and session length for IBL mouse ' + subject)
    axes[0].set_xlabel("days of training")
    axes[0].set_ylabel('accuracy on easy trials')
    axes[1].set_ylabel('number of trials')
    axes[0].legend([l1, l2], ["% correct", "# trials"])
    plt.subplots_adjust(0,0,1,1) 

# OLD FUNCTION
# def sigma_CV_testLl_plot_PWM(rat_id, stage_filter, K, folds, sigmaList, axes, title='', labels=None, color=0, linestyle='solid', penaltyW=False, save_fig=False):
#     ''' 
#     function for plotting the test LL vs sigma scalars for PWM real data
#     '''     
    
#     colormap = sns.color_palette("viridis")
#     sigmaListEven = [sigmaList[ind] for ind in range(len(sigmaList)) if ind%2==0]
#     sigmaListOdd = [sigmaList[ind] for ind in range(11) if ind%2==1] + [sigmaList[ind] for ind in range(11,len(sigmaList))]
#     for fold in range(0, folds):
#         testLl = np.load(f'../data_PWM/testLl_PWM_{rat_id}_sf={stage_filter}_{K}_state_fold-{fold}_multiple_sigmas_penaltyW={penaltyW}.npy')
#         axes.set_title(title)
#         # axes.scatter(np.log(sigmaList[1:]), testLl[1:], color=colormap[color+fold])
#         axes.plot(np.log(sigmaList[1:]), testLl[1:], '-o', color=colormap[color+fold], linestyle=linestyle, label=labels[fold])
#         if(sigmaList[0]==0):
#             axes.scatter(-2 + np.log(sigmaList[1]), testLl[0], color=colormap[color+fold])
#             if (K==1):
#                 axes.set_xticks([-2 + np.log(sigmaList[1])]+list(np.log(sigmaListOdd)),['GLM'] + [f'{np.round(sigma,4)}' for sigma in sigmaListOdd])
#             else:
#                 axes.set_xticks([-2 + np.log(sigmaList[1])]+list(np.log(sigmaListOdd)),['GLM-HMM'] + [f'{np.round(sigma,4)}' for sigma in sigmaListOdd])
#         else:
#             axes.scatter(np.log(sigmaList[0]), testLl[0], color=colormap[color+fold])
#             axes.set_xticks([np.log(sigmaListEven)],[f'{np.round(sigma,4)}' for sigma in sigmaListEven])
#         axes.set_ylabel("Test LL (per trial)")
#         axes.set_xlabel("sigma")
#     axes.legend()

#     if(save_fig==True):
#         plt.savefig(f'../figures/Sigma_vs_TestLl-{title}.png', bbox_inches='tight', dpi=400)


# OLD FUNCTION
# def sigma_testLl_plot(K, sigmaList, testLl, axes, title='', labels=None, color=0, save_fig=False):
#     ''' 
#     function for plotting the test LL vs sigma scalars
#     '''
#     inits = testLl.shape[0] # for mutiple initiaizations/models 
#     colormap = sns.color_palette("viridis")
#     sigmaListEven = [sigmaList[ind] for ind in range(len(sigmaList)) if ind%2==0]
#     if (len(sigmaList)>=17):
#         sigmaListOdd = [sigmaList[ind] for ind in range(len(sigmaList)-4) if ind%2==1] + [sigmaList[ind] for ind in range(17,len(sigmaList))]
#     else:
#         sigmaListOdd = [sigmaList[ind] for ind in range(len(sigmaList)) if ind%2==1]
#     flag = 0
#     if (labels is None):
#         labels = ['' for init in range(0,inits)]
#         flag = 1
#     for init in range(0,inits):
#         axes.set_title(title)
#         axes.scatter(np.log(sigmaList[1:]), testLl[init,1:], color=colormap[color])
#         axes.plot(np.log(sigmaList[1:]), testLl[init,1:], color=colormap[color])
#         if(sigmaList[0]==0):
#             axes.scatter(-1 + np.log(sigmaList[1]), testLl[init,0], color=colormap[color], label=labels[init])
#             if (K==1):
#                 axes.set_xticks([-1 + np.log(sigmaList[1])]+list(np.log(sigmaListOdd)),['GLM'] + [f'{np.round(sigma,3)}' for sigma in sigmaListOdd])
#             else:
#                 axes.set_xticks([-1 + np.log(sigmaList[1])]+list(np.log(sigmaListOdd)),['GLM-HMM'] + [f'{np.round(sigma,3)}' for sigma in sigmaListOdd])
#         else:
#             axes.scatter(np.log(sigmaList[0]), testLl[init,0], color=colormap[color], label=f'init {init}')
#             axes.set_xticks([np.log(sigmaListEven)],[f'{np.round(sigma,2)}' for sigma in sigmaListEven])
#     axes.set_ylabel("Test LL (per trial)")
#     axes.set_xlabel("sigma")
#     if (flag == 0):
#         axes.legend()

#     if(save_fig==True):
#         plt.savefig(f'../figures/Sigma_vs_TestLl-{title}.png', bbox_inches='tight', dpi=400)

# def plotting_weights_PWM(w, sessInd, axes, sessStop=None, yLim=[-3,3,-1,1], title='', save_fig=False):

#     # permute weights accordinng to highest sensory
#     sortedStateInd = get_states_order(w, sessInd)
#     w = w[:,sortedStateInd,:,:]

#     K = w.shape[1]
#     sess = len(sessInd)-1

#     if (K==1):
#         axes[0].axhline(0, alpha=0.3, color='black',linestyle='--')
#         axes[1].axhline(0, alpha=0.3, color='black',linestyle='--')
#         if (sessStop==None):
#             axes[1].plot(range(1,sess+1),w[sessInd[:-1],0,4,1],color='#59C3C3', linewidth=5, label='previous choice', alpha=0.8)
#             axes[1].plot(range(1,sess+1),w[sessInd[:-1],0,5,1],color='#9593D9',linewidth=5, label='previous correct', alpha=0.8)
#             axes[1].plot(range(1,sess+1),w[sessInd[:-1],0,3,1],color='#99CC66',linewidth=5, label='previous stim', alpha=0.8)
#             axes[0].plot(range(1,sess+1),w[sessInd[:-1],0,1,1],color="#A9373B",linewidth=5,label='stim A', alpha=0.8)
#             axes[0].plot(range(1,sess+1),w[sessInd[:-1],0,0,1],color='#FAA61A',linewidth=5, label='bias', alpha=0.8)
#             axes[0].plot(range(1,sess+1),w[sessInd[:-1],0,2,1],color="#2369BD",linewidth=5, label='stim B', alpha=0.8)
#         else:
#             axes[1].plot(range(1,sessStop+1),w[sessInd[:sessStop],0,4,1],color='#59C3C3', linewidth=5, label='previous choice', alpha=0.8)
#             axes[1].plot(range(1,sessStop+1),w[sessInd[:sessStop],0,5,1],color='#9593D9',linewidth=5, label='previous correct', alpha=0.8)
#             axes[1].plot(range(1,sessStop+1),w[sessInd[:sessStop],0,3,1],color='#99CC66',linewidth=5, label='previous stim', alpha=0.8)
#             axes[0].plot(range(1,sessStop+1),w[sessInd[:sessStop],0,1,1],color="#A9373B",linewidth=5,label='stim A', alpha=0.8)
#             axes[0].plot(range(1,sessStop+1),w[sessInd[:sessStop],0,0,1],color='#FAA61A',linewidth=5, label='bias', alpha=0.8)
#             axes[0].plot(range(1,sessStop+1),w[sessInd[:sessStop],0,2,1],color="#2369BD",linewidth=5, label='stim B', alpha=0.8)

#         #axes[0].set_title(title)
#         axes[0].set_ylabel("weights")
#         axes[0].set_xlabel('session')
#         # axes[0].set_yticks([-2,0,2])
#         axes[0].set_ylim(yLim[0:2])
#         axes[1].set_ylim(yLim[2:4])
#         # axes[1].set_yticks([0,2])
#         #axes[1].set_ylabel("weights")
#         axes[1].set_xlabel('session')

#         axes[0].legend()
#         axes[1].legend()

#     elif(K >= 2):
#         for i in range(0,K):
#             axes[i,0].axhline(0, alpha=0.3, color='black',linestyle='--')
#             axes[i,1].axhline(0, alpha=0.3, color='black',linestyle='--')
#             if (sessStop==None):
#                 axes[i,1].plot(range(1,sess+1),w[sessInd[:-1],i,4,1],color='#59C3C3', linewidth=5, label='previous choice', alpha=0.8)
#                 axes[i,1].plot(range(1,sess+1),w[sessInd[:-1],i,5,1],color='#9593D9',linewidth=5, label='previous correct', alpha=0.8)
#                 axes[i,1].plot(range(1,sess+1),w[sessInd[:-1],i,3,1],color='#99CC66',linewidth=5, label='previous stim', alpha=0.8)
#                 axes[i,0].plot(range(1,sess+1),w[sessInd[:-1],i,1,1],color="#A9373B",linewidth=5,label='stim A', alpha=0.8)
#                 axes[i,0].plot(range(1,sess+1),w[sessInd[:-1],i,0,1],color='#FAA61A',linewidth=5, label='bias', alpha=0.8)
#                 axes[i,0].plot(range(1,sess+1),w[sessInd[:-1],i,2,1],color="#2369BD",linewidth=5, label='stim B', alpha=0.8)
#             else:
#                 axes[i,1].plot(range(1,sessStop+1),w[sessInd[:sessStop],i,4,1],color='#59C3C3', linewidth=5, label='previous choice', alpha=0.8)
#                 axes[i,1].plot(range(1,sessStop+1),w[sessInd[:sessStop],i,5,1],color='#9593D9',linewidth=5, label='previous correct', alpha=0.8)
#                 axes[i,1].plot(range(1,sessStop+1),w[sessInd[:sessStop],i,3,1],color='#99CC66',linewidth=5, label='previous stim', alpha=0.8)
#                 axes[i,0].plot(range(1,sessStop+1),w[sessInd[:sessStop],i,1,1],color="#A9373B",linewidth=5,label='stim A', alpha=0.8)
#                 axes[i,0].plot(range(1,sessStop+1),w[sessInd[:sessStop],i,0,1],color='#FAA61A',linewidth=5, label='bias', alpha=0.8)
#                 axes[i,0].plot(range(1,sessStop+1),w[sessInd[:sessStop],i,2,1],color="#2369BD",linewidth=5, label='stim B', alpha=0.8)

#             axes[i,0].set_title(f'State {i+1}')
#             axes[i,1].set_title(f'State {i+1}')
#             axes[i,0].set_ylabel("weights")
#             axes[i,0].set_ylim(yLim[0:2])
#             axes[i,1].set_ylim(yLim[2:4])
#             # axes[i,0].set_yticks([-2,0,2])
#             # axes[i,0].set_ylim(-3,3)
#             # axes[i,1].set_ylim(-0.2,2.1)
#             # axes[i,1].set_yticks([0,2])
#             #axes[1].set_ylabel("weights")
#             if (i==K-1):
#                 axes[i,0].set_xlabel('session')
#                 axes[i,1].set_xlabel('session')
#             axes[i,0].legend(loc='upper right')
#             axes[i,1].legend(loc='upper right')

#     if(save_fig==True):
#         plt.savefig(f'../figures/Weights_PWM_{title}.png', bbox_inches='tight', dpi=400)
    