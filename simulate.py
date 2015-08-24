# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:35:24 2015

@author: merino1993
"""

import numpy as np
import cPickle, time
from math import ceil
from Utils import tile_raster_images
from classes import Network
from classes import TwoLayerInference
from classes import Updates
from classes import Data

rng=np.random.RandomState(0)

# Parameters
batch_size = 128
num_trials = 50000

filename='images.pkl'
patch_size = (16,16)
data=Data(filename, patch_size, seed=20150727)

# Neuron Parameters
#Layer 1
N = 256
'ask Jesse how to call N from classes using command of syntax data.get_batch(batch_size)'
OC1 = 6 #overcompleteness 
M1 = OC1*N #number of neurons

'''
If OC1 does not equal OC2 we get an error due to mismatching
'''

#Layer 2

OC2 = 6 #overcompleteness WITH RESPECT TO LAYER 1
M2 = OC2*N #number of neurons

# Network Parameters
p = .05 #p = target firing rate = # spikes per image

# Load Images
with open('images.pkl','rb') as f:
    images = cPickle.load(f)
imsize, imsize, num_images = images.shape
images = np.transpose(images,axes=(2,0,1))

# BUFF = 20 (number of pixels outside of image that don't get sampled from for inputs)

# Initialize Weights
Q1 = rng.randn(N,M1)
Q1 = Q1.dot(np.diag(1./np.sqrt(np.diag(Q1.T.dot(Q1)))))

Q2 = rng.randn(M1,M2)
Q2 = Q2.dot(np.diag(1./np.sqrt(np.diag(Q2.T.dot(Q2)))))

W1 = np.zeros((M1,M1))
W2 = np.zeros((M2,M2))

"""
Neuron has two 'memories': Q, W
    Q = weight of feed-forward connection weights (N entries in total, 1 per pixel)
    W = weight of feed-back connection from neuron j to neuron i (N-1 entries in total, 1 per neighbouring neuron)
"""

theta1 = 0.5*np.ones(M1)
theta2 = 0.5*np.ones(M2)

"""
Theta's = orientation within an image patch = direction of the axis with smallest moment of inertia
"""
network=Network([Q1, Q2], [W1, W2], [theta1, theta2])

# Learning Rates
alpha = 1.
beta = .02
gamma = .12

# Zero timing variables
data_time = 0.
algo_time = 0.

# Begin Learning

reconstruction_error = np.zeros((2, num_trials)) #want to keep track of during learning, run per batch
SNR = np.zeros((2, num_trials))
SNR_norm = np.zeros((2, num_trials))
Q_norm_mean = np.zeros((2, num_trials))
Q_norm_std = np.zeros((2, num_trials))

infer=TwoLayerInference(network)
updates=Updates(network, alpha, beta, gamma, p)

for tt in xrange(num_trials):
    # Extract image patches from images
    dt = time.time()
    X = data.get_batch(batch_size)    
    dt = time.time()-dt
    data_time += dt/60.
    
    dt = time.time()
    # Calculate network activities
    Y1,Y2 = infer.activities(X)
    muY1 = np.mean(Y1,axis=0)
    muY2 = np.mean(Y2,axis=0)
    Cyy1 = Y1.T.dot(Y1)/batch_size
    Cyy2 = Y2.T.dot(Y2)/batch_size
    Q1, Q2= network.feedforward_weights
      
    #reconstruction error
    X_rec = Y1.dot(Q1.T)
    Y1_rec = Y2.dot(Q2.T)
    X_norm = (X*X).sum(1, keepdims=True)
    Y1_norm = (Y1*Y1).sum(1, keepdims=True)
    X_rec_norm = (X_rec*X_rec).sum(1, keepdims=True)
    Y1_rec_norm = (Y1_rec*Y1_rec).sum(1, keepdims=True)
    X_rec_normed = X_rec*X_norm/X_rec_norm
    Y1_rec_normed = Y1_rec*Y1_norm/Y1_rec_norm

    reconstruction_error[0,tt] = np.mean((X-X_rec)**2)/2.
    reconstruction_error[1,tt] = np.mean((Y1-Y1_rec)**2)/2.
    SNR[0,tt] = (X.var(0)/(X-X_rec).var(0)).mean()
    SNR[1,tt] = (Y1.var(0)/(Y1-Y1_rec).var(0)).mean()
    SNR_norm[0,tt] = (X.var(0)/(X-X_rec_normed).var(0)).mean()
    SNR_norm[1,tt] = (Y1.var(0)/(Y1-Y1_rec_normed).var(0)).mean()
    Q_norm_mean[0,tt] = (Q1*Q1).sum(0).mean()
    Q_norm_mean[1,tt] = (Q2*Q2).sum(0).mean()
    Q_norm_std[0,tt] = (Q1*Q1).sum(0).std()
    Q_norm_std[1,tt] = (Q2*Q2).sum(0).std()
    
    updates.update_inhibitory_weights(Cyy1, Cyy2)
    updates.update_feedforward_weights(Y1, Y2, X, batch_size)
    updates.update_thresholds(muY1, muY2, gamma, p, batch_size)
    dt = time.time()-dt
    algo_time += dt/60.   

    if tt%100 == 0:
        print 'Batch: '+str(tt)+' out of '+str(num_trials)
        print 'Cumulative time spent gathering data: '+str(data_time)+' min'
        print 'Cumulative time spent in SAILnet: '+str(algo_time)+' min'
        print 'SNR1: '+str(SNR[0,tt])
        print 'SNR2: '+str(SNR[1,tt])
        print 'SNR1_norm: '+str(SNR_norm[0,tt])
        print 'SNR2_norm: '+str(SNR_norm[1,tt])
        print ''

   
print ''        
   
with open('output.pkl','wb') as f:
    cPickle.dump((network,reconstruction_error, SNR, SNR_norm, Q_norm_mean, Q_norm_std),f)
     
