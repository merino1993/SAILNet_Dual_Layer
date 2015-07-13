# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:35:24 2015

@author: merino1993
"""

import numpy as np
import cPickle, time
from math import ceil
from Utils import tile_raster_images
from classtrial1 import Network
from classtrial1 import TwoLayerInference
from classtrial1 import Updates

rng = np.random.RandomState(0)

# Parameters
batch_size = 128
num_trials = 10000
#change num_trials to 10000, reduce batch_size 128

# Load Images
with open('images.pkl','rb') as f:
    images = cPickle.load(f)
imsize, imsize, num_images = images.shape
images = np.transpose(images,axes=(2,0,1))

BUFF = 20 #number of pixels outside of image that don't get sampled from for inputs

# Neuron Parameters
#Layer 1
N = 256 #number of inputs
sz = np.sqrt(N).astype(np.int)
OC1 = 6 #overcompleteness 
M1 = OC1*N #number of neurons

'''
If OC1 does not equal OC2 we get an error due to mismatching
'''

#Layer 2
sz = np.sqrt(N).astype(np.int)
OC2 = 6 #overcompleteness WITH RESPECT TO LAYER 1
M2 = OC2*N #number of neurons

# Network Parameters
p = .05 #p = target firing rate = # spikes per image

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

eta_ave = .3

Y_ave1 = p
Y_ave2 = p
Cyy_ave1 = p**2
Cyy_ave2 = p**2


# Zero timing variables
data_time = 0.
algo_time = 0.

# Begin Learning
X = np.zeros((batch_size,N))
reconstruction_error1 = np.zeros(num_trials) #want to keep track of during learning, run per batch
reconstruction_error2 = np.zeros(num_trials)
SNR_1=np.zeros(num_trials)
SNR_2=np.zeros(num_trials)

infer=TwoLayerInference(network)
updates=Updates(network, alpha, beta, gamma, p)

for tt in xrange(num_trials):
    # Extract image patches from images
    dt = time.time()
    for ii in xrange(batch_size):
        r = BUFF+int((imsize-sz-2.*BUFF)*rng.rand())
        c = BUFF+int((imsize-sz-2.*BUFF)*rng.rand())
        myimage = images[int(num_images*rng.rand()),r:r+sz,c:c+sz].ravel()
        myimage = myimage-np.mean(myimage)
        myimage = myimage/np.std(myimage)
        X[ii] = myimage
        
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
    reconstruction_error1[tt] = np.mean((X-Y1.dot(Q1.T))**2)/2.
    reconstruction_error2[tt] = np.mean((Y1-Y2.dot(Q2.T))**2)/2.
    
    SNR_1[tt] = np.var(X)/np.var(X-Y1.dot(Q1.T))
    SNR_2[tt] = np.var(Y1)/np.var(Y1-Y2.dot(Q2.T))    
    
    updates.update_inhibitory_weights(Cyy1, Cyy2)
    updates.update_feedforward_weights(Y1, Y2, X, batch_size)
    updates.update_thresholds(muY1, muY2, gamma, p, batch_size)

    Y_ave1 = (1.-eta_ave)*Y_ave1 + eta_ave*muY1
    Y_ave2 = (1.-eta_ave)*Y_ave2 + eta_ave*muY2
    Cyy_ave1=(1.-eta_ave)*Cyy_ave1 + eta_ave*Cyy1
    Cyy_ave2=(1.-eta_ave)*Cyy_ave2 + eta_ave*Cyy2
    
    if tt%100 == 0:
        print 'Batch: '+str(tt)+' out of '+str(num_trials)
        print 'Cumulative time spent gathering data: '+str(data_time)+' min'
        print 'Cumulative time spent in SAILnet: '+str(algo_time)+' min'
        print ''
        print 'print spiking rates of neuron in layer 1'
        print Y_ave1
        print ''
        print 'print spiking rates of neuron in layer 2'
        print Y_ave2
        #print 'correlation of timing in layer 1' Cyy_ave1   

        dt = time.time()-dt
        algo_time += dt/60.   
   
print ''        
   
with open('output.pkl','wb') as f:
    cPickle.dump((Q1,Q2,W1,W2,theta1,theta2,reconstruction_error1,reconstruction_error2, SNR_1, SNR_2),f)
     