# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:26:16 2015

@author: merino1993
"""

import numpy as np
from Utils import tile_raster_images

class Network(object):
    def __init__(self, feedforward_weights, inhibitory_weights, thresholds):
        self.feedforward_weights = feedforward_weights
        self.inhibitory_weights = inhibitory_weights
        self.thresholds = thresholds
    def get_feedforward_weights(self):
        return self.feedforward_weights
    def get_inhibitory_weights(self):
        return self.inhibitory_weights
    def get_thresholds(self):   
        return self.thresholds
        
class BaseInference(object):
    def __init__(self, network):
        self.network = network
    def activities(self, inputs):
        raise NotImplementedError
        
class TwoLayerInference(BaseInference):
    def activities(self, inputs):
        batch_size, N = inputs.shape
        sz = int(np.sqrt(N))
        Q1, Q2 = self.network.get_feedforward_weights()
        W1, W2 = self.network.get_inhibitory_weights()        
        theta1, theta2 = self.network.get_thresholds() 
        
        M1 = Q1, Q2.shape[1] #dimension N x M
        M2 = Q2.shape[1] #dimension M1 x M2
    
        num_iterations = 17
        
        eta = .1
    
        B1 = inputs.dot(Q1)
        B2 = np.zeros((batch_size,M1)) #initialise since aas hasn't been defined yet, need to re-update after each step since inputs for layer 2 are the changing spiking patterns
    
        T1 = np.tile(theta1,(batch_size,1))
        T2 = np.tile(theta2,(batch_size,1))
    
        Ys1 = np.zeros((batch_size,M1))
        aas1 = np.zeros((batch_size,M1)) #spiking patterns from Layer 1
        Y1 = np.zeros((batch_size,M1))
        
        Ys2 = np.zeros((batch_size,M2))
        aas2 = np.zeros((batch_size,M2)) #spiking patterns from Layer 2
        Y2 = np.zeros((batch_size,M2))
        
        for tt in xrange(num_iterations):
            Ys1 = (1.-eta)*Ys1+eta*(B1-aas1.dot(W1))
            Ys2 = (1.-eta)*Ys2+eta*(B2-aas2.dot(W2))
            aas1 = np.zeros((batch_size,M1))
            aas1[Ys1 > T1] = 1.
            Y1 += aas1
            Ys1[Ys1 > T1] = 0.
            
            aas2 = np.zeros((batch_size,M2))
            aas2[Ys2 > T2] = 1.
            Y2 += aas2
            Ys2[Ys2 > T2] = 0.
            
            B2 = aas1.dot(Q2) #weights that determine how strongly layer 2 neurons excites layer 1 neurons
    
        return Y1, Y2
        
class Updates(object):
    def __init__(self, network, alpha, beta, gamma, p):
        self.network=network
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.p=p
    def update_inhibitory_weights(self, Cyy1, Cyy2):
        dW1 = self.alpha*(Cyy1-self.p**2)
        W1=self.network.inhibitory_weights        
        W1 += dW1
        W1 = W1-np.diag(np.diag(W1))
        W1[W1 < 0] = 0.
        self.network.inhibitory_weights=W1
        
        dW2 = self.alpha*(Cyy2-self.p**2)
        W2=self.network.inhibitory_weights       
        W2 += dW2
        W2 = W2-np.diag(np.diag(W2))
        W2[W2 < 0] = 0.
        self.network.inhibitory_weights=W2
    def update_feedforward_weights(self, Y1, Y2, X):
        square_act1 = np.sum(Y1*Y1,axis=0)
        square_act2 = np.sum(Y2*Y2,axis=0)
        mymat1 = np.diag(square_act1)
        mymat2 = np.diag(square_act2)
        dQ1 = beta*X.T.dot(Y1)/batch_size - beta*Q1.dot(mymat1)/batch_size
        self.network.feedforward_weights += dQ1
        dQ2 = beta*Y1.T.dot(Y2)/batch_size - beta*Q2.dot(mymat2)/batch_size
        self.network.feedforward_weights += dQ2
    def update_thresholds(self, Y1, Y2):
        dtheta1 = gamma*(np.sum(Y1,axis=0)/batch_size-p)
        dtheta2 = gamma*(np.sum(Y2,axis=0)/batch_size-p)
        self.network.thresholds += dtheta1
        self.network.thresholds += dtheta2


    
'''Get new code to run, don't worry about the output'''
        
'''
1) Make all significant, but nonchanging, parts of the 
two layer code into separate classes for example updates
and leave the changing parts alone to avoid having to go
to a separate file instead of manually changing just one
file. Should only have two more to do, updates and figures
'''
        
'''
2) Once all classes are done, make changes to the original
two layer model script by replacing all the chunks of code 
that are already made into classes to make it look nicer.
'''
        
'''
3) Get code to run/work
'''
        
'''
4) Bonus: Get code to create exactly what the original 
two layer model script outputs
'''