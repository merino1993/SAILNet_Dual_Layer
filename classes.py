# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:26:16 2015

@author: merino1993
"""

import numpy as np
from Utils import tile_raster_images
import cPickle

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
        
        M1 = Q1.shape[1] #dimension N x M
        M2 = Q2.shape[1] #dimension M1 x M2

        #c = M1.ravel()
        #d = M2.ravel()
        num_iterations = 50
        
        eta = .1
    
        B1 = inputs.dot(Q1)
       
        #batch_size,M1
        T1 = np.tile(theta1,(batch_size,1))
        T2 = np.tile(theta2,(batch_size,1))
    
        Ys1 = np.zeros((batch_size,M1))
        #batch_size,M1
        aas1 = np.zeros((batch_size,M1)) #spiking patterns from Layer 1
        #batch_size,M1        
        Y1 = np.zeros((batch_size,M1))
        #batch_size,M1
        
        Ys2 = np.zeros((batch_size,M2))
        aas2 = np.zeros((batch_size,M2)) #spiking patterns from Layer 2
        Y2 = np.zeros((batch_size,M2))
        
        for tt in xrange(num_iterations):
            Ys1 = (1.-eta)*Ys1+eta*(B1-aas1.dot(W1))
            
            aas1 = np.zeros((batch_size,M1))
            aas1[Ys1 > T1] = 1.
            Y1 += aas1
            Ys1[Ys1 > T1] = 0.
        B2 = aas1.dot(Q2) #weights that determine how strongly layer 2 neurons excites layer 1 neurons
        for tt in xrange(num_iterations):
            Ys2 = (1.-eta)*Ys2+eta*(B2-aas2.dot(W2))
            aas2 = np.zeros((batch_size,M2))
            aas2[Ys2 > T2] = 1.
            Y2 += aas2
            Ys2[Ys2 > T2] = 0.
            
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
        W1, W2 =self.network.inhibitory_weights   
        W1 += dW1
        W1 = W1-np.diag(np.diag(W1))
        W1[W1 < 0] = 0.
        
        dW2 = self.alpha*(Cyy2-self.p**2)
        W2 += dW2
        W2 = W2-np.diag(np.diag(W2))
        W2[W2 < 0] = 0.
        self.network.inhibitory_weights= W1, W2
    def update_feedforward_weights(self, Y1, Y2, X, batch_size):
        beta=self.beta
        square_act1 = np.sum(Y1*Y1,axis=0)
        square_act2 = np.sum(Y2*Y2,axis=0)
        mymat1 = np.diag(square_act1)
        mymat2 = np.diag(square_act2)
        Q1, Q2 = self.network.feedforward_weights
        dQ1 = beta*X.T.dot(Y1)/batch_size - beta*Q1.dot(mymat1)/batch_size
        Q1 += dQ1
        dQ2 = beta*Y1.T.dot(Y2)/batch_size - beta*Q2.dot(mymat2)/batch_size
        Q2 += dQ2
        self.network.feedforward_weights = Q1, Q2
    def update_thresholds(self, muY1, muY2, gamma, p, batch_size):
        dtheta1 = gamma*(muY1-p)
        dtheta2 = gamma*(muY2-p)
        theta1, theta2 = self.network.thresholds
        theta1 += dtheta1
        theta2 += dtheta2
        self.network.thresholds = theta1, theta2
        
class Data(object):
    def __init__(self, filename, patch_size, seed=20150727):
        self.patch_size=patch_size
        self.rng=np.random.RandomState(seed)
        with open(filename,'r') as f:
            images = cPickle.load(f)
        self.imsize, self.imsize, self.num_images = images.shape
        self.images = np.transpose(images,axes=(2,0,1))
        self.BUFF = 20
    def get_batch(self,batch_size):
        N=np.prod(self.patch_size)
        X = np.zeros((batch_size,N))
        BUFF=self.BUFF
        rng=self.rng
        sz_1, sz_2=self.patch_size
        imsize=self.imsize
        images=self.images
        num_images=self.num_images
        for ii in xrange(batch_size):
            r = BUFF+int((imsize-sz_1-2.*BUFF)*rng.rand())
            c = BUFF+int((imsize-sz_2-2.*BUFF)*rng.rand())
            myimage = images[int(num_images*rng.rand()),
                             r:r+sz_1,
                             c:c+sz_2].ravel()
            X[ii] = myimage
        X=X-X.mean(axis=1,keepdims=True)
        X=X/X.std(axis=1,keepdims=True)
        return X
        