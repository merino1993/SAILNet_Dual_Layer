# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 19:09:00 2014

@author: merino1993
"""
#Fun times with git!!!!1

import numpy as np
import cPickle, time
from math import ceil
from Utils import tile_raster_images

def activities(X,Q1,W1,theta1,Q2,W2,theta2):
    batch_size, N = X.shape
    sz = int(np.sqrt(N))

    M1 = Q1.shape[1] #dimension N x M
    M2 = Q2.shape[1] #dimension M1 x M2

    num_iterations = 17

    eta = .1

    B1 = X.dot(Q1)
    B2 = np.zeros((batch_size,M1)) #initialise since aas hasn't been defined yet, need to re-update after each step since inputs for layer 2 are the changing spiking patterns
    
    """
    Q = weight of feed-forward connection between input pixel k, neuron i
    """

    T1 = np.tile(theta1,(batch_size,1))
    T2 = np.tile(theta2,(batch_size,1))
    
    """
    T = membrane threshold, a vector of length M
    batch_size = number of copies
    tile = construct an array by repeating A, reps number of times - np.tile(A, reps)
    """

    Ys1 = np.zeros((batch_size,M1))
    aas1 = np.zeros((batch_size,M1)) #spiking patterns from Layer 1
    Y1 = np.zeros((batch_size,M1))
    
    """    
    aas determines who spikes. Subtracting aas.dot(W) creates inhibition based on the weight.
    aas is either 1 or 0, either fired or not.
    """

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

rng = np.random.RandomState(0)

# Parameters
batch_size = 1000
num_trials = 100

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
OC1 = 2 #overcompleteness 
M1 = OC1*N #number of neurons

'''
If OC1 does not equal OC2 we get an error due to mismatching
'''

#Layer 2
sz = np.sqrt(N).astype(np.int)
OC2 = 2 #overcompleteness WITH RESPECT TO LAYER 1
M2 = OC2*N #number of neurons

# Network Parameters
p = .05 #p = target firing rate = # spikes per image

# Initialize Weights
Q1 = rng.randn(N,M1)
Q1 = Q1.dot(np.diag(1./np.sqrt(np.diag(Q1.T.dot(Q1)))))

Q2 = rng.randn(M1,M2)
Q2 = Q2.dot(np.diag(1./np.sqrt(np.diag(Q2.T.dot(Q2)))))

W1 = np.zeros((M1,M2))
W2 = np.zeros((M1,M2))

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
reconstruction_error = np.zeros(num_trials) #want to keep track of during learning, run per batch

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
    Y1,Y2 = activities(X,Q1,W1,theta1,Q2,W2,theta2)
    muy1 = np.mean(Y1,axis=1)
    muy2 = np.mean(Y2,axis=1)
    Cyy1 = Y1.T.dot(Y1)/batch_size
    Cyy2 = Y2.T.dot(Y2)/batch_size
         
    #reconstruction error
    reconstruction_error[tt] = np.mean((X-Y1.dot(Q1.T))**2)/2.

    # Update lateral weights
    dW1 = alpha*(Cyy1-p**2)
    W1 += dW1
    W1 = W1-np.diag(np.diag(W1))
    W1[W1 < 0] = 0.
    
    dW2 = alpha*(Cyy2-p**2)
    W2 += dW2
    W2 = W2-np.diag(np.diag(W2))
    W2[W2 < 0] = 0.

    # Update feedforward weights
    square_act1 = np.sum(Y1*Y1,axis=0)
    square_act2 = np.sum(Y2*Y1,axis=0)
    mymat1 = np.diag(square_act1)
    mymat2 = np.diag(square_act2)
    dQ1 = beta*X.T.dot(Y1)/batch_size - beta*Q1.dot(mymat1)/batch_size
    Q1 += dQ1
    dQ2 = beta*Y1.T.dot(Y2)/batch_size - beta*Q2.dot(mymat2)/batch_size
    Q2 += dQ2

    # Update thresholds
    dtheta1 = gamma*(np.sum(Y1,axis=0)/batch_size-p)
    dtheta2 = gamma*(np.sum(Y2,axis=0)/batch_size-p)
    theta1 += dtheta1
    theta2 += dtheta2
    dt = time.time()-dt
    algo_time += dt/60.

    Y_ave1 = (1.-eta_ave)*Y_ave1 + eta_ave*muy1
    Y_ave2 = (1.-eta_ave)*Y_ave2 + eta_ave*muy2
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

    
print ''        
import matplotlib.pyplot as plt
K = Q2 #Q2.shape = (25,25)
idx = K.argsort() #sorts in increasing order
print '5 strongest receptive fields in layer 2 (decreasing order): '
print K[idx[-1]], ' index '+str(idx[-1])
print K[idx[-2]], ' index '+str(idx[-2])
print K[idx[-3]], ' index '+str(idx[-3])
print K[idx[-4]], ' index '+str(idx[-4])
print K[idx[-5]], ' index '+str(idx[-5])


"""
syntax: [:,0] not specifying index before colon lists all
"""

"""
moving images
 1) run for 5000-10000 iterations
 2) plot layer 1 neurons (using Jesse's script), hopefully will converge to Gabor
 3) increase OC1 to 1, run on desktop if necessary
 4) simplest way of getting movies
     -run activities loop (n_iterations) for longer,
     instead of 50(ms) run for first ~17 (what it is for movies) iterations
     for next ~16 frames,[#(possible new code after this point)] keep 'visual range' constant but move it several pixels over
     do this for ~16-17 steps / frame (maybe 3ish frames)
     ***to do the 'moving', must step towards a new direction
     -in activities for loop, before calling inputs X, try to make 3d tensor, for each batch move in different time direction
     in activities, would only have to grab several "time" slices

"""
       
#separate plot
print ''
total_time = data_time+algo_time
print 'Percent time spent gathering data: '+str(data_time/total_time)+' %'
print 'Percent time spent in SAILnet: '+str(algo_time/total_time)+' %'
print ''  

plt.figure()
plt.plot(reconstruction_error)
plt.title("Mean Squared Error of SAILNet's Reconstruction with 2 Layer Model")   
'''what are the axis of plt.title("Mean Squared Error of SAILNet's Reconstruction with 2 Layer Model")'''

#plot all receptive fields
plt.figure()
side = int(np.sqrt(N))
img = tile_raster_images(Q1.T, img_shape = (side,side), tile_shape = (2*side,side*OC1/2), tile_spacing=(1, 1), scale_rows_to_unit_interval=True, output_pixel_vals=True)
plt.imshow(img,cmap=plt.cm.Greys, interpolation='nearest')
plt.title('Receptive Fields')
plt.show()

with open('output.pkl','wb') as f:
    cPickle.dump((Q1,W1,W2,theta1,Q2,theta2),f)
    
#Moving Images
'''Add a decaying function to Tao and set Tao to 50000
   Decay Neuron
   The expected result should'''