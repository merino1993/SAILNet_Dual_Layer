# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:33:07 2015

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

with open('output.pkl','r') as f:
    Q1,Q2,W1,W2,theta1,theta2,reconstruction_error1,reconstruction_error2,SNR_1,SNR_2,X=cPickle.load(f)
    

N=Q1.shape[0]
OC1=Q1.shape[1]/N
M1=Q1.shape[1]
OC2=Q2.shape[1]/M1

p = .05
alpha = 1.
beta = .02
gamma = .12
sz = int(np.sqrt(N))

batch_size = 128
patch_size = (16,16)

network=Network([Q1, Q2], [W1, W2], [theta1, theta2])
infer=TwoLayerInference(network)
Y=infer.activities(X)


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
pp = PdfPages('plots.pdf')


#Find the 2 layer neurons that have the strongest connection to the 1 layer neurons
nL1=5
nL2=5
indxs=np.zeros((nL2,nL1))
for n in range(nL2):
    v=Q2[:,n]
    for c in range(nL1):
        idx=np.argmax(v)
        indxs[n,c]=idx
        v[idx]=0
L2C=np.zeros((nL1*nL2,N))
for i,n in enumerate(indxs.ravel()):
    L2C[i]=Q1[:,n]


#plot for Layer 2 Receptive Field
plt.figure()
side = int(np.sqrt(N))
img = tile_raster_images(L2C, img_shape = (side,side), tile_shape = (nL1,nL2), tile_spacing=(1, 1), scale_rows_to_unit_interval=True, output_pixel_vals=True)
plt.imshow(img,cmap=plt.cm.Greys, interpolation='nearest')
plt.title('Layer 2 Receptive Fields')
pp.savefig()


#STA: Spike Triggered Average
STA=X.T.dot(Y)/batch_size

#plot for Spike Triggered Average
plt.figure()
side = int(np.sqrt(N))
img = tile_raster_images(sz, img_shape = (side,side), tile_shape = patch_size, tile_spacing=(1, 1), scale_rows_to_unit_interval=True, output_pixel_vals=True)
plt.imshow(img,cmap=plt.cm.Greys, interpolation='nearest')
plt.title('Spike Triggered Average')
pp.savefig()

#plot for Mean Squared Error of SAILNet's Reconstruction with Layer 1
plt.figure()
plt.plot(reconstruction_error1)
plt.title("Mean Squared Error of SAILNet's Reconstruction with Layer 1")   
pp.savefig()


#plot for Mean Squared Error of SAILNet's Reconstruction with Layer 2
plt.figure()
plt.plot(reconstruction_error2)
plt.title("Mean Squared Error of SAILNet's Reconstruction with Layer 2")
pp.savefig()


#plot for Signal to Noise ratio for Layer 1
plt.figure()
plt.plot(SNR_1)
plt.title("Signal to Noise ratio for Layer 1")
pp.savefig()


#plot for Signal to Noise ratio for Layer 2
plt.figure()
plt.plot(SNR_2)
plt.title("Signal to Noise ratio for Layer 2")
pp.savefig()


#plot for Layer 1 Receptive Field
plt.figure()
side = int(np.sqrt(N))
img = tile_raster_images(Q1.T, img_shape = (side,side), tile_shape = (2*side,side*OC1/2), tile_spacing=(1, 1), scale_rows_to_unit_interval=True, output_pixel_vals=True)
plt.imshow(img,cmap=plt.cm.Greys, interpolation='nearest')
plt.title('Layer 1 Receptive Fields')
pp.savefig()
pp.close()
plt.show()