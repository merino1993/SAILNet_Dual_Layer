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
    network,reconstruction_error,SNR,SNR_norm,Q_norm_mean,Q_norm_std=cPickle.load(f)

Q1, Q2 = network.get_feedforward_weights()
W1, W2 = network.get_inhibitory_weights()
theta1, theta2 = network.get_thresholds()

N=Q1.shape[0]
OC1=Q1.shape[1]/N
M1=Q1.shape[1]
OC2=Q2.shape[1]/M1

p = .05
alpha = 1.
beta = .02
gamma = .12
sz = int(np.sqrt(N))

batch_size = 500
patch_size = (16,16)

filename='images.pkl'
data=Data(filename, patch_size, seed=20150727)
X = data.get_batch(batch_size)

network=Network([Q1, Q2], [W1, W2], [theta1, theta2])
infer=TwoLayerInference(network)
Y1, Y2 =infer.activities(X)


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
pp = PdfPages('plots.pdf')


#Find the 2 layer neurons that have the strongest connection to the 1 layer neurons
nL1=10
nL2=10
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


#plot for Layer 1 Receptive Field
plt.figure()
side = int(np.sqrt(N))
img = tile_raster_images(Q1.T, img_shape = (side,side), tile_shape = (2*side,side*OC1/2), tile_spacing=(2, 2), scale_rows_to_unit_interval=True, output_pixel_vals=True)
plt.imshow(img,cmap=plt.cm.Greys, interpolation='nearest')
plt.title('Layer 1 Receptive Fields')
pp.savefig()


#plot for Layer 2 connection strengths to Layer 1
plt.figure()
side = int(np.sqrt(N))
img = tile_raster_images(L2C, img_shape = (side,side), tile_shape = (nL1,nL2), tile_spacing=(2, 2), scale_rows_to_unit_interval=True, output_pixel_vals=True)
plt.imshow(img,cmap=plt.cm.Greys, interpolation='nearest')
plt.title('Layer 2 connection strengths to Layer 1')
plt.xlabel('Layer 1 Receptive Fields')
plt.ylabel('Layer 2 Neurons')
pp.savefig()


#STA: Spike Triggered Average
STA1=X.T.dot(Y1)/batch_size
STA2=X.T.dot(Y2)/batch_size

#plot for Spike Triggered Average for Layer 1
plt.figure()
side = int(np.sqrt(N))
img = tile_raster_images(STA1.T, img_shape = patch_size, tile_shape = patch_size, tile_spacing=(2, 2), scale_rows_to_unit_interval=True, output_pixel_vals=True)
plt.imshow(img,cmap=plt.cm.Greys, interpolation='nearest')
plt.title('Spike Triggered Average for Layer 1')
pp.savefig()

#plot for Spike Triggered Average for Layer 2
plt.figure()
side = int(np.sqrt(N))
img = tile_raster_images(STA2.T, img_shape = patch_size, tile_shape = patch_size, tile_spacing=(2, 2), scale_rows_to_unit_interval=True, output_pixel_vals=True)
plt.imshow(img,cmap=plt.cm.Greys, interpolation='nearest')
plt.title('Spike Triggered Average for Layer 2')
pp.savefig()

#plot for Mean Squared Error of SAILNet's Reconstruction with Layer 1
plt.figure()
plt.plot(reconstruction_error[0])
plt.title("Mean Squared Error of SAILNet's Reconstruction with Layer 1")   
pp.savefig()


#plot for Mean Squared Error of SAILNet's Reconstruction with Layer 2
plt.figure()
plt.plot(reconstruction_error[1])
plt.title("Mean Squared Error of SAILNet's Reconstruction with Layer 2")
pp.savefig()


#plot for Signal to Noise ratio for Layer 1
plt.figure()
plt.plot(SNR[0])
plt.title("Signal to Noise ratio for Layer 1")
pp.savefig()


#plot for Signal to Noise ratio for Layer 2
plt.figure()
plt.plot(SNR[1])
plt.title("Signal to Noise ratio for Layer 2")
pp.savefig()


#plot for Normalized Signal to Noise ratio for Layer 1
plt.figure()
plt.plot(SNR_norm[0])
plt.title("Normalized Signal to Noise ratio for Layer 1")
pp.savefig()

#plot for Normalized Signal to Noise ratio for Layer 2
plt.figure()
plt.plot(SNR_norm[1])
plt.title("Normalized Signal to Noise ratio for Layer 2")
pp.savefig()


#plot for Log Mean Filter Norm
plt.figure()
plt.plot(np.log(Q_norm_mean[0]))
plt.plot(np.log(Q_norm_mean[1]))
plt.title("Log Mean Filter Norm")
pp.savefig()


#plot for Log Standard Deviation Filter Norm
plt.figure()
plt.plot(np.log(Q_norm_std[0]))
plt.plot(np.log(Q_norm_std[1]))
plt.title("Log Standard Deviation Filter Norm")
pp.savefig()

pp.close()
plt.show()
