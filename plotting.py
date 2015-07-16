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

with open('output.pkl','r') as f:
    Q1,Q2,W1,W2,theta1,theta2,reconstruction_error1,reconstruction_error2,SNR_1,SNR_2=cPickle.load(f)

N=Q1.shape[0]
OC1=Q1.shape[1]/N
M1=Q1.shape[1]
OC2=Q2.shape[1]/M1

import matplotlib.pyplot as plt

K = Q2 #Q2.shape = (25,25)
idx = K.argsort() #sorts in increasing order
print '5 strongest receptive fields in layer 2 (decreasing order): '
print K[idx[-1]], ' index '+str(idx[-1])
print K[idx[-2]], ' index '+str(idx[-2])
print K[idx[-3]], ' index '+str(idx[-3])
print K[idx[-4]], ' index '+str(idx[-4])
print K[idx[-5]], ' index '+str(idx[-5])

plot1=plt.figure()
plt.plot(reconstruction_error1)
plt.title("Mean Squared Error of SAILNet's Reconstruction with Layer 1")   
plt.savefig('plot1.png')

plot2=plt.figure()
plt.figure()
plt.plot(reconstruction_error2)
plt.title("Mean Squared Error of SAILNet's Reconstruction with Layer 2")
plt.savefig('plot2.png')

plot3=plt.figure()
plt.figure()
plt.plot(SNR_1)
plt.title("Signal to Noise ratio for Layer 1")
plt.savefig('plot3.png')

plot4=plt.figure()
plt.figure()
plt.plot(SNR_2)
plt.title("Signal to Noise ratio for Layer 2")
plt.savefig('plot4.png')

#plot all receptive fields
plot5=plt.figure()
plt.figure()
side = int(np.sqrt(N))
img = tile_raster_images(Q1.T, img_shape = (side,side), tile_shape = (2*side,side*OC1/2), tile_spacing=(1, 1), scale_rows_to_unit_interval=True, output_pixel_vals=True)
plt.imshow(img,cmap=plt.cm.Greys, interpolation='nearest')
plt.title('Receptive Fields')
plt.savefig('plot5.png')
plt.show()