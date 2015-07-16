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

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
pp = PdfPages('plots.pdf')

indxs=np.zeros((5,5))
for n in range(5):
    v=Q2[:,n]
    for c in range(5):
        idx=np.argmax(abs(v))
        indxs[n,c]=idx
        v[idx]=0

plt.figure()
plt.plot(indxs)
plt.title("Strongest Layer 2 Neuron Connections to Layer 1 Neurons")
pp.savefig()
#Find the 2 layer neurons that have the strongest connection to the 1 layer neurons

K = Q2 #Q2.shape = (25,25)
idx = K.argsort() #sorts in increasing order
print '5 strongest receptive fields in layer 2 (decreasing order): '
print K[idx[-1]], ' index '+str(idx[-1])
print K[idx[-2]], ' index '+str(idx[-2])
print K[idx[-3]], ' index '+str(idx[-3])
print K[idx[-4]], ' index '+str(idx[-4])
print K[idx[-5]], ' index '+str(idx[-5])


plt.figure()
plt.plot(reconstruction_error1)
plt.title("Mean Squared Error of SAILNet's Reconstruction with Layer 1")   
pp.savefig()

plt.figure()
plt.plot(reconstruction_error2)
plt.title("Mean Squared Error of SAILNet's Reconstruction with Layer 2")
pp.savefig()

plt.figure()
plt.plot(SNR_1)
plt.title("Signal to Noise ratio for Layer 1")
pp.savefig()

plt.figure()
plt.plot(SNR_2)
plt.title("Signal to Noise ratio for Layer 2")
pp.savefig()

#plot all receptive fields
plt.figure()
side = int(np.sqrt(N))
img = tile_raster_images(Q1.T, img_shape = (side,side), tile_shape = (2*side,side*OC1/2), tile_spacing=(1, 1), scale_rows_to_unit_interval=True, output_pixel_vals=True)
plt.imshow(img,cmap=plt.cm.Greys, interpolation='nearest')
plt.title('Receptive Fields')
pp.savefig()
pp.close()
plt.show()