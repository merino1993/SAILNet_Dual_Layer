# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:33:07 2015

@author: merino1993
"""

import numpy as np
import cPickle, time
from math import ceil
from Utils import tile_raster_images
from classtrial1 import Network
from classtrial1 import TwoLayerInference
from classtrial1 import Updates

with open('images.pkl','rb') as f:
    images = cPickle.load(f)

#separate plot
print ''
total_time = data_time+algo_time
print 'Percent time spent gathering data: '+str(data_time/total_time)+' %'
print 'Percent time spent in SAILnet: '+str(algo_time/total_time)+' %'
print ''  

plt.figure()
plt.plot(reconstruction_error1)
plt.title("Mean Squared Error of SAILNet's Reconstruction with Layer 1")   
'''what are the axis of plt.title("Mean Squared Error of SAILNet's Reconstruction with 2 Layer Model")'''

plt.figure()
plt.plot(reconstruction_error2)
plt.title("Mean Squared Error of SAILNet's Reconstruction with Layer 2")

#plot all receptive fields
plt.figure()
side = int(np.sqrt(N))
img = tile_raster_images(Q1.T, img_shape = (side,side), tile_shape = (2*side,side*OC1/2), tile_spacing=(1, 1), scale_rows_to_unit_interval=True, output_pixel_vals=True)
plt.imshow(img,cmap=plt.cm.Greys, interpolation='nearest')
plt.title('Receptive Fields')
plt.show()
