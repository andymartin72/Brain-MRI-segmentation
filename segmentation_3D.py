#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:40:41 2024

@author: andrewmartin
"""

import skimage
import scipy
import time
from matplotlib import pyplot as plt
%matplotlib inline
import numpy as np
from scipy.io import loadmat
import sys
from skimage.feature import canny
from skimage import filters
from sklearn.cluster import KMeans
import cv2
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening, disk)
from scipy import ndimage as ndi
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from skimage.measure import label
import napari
import pandas as pd
#import apoc

import pyclesperanto_prototype as cle

from sklearn import metrics

data = loadmat('Brain.mat')
T1 = data['T1']
label = data['label']

img_3d = np.transpose(T1, (2,0,1))
label_3d = np.transpose(label, (2,0,1))

# View images
viewer = napari.view_image(img_segmented_4)
viewer.add_image(label_r)

# Mean thresholding
def mean_thresholding(img_3d):
    mean = img_3d.mean()
    img_3d_thresholded = np.zeros([img_3d.shape[0], img_3d.shape[1], img_3d.shape[2]])
    img_3d_thresholded = (img_3d > mean) * 1
    return img_3d_thresholded
    
# Mathematical morphology opening
def morph_opening(img, iter):
    for i in range(iter):
        img = skimage.morphology.erosion(img)
    for j in range(iter):
        img = skimage.morphology.dilation(img)
    return img

# Mathematical morphology closing
def morph_closing(img, iter):
    for i in range(iter):
        img = skimage.morphology.dilation(img)
    for j in range(iter):
        img = skimage.morphology.erosion(img)
    return img

# Segmentation
img_3d_thresholded = mean_thresholding(img_3d)

# Remove small objects through mathematical morphology
width = 50
remove_objects = skimage.morphology.remove_small_objects(img_3d_thresholded, min_size=width ** 3)

# Performing opening with erosion and dilation
img_opened = morph_opening(remove_objects, 2)

# Connected components labeling
img_connected_component_labeled = cle.connected_components_labeling_box(img_opened)

# Removing irregular comoponent
img_connected_component_labeled = np.where(img_connected_component_labeled <= 2, img_connected_component_labeled, 0)

# Separating skull, skin and the rest
img_segmented_1 = np.where(img_connected_component_labeled == 2, 0, img_connected_component_labeled)

# Performing closing with dilation and erosion to remove holes
img_segmented_opened = morph_closing(img_segmented_1, 2)

# Flipping background and foreground
img_segmented_1 = np.where(img_segmented_opened == 1, 0, 1)
img_connected_component_labeled_2 = cle.connected_components_labeling_box(img_segmented_1)
img_connected_component_labeled_2 = np.asarray(img_connected_component_labeled_2)

img_r = img_connected_component_labeled_2.reshape(img_3d.shape[0], img_3d.shape[1]*img_3d.shape[2])

for i in range(img_r.shape[0]):
    for j in range (img_r.shape[1]):
        if img_r[i][j] == 0:
            img_r[i][j] = 1
        elif img_r[i][j] == 1:
            img_r[i][j] = 0

img_segmented_2 = img_r.reshape(img_3d.shape[0], img_3d.shape[1], img_3d.shape[2])

# Fetching image values of region inside the skull

img_skull = np.zeros([img_3d.shape[0]* img_3d.shape[1]* img_3d.shape[2]])

img_segmented_2_r = img_segmented_2.reshape(img_3d.shape[0]* img_3d.shape[1]*img_3d.shape[2])
img_3d_r = img_3d.reshape(img_3d.shape[0]* img_3d.shape[1]*img_3d.shape[2])

for i in range(img_3d_r.shape[0]):
    if img_segmented_2_r[i] >= 2:
        img_skull[i] = img_3d_r[i]
    else:
        img_skull[i] = 0
        
img_skull = img_skull.reshape(img_3d.shape[0], img_3d.shape[1], img_3d.shape[2])

# Separate skull and CSF

thresh = img_skull.mean()
img_skull_thresholded = (img_skull > thresh) * 1

# Fetching CSF mask through Mathematical morphology
img_skull_opened = morph_opening(img_skull_thresholded, 3)
csf_mask = morph_closing(img_skull_opened, 20)

# Fetching image values of region inside CSF
img_segmented_2_r = img_segmented_2.reshape(img_3d.shape[0]* img_3d.shape[1]*img_3d.shape[2])
img_3d_r = img_3d.reshape(img_3d.shape[0]* img_3d.shape[1]*img_3d.shape[2])
csf_mask_r = csf_mask.reshape(img_3d.shape[0]* img_3d.shape[1]*img_3d.shape[2])
img_segmented_3 = img_segmented_2_r
img_csf = np.zeros([img_3d.shape[0]* img_3d.shape[1]* img_3d.shape[2]])

for i in range(img_3d_r.shape[0]):
    if img_segmented_2_r[i] == 2:
        if csf_mask_r[i] == 0:
            img_segmented_3[i] = 2
        else:
            img_segmented_3[i] = 3
            img_csf[i] = img_3d_r[i]
            
img_segmented_3 = img_segmented_3.reshape(img_3d.shape[0], img_3d.shape[1], img_3d.shape[2])
img_csf = img_csf.reshape(img_3d.shape[0], img_3d.shape[1], img_3d.shape[2])

# Segmenting regions within CSF - CSF, grey matter and white matter
 
thresholds = filters.threshold_multiotsu(img_csf, classes = 4)
csf_segmented = np.digitize(img_csf, bins=thresholds)

img_segmented_3_r = img_segmented_3.reshape(img_3d.shape[0]* img_3d.shape[1]*img_3d.shape[2])
img_3d_r = img_3d.reshape(img_3d.shape[0]* img_3d.shape[1]*img_3d.shape[2])
img_segmented_4 = img_segmented_3_r
csf_segmented = csf_segmented.reshape(img_3d.shape[0]* img_3d.shape[1]*img_3d.shape[2])

for i in range(img_3d_r.shape[0]):
    if img_segmented_3_r[i] == 3:
        if csf_segmented[i] == 1:
            img_segmented_4[i] = 3
        elif csf_segmented[i] == 2:
            img_segmented_4[i] = 4
        elif csf_segmented[i] == 3:
            img_segmented_4[i] = 5
             
img_segmented_4 = img_segmented_4.reshape(img_3d.shape[0], img_3d.shape[1], img_3d.shape[2])

img_segmented_4_r = img_segmented_4.reshape(img_3d.shape[0] * img_3d.shape[1] * img_3d.shape[2])
label_3d_r = label_3d.reshape(img_3d.shape[0] * img_3d.shape[1] * img_3d.shape[2])

metrics.f1_score(label_3d_r, img_segmented_4_r, average = 'macro')