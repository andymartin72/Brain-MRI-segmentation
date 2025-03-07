#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing modules

import skimage
import scipy
import time
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
from skimage import filters
from sklearn.cluster import KMeans
import cv2
from sklearn import metrics

# Load data
data = loadmat('Brain.mat')
T1 = data['T1']
label = data['label']
# Flips x and y coordinates of contour as numpy array
def flip_coordinates(contour):
    contour_r = contour[:, 1]
    contour_r = np.stack((contour_r, contour[:, 0]), axis = 1)
    return contour_r
# Finding most significant contours using Marching cubes algorithm
def finding_significant_contours(img, count):
    contours = skimage.measure.find_contours(img)
    # Sorting in descending order of contour length
    n = len(contours)
    for i in range(n-1):
        for j in range(0, n-i-1):
            if len(contours[j]) < len(contours[j+1]):
                contours[j], contours[j+1] = contours[j+1], contours[j]
    significant_contours = []
    for j in range(count):
        contour = flip_coordinates(contours[j])
        significant_contours.append(contour)
    return significant_contours

# Fetching skin, CSF and skull masks from contours
def fetch_masks_from_contours(img, contours):
    # Separating skin mask
    contour_skin = contours[1]
    skin_mask = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(skin_mask, np.int32([contour_skin]), 1)
    # Separating csf mask
    contour_csf = contours[0]
    csf_mask = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(csf_mask, np.int32([contour_csf]), 1)
    # Separating skull mask
    contour_skull = contours[2]
    skull_mask = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(skull_mask, np.int32([contour_skull]), 1)
    return skin_mask, skull_mask, csf_mask

# Segmenting Air, skin, skull and CSF from their masks
def segment_regions_from_masks(img, skin_mask, skull_mask, csf_mask):
    # Segmenting skin, air and skull
    img_segmented = skin_mask
    img_r = img_segmented.reshape(img.shape[0]*img.shape[1])
    skull_mask_r = skull_mask.reshape(img.shape[0]*img.shape[1])
    for i in range(img_r.shape[0]):
        if img_r[i] == 1:
            if skull_mask_r[i] == 0:
                img_r[i] = 1
            else:
                img_r[i] = 2
            
    img_segmented = img_r.reshape(img.shape[0], img.shape[1])
    # Segmenting CSF from skull

    img_r = img_segmented.reshape(img.shape[0]*img.shape[1])
    csf_mask_r = csf_mask.reshape(img.shape[0]*img.shape[1])

    for i in range(img_r.shape[0]):
        if img_r[i] == 2:
            if csf_mask_r[i] == 0:
                img_r[i] = 2
            else:
                img_r[i] = 3
            
    img_segmented = img_r.reshape(img.shape[0], img.shape[1])
    return img_segmented

# Fetching MRI values inside CSF mask
def fetch_csf_mri(img, csf_mask):
    img_r = img.reshape(img.shape[0]*img.shape[1])
    csf_mask_r = csf_mask.reshape(img.shape[0]*img.shape[1])
    img_csf = np.zeros(img.shape[0] * img.shape[1])
    for i in range(img_r.shape[0]):
        if csf_mask_r[i] == 1:
            img_csf[i] = img_r[i] 
            
    img_csf = img_csf.reshape(img.shape[0], img.shape[1])
    return img_csf

# Integrating contour segmentation with CSF segmentation
def join_segmentation(img, img_segmented, segmented_csf):
    img_r = img_segmented.reshape(img.shape[0]*img.shape[1])
    segmented_csf_r = segmented_csf.reshape(img.shape[0]*img.shape[1])
    for i in range(img_r.shape[0]):
        if img_r[i] >= 3:
            if segmented_csf_r[i] == 1:
                img_r[i] = 3
            elif segmented_csf_r[i] == 2:
                img_r[i] = 4
            elif segmented_csf_r[i] == 3:
                img_r[i] = 5
    img_segmented = img_r.reshape(img.shape[0], img.shape[1])
    return img_segmented

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

# Step 1 - Algorithm 1 - Mean thresholding
def mean_thresholding(img):
    img_r = img.reshape(img.shape[0]*img.shape[1])
    mean = img_r.mean()
    for i in range(img_r.shape[0]):
        if img_r[i] > mean:
            img_r[i] = 1
        else:
            img_r[i] = 0    
    img = img_r.reshape(img.shape[0], img.shape[1])
    return img

# Step 1 - Algorithm 2 - Otsu thresholding
def otsu_thresholding(img):
    img_r = img.reshape(img.shape[0]*img.shape[1])
    thresh = filters.threshold_otsu(img)
    for i in range(img_r.shape[0]):
        if img_r[i] > thresh:
            img_r[i] = 1
        else:
            img_r[i] = 0    
    img = img_r.reshape(img.shape[0], img.shape[1])
    return img

# Step 2 - Algorithm 1 - Segmenting with MAthematical morphology
def segmenting_with_mathematical_morphology(img, img_thresholded):
    #Opening to seprate area within CSF
    img_opened = morph_opening(img_thresholded, 12)
    csf_mask = morph_closing(img_opened, 25)
    #Closing to separate area within skin
    img_closed = morph_closing(img_thresholded, 8)
    skin_mask = scipy.ndimage.binary_fill_holes(img_closed).astype(int)
    #Separating area between skin and skull
    img_sub_mask = skin_mask - csf_mask 
    #Replacing image pixel values with the previous subtracted mask
    img_r = img.reshape(img.shape[0] * img.shape[1])
    img_sub_r = img_sub_mask.reshape(img.shape[0] * img.shape[1])
    for i in range(img_r.shape[0]):
        if img_sub_r[i] == 1:
           img_sub_r[i] = img_r[i] 
    img_sub = img_sub_r.reshape(img.shape[0], img.shape[1])
    # Thresholding to separate skin and skull
    img_sub_thresh = otsu_thresholding(img_sub)
    img_sub_thresh = morph_opening(img_sub_thresh, 3)
    # Separating area inside skull area
    img_sub_mask_2 = img_sub_mask - img_sub_thresh
    # Segmenting air, skin, skull and the rest of the brain
    img_segmented = skin_mask
    img_segmented_r = img_segmented.reshape(img.shape[0] * img.shape[1])
    img_sub_mask_2_r = img_sub_mask_2.reshape(img.shape[0] * img.shape[1])
    csf_mask_r = csf_mask.reshape(img.shape[0] * img.shape[1])
    for i in range(img_r.shape[0]):
        if img_segmented_r[i] == 1:
            if img_sub_mask_2_r[i] == 1:
                img_segmented_r[i] = 2
            elif csf_mask_r[i] == 1:
                img_segmented_r[i] = 3
    img_segmented_1 = img_segmented_r.reshape(img.shape[0], img.shape[1])
    return img_segmented_1, csf_mask

# Step 2 - Algorithm 2 - Segmenting with marching cubes
def segmenting_with_marching_cubes(img):
    # Find contours of skin, skull and CSF
    contours = finding_significant_contours(img, 3)
    # Fetch masks from the contours 
    skin_mask, skull_mask, csf_mask = fetch_masks_from_contours(img, contours)
    # Segment  Air, skin, skull and CSF regions from masks
    img_segmented = segment_regions_from_masks(img, skin_mask, skull_mask, csf_mask)
    # Segmenting regions within CSF - CSF, grey matter and white matter
    img_csf = fetch_csf_mri(img, csf_mask)
    return img_segmented, csf_mask

# Step 3 - Algorithm 1 - Segmenting with Multi-Otsu thresholding
def segmenting_csf_with_thresholding(img_csf):
    thresholds = filters.threshold_multiotsu(img_csf, classes = 4)
    segmented_csf = np.digitize(img_csf, bins=thresholds)
    return segmented_csf 

# Step 3 - Algorithm 2 - Segmenting with KMeans
def segmenting_csf_with_Kmeans(img, img_csf):
    img_csf_r = img_csf.reshape(img_csf.shape[0]*img_csf.shape[1], 1)
    img_kmeans = KMeans(n_clusters=4, random_state=0, n_init=10).fit(img_csf_r)
    segmented_csf_r = img_kmeans.labels_
    segmented_csf = segmented_csf_r.reshape(img.shape[0], img.shape[1])
    return segmented_csf 

# Main segmentation function.
# Pass algorithms for step 1, 2 and 3 in arguments
def segment_brain_mri(img, thresholding_method = 'mean', segmenting_contours_method = 'marching_cubes', segmenting_csf_method = 'thresholding'):
    # Gaussian smoothing
    # img_smoothed = gaussian_blur(img, 1)
    # Otsu thresholding
    if thresholding_method == "otsu":
        img_thresholded = otsu_thresholding(img)
    else:
        img_thresholded = mean_thresholding(img)
    
    if segmenting_contours_method == 'mathematical_morphology':
        img_segmented, csf_mask = segmenting_with_mathematical_morphology(img, img_thresholded)
    else:
        img_segmented, csf_mask = segmenting_with_marching_cubes(img_thresholded)
        
    img_csf = fetch_csf_mri(img, csf_mask)
    if segmenting_csf_method == 'KMeans':
        segmented_csf = segmenting_csf_with_Kmeans(img, img_csf)
    else:
        # Otsu threshold
        segmented_csf = segmenting_csf_with_thresholding(img_csf)
    
    # Integrating first segmentation with the segmentation inside CSF
    img_segmented_2 = join_segmentation(img, img_segmented, segmented_csf)
    return img_segmented_2
    
# Evaluation
def evaluation(T1_pred, labels):
    accuracy_list = []
    precision_list_macro = []
    recall_list_macro = []
    precision_list_micro = []
    recall_list_micro = []
    dice_coefficient_list = []
    for i in range(T1_pred.shape[2]):
        img_pred = T1_pred[:,:,i]
        label_img = labels[:,:,i]
        img_pred_r = img_pred.reshape(img_pred.shape[0] * img_pred.shape[1])
        label_img_r = label_img.reshape(label_img.shape[0] * label_img.shape[1])

        # Dice coefficient
        dice_coeff = metrics.f1_score(label_img_r, img_pred_r, average = 'macro')
        dice_coefficient_list.append(dice_coeff)
        
    dice_coefficient = sum(dice_coefficient_list) / len(dice_coefficient_list)
    print("Dice cofficient: ", dice_coefficient)

def segment_MRIs(T1, thresholding_method = 'mean', segmenting_contours_method = 'marching_cubes', segmenting_csf_method = 'thresholding'):
    T1_pred = np.zeros([T1.shape[0], T1.shape[1], T1.shape[2]])
    for i in range(T1.shape[2]):
        img = T1[:,:,i]
        T1_pred[:,:,i] = segment_brain_mri(img, thresholding_method, segmenting_contours_method, segmenting_csf_method)
    return T1_pred

thresholding = ['mean', 'otsu']
segmenting_contours = ['mathematical_morphology', 'marching_cubes']
segmenting_csf = ['thresholding', 'KMeans']

# Algorithm 1
segmented_1 = segment_MRIs(T1, 'otsu', 'mathematical_morphology', 'thresholding')
evaluation(segmented_1, label)
# Algorithm 2
segmented_2 = segment_MRIs(T1, 'otsu', 'marching_cubes', 'thresholding')
evaluation(segmented_2, label)
# Algorithm 3
segmented_3 = segment_MRIs(T1, 'mean', 'marching_cubes', 'KMeans')
evaluation(segmented_3, label)
# Algorithm 4
segmented_4 = segment_MRIs(T1, 'mean', 'marching_cubes', 'thresholding')
evaluation(segmented_4, label)

plt.title('2D Segmented output')
plt.imshow(segmented_2[:,:,0], cmap = 'jet')
    