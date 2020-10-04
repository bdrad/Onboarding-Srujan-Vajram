#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:18:41 2020

@author: srujanvajram
"""

import pydicom 
import os   
import numpy as np
import pandas as pd
from skimage.transform import resize

description = pd.read_csv("Mass-Training-Description.csv") 

# Look into using JSON files. 
# 1,2 = FALSe. 3,4 = TRUE for breast density. 
# Dense breasts have slightly higher probability of breast cancer. 
# Try to load the DICOM files sepeartely and review the images. 
# Make sure to save the new dimension sizes prior to saving it 
# Ger a dicom viewer 

# =========================================================================== #
def dicom_resize(path, dimension) -> np.ndarray:
    
    # Holds a reference to the directory that stores the DICOM files of interest
    PathDicom = path
    
     # Will store the DICOM files 
    list_dicom = []        
    
    # For loop walks through the direcotries and grabs the Dicom files 
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for fileName in fileList:
            if ".dcm" in fileName:
                list_dicom.append(os.path.join(dirName,fileName))
    # -------------------------------------- # 
                
    # Hold a reference to the first DICOM file in the list
    RefD = pydicom.read_file(list_dicom[1])
    
    # Set the new dimensions 
    IMG_PX_SIZE = dimension
    
    # Set new dimensions 
    newDims = (IMG_PX_SIZE, IMG_PX_SIZE, len(list_dicom))
    
    # Initialize an empty array of zeros based on new dimensions 
    ArrayDicom = np.zeros(newDims, dtype=RefD.pixel_array.dtype)
    
    # -------------------------------------- # 
    # Read each dicom file 
    i = 0
    for file in list_dicom:
        
        ds = pydicom.read_file(file)
        data = ds.pixel_array
        
        resized_img = resize(data, (IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True)
        ArrayDicom[:, :, list_dicom.index(file)] = resized_img
        print("Operating on image " + str(i) + " of " + str(len(list_dicom)))
        print("Breast density is: ")
        print(description.breast_density[i])
        i+=1
        
    # -------------------------------------- # 
    
    return ArrayDicom
    
# =========================================================================== #

pathName = "/Users/srujanvajram/Documents/Internship related/UCSF/CBIS-DDSM-Train"
dimension = 299

dicom_resize(pathName,dimension)
            
            
        
        
    
   
