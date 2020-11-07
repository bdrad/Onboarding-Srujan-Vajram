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
from matplotlib import pyplot as plt
from math import sqrt

description = pd.read_csv("Mass-Training-Description.csv") 

# =========================================================================== #
def dicom_resize(path, dimension, savedir) -> int:
    
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
    IMG_SIZE = dimension
    TARGET_PX_AREA = dimension*dimension
    
    # Set new dimensions 
    #newDims = (IMG_SIZE, IMG_SIZE, len(list_dicom))
    
    # Initialize an empty array of zeros based on new dimensions 
    # ArrayDicom = np.zeros(newDims, dtype=RefD.pixel_array.dtype)
    
    # -------------------------------------- # 
    # Read each dicom file 
    i = 0
    for file in list_dicom:
        
        ds = pydicom.read_file(file)
        
        # Grab the rows and columns
        rows = ds.Rows
        cols = ds.Columns
        
        # Calculate a scale factor that can be multiplied to each dimension
        # of the original DICOM. Our goal is to recale the images as close to 
        # the target pixel area as possible without modifying Aspect Ratio. 
        scale_factor = sqrt( TARGET_PX_AREA / float(rows*cols) )
        
        # Calculate the new dimensions based on scale factor
        newRows = int(np.floor(rows * scale_factor))
        newCols = int(np.floor(cols * scale_factor))
                
        # Extract the relevant pixel data from DICOM
        image = ds.pixel_array
        resized_image = resize(image, (newRows, newCols), anti_aliasing=True)
        
        # Saves the numpy array to specific folder in directory 
        np.savez( os.path.join(savedir, ds.PatientID), resized_image, str(ds.PatientID), int(description.breast_density[i]), str(description.pathology[i]))
        
        # -------------------------------------- # 
        print("Operating on image " + str(i) + " of " + str(len(list_dicom)))
        print("Breast density is: ")
        print(description.breast_density[i])
        i+=1
        
    # -------------------------------------- # 
    
    return 1
    
# =========================================================================== #

pathName = "/Users/srujanvajram/Documents/Internship related/UCSF/DICOM/CBIS-DDSM-Train"
dimension = 300
savedir = 'saved_numpy_files'

dicom_resize(pathName,dimension,savedir)
            
            
       
