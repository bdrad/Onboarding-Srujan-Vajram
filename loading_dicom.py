#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:18:41 2020

@author: srujanvajram
"""
# =========================================================================== #
# Import relevant packages 

import pydicom 
import os   
import numpy as np
import pandas as pd
from skimage.transform import resize
from matplotlib import pyplot as plt
from math import sqrt

# Load CSV data of DICOM images 
description = pd.read_csv("Mass-Training-Description.csv") 

# =========================================================================== #
# Define a function to resize images

def dicom_resize(path, dimension, savedir) -> int:
    
    # Holds a reference to the directory that stores the DICOM files of interest
    PathDicom = path
    
     # Will hold DICOM files as we encounter them
    list_dicom = []        
    
    # For loop walks through the direcotries and grabs the Dicom files 
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for fileName in fileList:
            if ".dcm" in fileName:                                  # If we find a dicom extension 
                list_dicom.append(os.path.join(dirName,fileName))   # Add it to the list 
    # -------------------------------------- # 
                
    # Hold a reference to the first DICOM file in the list
    RefD = pydicom.read_file(list_dicom[1])
    
    # Calculate the target pixel area for our images  
    IMG_SIZE = dimension
    TARGET_PX_AREA = dimension*dimension

    # -------------------------------------- # 
    # Process each DICOM file from the list we just created
    
    i = 0
    for file in list_dicom:
        
        ds = pydicom.read_file(file)    # Read the DICOM file
        
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
        
        # Plot the resized image
        plt.imshow(resized_image)
        plt.show()
        
        # Add a new dimension to the image
        resized_image = resized_image[:,:, np.newaxis]
        
        # Add the breast density information to create a numpyZ file
        resized_image[:,:,-1] = description.breast_density[i]
        
        # Saves the numpy array to specific folder in directory 
        np.save( os.path.join(savedir, ds.PatientID), resized_image)
        
        # Print messages describing which image is being processed 
        print("Operating on image " + str(i) + " of " + str(len(list_dicom)))
        print("Breast density is: ")
        print(description.breast_density[i])
        i+=1
        
    # -------------------------------------- # 
    
    # Return 1 when finished 
    return 1
    
# =========================================================================== #

Example use: 
pathName = "/Users/srujanvajram/Documents/Internship related/UCSF/CBIS-DDSM-Train"
dimension = 499
savedir = 'saved_numpy_files'   # The filename that we are storing the numpy files to (file should be in the same directory as the .py) 

dicom_resize(pathName,dimension,savedir)
# =========================================================================== #
