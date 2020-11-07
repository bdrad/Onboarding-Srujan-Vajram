#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 21:05:59 2020

@author: srujanvajram
"""

import numpy as np
import os
import shutil
# =========================================================================== # 

load_path = '/Users/srujanvajram/Documents/Internship related/UCSF/DICOM/saved_numpy_files'
malignantPath = '/Users/srujanvajram/Documents/Internship related/UCSF/DICOM/CLASSES/MALIGNANT'
benighPath = '/Users/srujanvajram/Documents/Internship related/UCSF/DICOM/CLASSES/BENIGN'

# =========================================================================== # 

for filename in os.listdir(load_path):
    if filename.endswith(".npz"):
        
        currPath = load_path + '/' + filename
        npzfile = np.load(currPath)
        npzfile.files
        
        Class = npzfile['arr_3']
        
        # -------------------------------------- # 
        if (Class == 'MALIGNANT'):
            shutil.move(currPath,malignantPath)
        else:
            shutil.move(currPath,benighPath)
            
# =========================================================================== # 
