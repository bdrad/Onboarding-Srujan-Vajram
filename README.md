# Onboarding-Srujan-Vajram
Onboarding Project for loading and resizing DICOM images.
Features a function that takes three arguments

dicom_resize(path, dimension, savedir)

1. path: DIRECTORY PATH where the DICOM CSV file is stored 
2. dimension: desired DIMENSION for RESIZING
3. savedir: NAME of the folder (should be placed in the same directory as the DICOM CSV file) where the new resized numpy files are saved

Note: This function attempts to match PIXEL AREA. i.e: If a dimension size of 400 is specified, the function will attempt to scale the dimensions of each DICOM
      image by a scale_factor, such that the AREA of the resized DICOM image is as close to dimension^2 as possible. 
      
      
