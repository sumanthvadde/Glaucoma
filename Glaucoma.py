# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:08:56 2019

@author: DELL
"""
import cv2
import os
import numpy as np
import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
Data= os.listdir('C:\\Users\\DELL\\Desktop\\Glaucoma-Detection\\Images')
for i in range(6,10):
    img = cv2.imread('C:\\Users\\DELL\\Desktop\\Glaucoma-Detection\\Images' + "/" + Data[i])
    print(Data[i])
  
    R,G,B= cv2.split(img)  
    

    
    #splitting into 3 channels
    
    RC = R-R.mean()#Preprocessing Red
    RC = RC-RC.mean()-R.std() #Preprocessing Red
    RC = RC-RC.mean()-RC.std() #Preprocessing Red
    
    MeanR = RC.mean()#Mean of preprocessed red
    SDR = RC.std()#SD of preprocessed red
    
    # Thr = 49.5 - 12 - Ar.std()               
    #OD Threshold
    
    Thr = RC.std()
    #print(Thr)
    
    GC = G - G.mean()#Preprocessing Green
    GC= GC- GC.mean()-G.std() #Preprocessing Green
    
    MeanG = GC.mean()#Mean of preprocessed green
    SDG = GC.std()#SD of preprocessed green
    Thg = GC.mean() + 2*GC.std() + 49.5 + 12 #OC Threshold
    
    filter = signal.gaussian(99, std=6) #Gaussian Window
    filter=filter/sum(filter)
    
    hist,bins = np.histogram(GC.ravel(),256,[0,256])#Histogram of preprocessed green channel
    histr,binsr = np.histogram(RC.ravel(),256,[0,256])#Histogram of preprocessed red channel
    
    smooth_hist_g=np.convolve(filter,hist)  #Histogram Smoothing Green
    smooth_hist_r=np.convolve(filter,histr) #Histogram Smoothing Red
    
    #mse(smooth_hist_g,hist)
    """
    plt.subplot(2, 2, 1)
    plt.plot(hist)
    plt.title("Preprocessed Green Channel")
    
    plt.subplot(2, 2, 2)
    plt.plot(smooth_hist_g)
    plt.title("Smoothed Histogram Green Channel")
    
    plt.subplot(2, 2, 3)
    plt.plot(histr)
    plt.title("Preprocessed Red Channel")
    
    plt.subplot(2, 2, 4)
    plt.plot(smooth_hist_r)
    plt.title("Smoothed Histogram Red Channel")
    
    plt.show()
    
    """
    
    r,c = GC.shape
    Dd = np.zeros(shape=(r,c))
    Dc = np.zeros(shape=(r,c))
    
    for i in range(1,r):
    	for j in range(1,c):
    		if RC[i,j]>Thr:
    			Dd[i,j]=255
    		else:
    			Dd[i,j]=0
    
    for i in range(1,r):
    	for j in range(1,c):
    		if GC[i,j]>Thg:
    			Dc[i,j]=1
    		else:
    			Dc[i,j]=0
    
    cv2.imwrite('disk.png',Dd)
    cv2.imwrite('cup.png',Dc)
    """
    plt.imshow(Dd, cmap = 'gray', interpolation = 'bicubic')
    plt.title("Optic Disk")
    plt.show()
    
    plt.imshow(Dc, cmap = 'gray', interpolation = 'bicubic')
    plt.title("Optic Cup")
    plt.show()"""
    
    Cup=cv2.imread("C:\\Users\\DELL\\Desktop\\Glaucoma-Detection\\Scripts\\cup.png",0)
    Disk=cv2.imread("C:\\Users\\DELL\\Desktop\\Glaucoma-Detection\\Scripts\\disk.png",0)
    #morphological closing and opening operations
    R1 = cv2.morphologyEx(Cup, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)), iterations = 1)
    r1 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 1)
    R2 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,21)), iterations = 1)
    r2 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,1)), iterations = 1)
    R3 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33)), iterations = 1)	
    r3 = cv2.morphologyEx(R3, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(43,43)), iterations = 1)

    img1 = clahe.apply(r3)
    
    
    ret,thresh = cv2.threshold(Cup,127,255,0)
    
   
    contours,hierarchy = cv2.findContours(Cup, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Getting all possible contours in the segmented image
    cup_diameter = 0
    largest_area = 0
    
    el_cup = 0
    if len(contours) != 0:
        for i in range(len(contours)):
            if len(contours[i]) >=8:
                area = cv2.contourArea(contours[i]) #Getting the contour with the largest area
                if (area>largest_area):
                    largest_area=area
                    index = i
                    el_cup = cv2.fitEllipse(contours[i])
                
     
    x,y,w,h = cv2.boundingRect(contours[index]) #fitting a rectangle on the ellipse to get the length of major axis
    cup_diameter = max(w,h) #major axis is the diameter

    #morphological closing and opening operations
    R1 = cv2.morphologyEx(Disk, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)), iterations = 1)
    r1 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 1)
    R2 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,21)), iterations = 1)
    r2 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,1)), iterations = 1)
    R3 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33)), iterations = 1)
    r3 = cv2.morphologyEx(R3, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(43,43)), iterations = 1)

    img2 = clahe.apply(r3)
    
    ret,thresh = cv2.threshold(Disk,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
    #Getting all possible contours in the segmented image
    disk_diameter = 0
    largest_area = 0
    el_disc = el_cup
    if len(contours) != 0:
          for i in range(len(contours)):
            if len(contours[i]) >= 8:
                area = cv2.contourArea(contours[i]) #Getting the contour with the largest area
                if (area>largest_area):
                    largest_area=area
                    index = i
                    el_disc = cv2.fitEllipse(contours[i])
                    
                    
    cv2.ellipse(img2,el_disc,(140,60,150),3) #fitting ellipse with the largest area
    x,y,w,h = cv2.boundingRect(contours[index]) #fitting a rectangle on the ellipse to get the length of major axis
    disk_diameter = max(w,h) #major axis is the diameter
    if(disk_diameter == 0):
        print("The disk_diameter is zero. Hence the CDR value cannot be counted.")
        
    
    cdr = cup_diameter/disk_diameter 
    CDR=[]
    CDR.append(cdr)
    print(CDR)
    
    

    