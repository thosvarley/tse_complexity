#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:52:43 2020

@author: thosvarley
"""

import numpy as np 
import pandas as pd 

in_dir = '/home/thosvarley/Data/wenzel/rasters/'
raster = pd.read_csv(in_dir + "raster_awake_160402.csv",header=None).values.astype("long")
where = np.where(np.sum(raster, axis=1) != 0)[0]
raster = raster[where]
old_integ = 11.391501517594728

n = raster.shape[0]
print(n-2)

#Joint Entropy 

unique, counts = np.unique(raster, axis=1, return_counts=True)
probs = counts / raster.shape[1]

joint_ent = 0.0
for i in range(probs.shape[0]):
    if probs[i] > 0:
        joint_ent += probs[i]*np.log2(probs[i])
joint_ent *= -1

print((n-2)*joint_ent)

summation = 0
for i in range(raster.shape[0]):
    #Single Channel Entropy
    
    p1 = np.sum(raster[i]) / raster.shape[1]
    p0 = 1 - p1
    
    channel_ent = -1*((p1*np.log2(p1)) + (p0*np.log2(p0)))
    
    #Reduced Joint Entropy
    raster_reduced = raster[[x for x in range(raster.shape[0]) if x != i]]
    unique, counts = np.unique(raster_reduced, axis=1, return_counts=True)
    probs = counts / raster_reduced.shape[1]
    
    joint_ent_reduced = 0.0
    for j in range(probs.shape[0]):
        if probs[j] > 0:
            joint_ent_reduced += probs[j]*np.log2(probs[j])
    joint_ent_reduced *= -1
    
    summation += channel_ent - joint_ent_reduced
    
    print(i / raster.shape[0])

print(summation)