#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:27:47 2020

@author: thosvarley
"""
cimport cython
import numpy as np 
cimport numpy as np 
from scipy.stats import entropy
from libc.math cimport log2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def jitter_raster(raster,  int scale):
    
    cdef int N0 = raster.shape[0]
    cdef int N1 = raster.shape[1]
    flat_raster = raster.flatten()
    cdef long[:] jitter_raster = np.zeros(N0*N1, dtype="long")
    
    cdef long[:] where = np.where(flat_raster == 1)[0]
    cdef long[:] rands = np.random.normal(loc=0, scale=scale, size=where.shape[0]).astype(int)
    cdef long[:] jitter = np.mod(np.add(where, rands), N0*N1)
    
    cdef int i 
    cdef long rand 
    
    for i in range(jitter.shape[0]):
        
        rand=jitter[i]
        check = False
    
        while check == False:
            
            if (jitter_raster[rand] == 0) and (rand/N1 == where[i]/N1):
                jitter_raster[rand] = 1
                check=True    
            else:
                rand = np.mod(int(np.random.normal(loc=where[i], scale=scale)), N0*N1)

    return np.reshape(jitter_raster, (N0, N1))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def integration(long[:,:] raster):
    """
    I(X) = sum(H(X_i)) - H(X)
    """
    cdef int N0 = raster.shape[0]
    cdef int N1 = raster.shape[1]
    cdef double N1f = raster.shape[1]
    cdef long[:] spikes = np.sum(raster, axis=1)
    
    cdef double sum_ents = 0.0
    cdef int i 
    cdef double p1, p0
    
    for i in range(N0):
        
        if spikes[i] > 0:
            
            p1 = spikes[i] / N1f
            p0 = 1 - p1
            
            sum_ents += -1*((p0*log2(p0)) + (p1*log2(p1)))
    
    cdef long[:,:] unique
    cdef long[:] counts 
    
    unique, counts = np.unique(raster, return_counts=True, axis=1)
    
    cdef double whole = 0.0    
    for i in range(counts.shape[0]):
        whole += (counts[i] / N1f)*log2(counts[i] / N1f)
    whole *= -1
    
    return sum_ents - whole

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def integration_local(long[:,:] raster):

    cdef double[:] p1s = np.divide(np.sum(raster, axis=1), raster.shape[1]).astype("double")
    cdef double[:] p0s = np.subtract(1, p1s).astype("double")

    cdef double[:] h1s = np.multiply(-1, np.log2(p1s)).astype("double")
    cdef double[:] h0s = np.multiply(-1, np.log2(p0s)).astype("double")
    
    cdef long[:,:] unique 
    cdef long[:] counts
    
    unique, counts = np.unique(raster, axis=1, return_counts=True)
    cdef double[:] h_joints = -1*np.log2(np.divide(counts, raster.shape[1])).astype("double")

    cdef int i 
    cdef long x 
    cdef dict moment_lookup = {"".join((str(x) for x in unique[:,i])) : h_joints[i] for i in range(unique.shape[1])}
    
    cdef double local_ents
    cdef double[:] local_integration = np.zeros(raster.shape[1])
    for i in range(raster.shape[1]):
        
        local_ents = 0
        for j in range(raster.shape[0]):
        
            if raster[j][i] == 1:
                local_ents += (h1s[j])
            elif raster[j][i] == 0:
                local_ents += (h0s[j])
                    
        local_integration[i] = local_ents - moment_lookup["".join((str(x) for x in raster[:,i]))]
    
    return local_integration

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def complexity(long[:,:] raster):
    
    cdef int N0 = raster.shape[0]
    cdef int N1 = raster.shape[1]
    
    cdef double integ = integration(raster)
    cdef double[:] integ_exclude = np.zeros(N0)
    
    cdef int i, x 
    
    for i in range(N0):
        integ_exclude[i] = integration(raster.base[[x for x in range(N0) if x != i]])
    
    cdef double cn = ((N0-1)*integ) - (N0*(np.mean(integ_exclude)))
    
    return cn, integ_exclude

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def O_information(long[:,:] raster):
    
    cdef int N0 = raster.shape[0]
    cdef double n = raster.shape[0]
    cdef double N1 = raster.shape[1]
    
    #Joint Entropy
    cdef double[:] probs = np.unique(raster, axis=1, return_counts=True)[1].astype("double") / N1

    cdef double joint_ent = 0.0
    cdef int i, j, x
    cdef double p1, p0, channel_ent, joint_ent_reduced
    cdef long[:,:] raster_reduced
    
    for i in range(probs.shape[0]):
        if probs[i] > 0:
            joint_ent += probs[i]*log2(probs[i])
    joint_ent *= -1

    cdef double summation = 0
    for i in range(N0):
        #Single Channel Entropy
        
        p1 = np.sum(raster[i]) / N1
        p0 = 1 - p1
        
        channel_ent = -1*((p1*log2(p1)) + (p0*log2(p0)))
        
        #Reduced Joint Entropy
        raster_reduced = raster.base[[x for x in range(raster.shape[0]) if x != i]]
        probs = np.unique(raster_reduced, axis=1, return_counts=True)[1].astype("double") / N1
        
        joint_ent_reduced = 0.0
        for j in range(probs.shape[0]):
            if probs[j] > 0:
                joint_ent_reduced += probs[j]*log2(probs[j])
        joint_ent_reduced *= -1
        
        summation += channel_ent - joint_ent_reduced
    
    return ((n-2)*joint_ent)+summation
"""
import pandas as pd 
import numpy as np
in_dir = '/home/thosvarley/Data/wenzel/rasters/'
raster = pd.read_csv(in_dir + "raster_awake_160402.csv",header=None).values.astype("long")
where = np.where(np.sum(raster, axis=1) != 0)[0]
raster = raster[where]

old_integ = 11.391501517594728
"""