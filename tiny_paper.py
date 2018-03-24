# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 13:30:32 2018

@author: DXL
"""

import numpy as np
import h5py
import scipy
import os
import matplotlib.pyplot as plt
import baseFun_frDepthImage as _base
import baseFun_frDepthImage as _baseFun

def get_part6000_3Dface():
    dataRoot = 'E:\\数据库'   
    indName = 'ind'
    f = h5py.File(indName, 'r')
    ind = f.get(indName)[:]
    f.close()
    
    vTestName = 'E:\\数据库\\vTest.mat'
    vTest = scipy.io.loadmat(vTestName)
    v = vTest['vTest']
    
    pTestName = 'E:\\数据库\\pTest.mat'
    pTest = scipy.io.loadmat(pTestName)
    p = pTest['pTest']
    p1 = p[:, :, 0, 0]
    
    numPoint = np.size(ind)
    pt6000 = np.zeros([numPoint, 3])
    for i in range(numPoint):
        indT = ind[0, i]
        pt6000[i, :] = p1[indT, :]
    
    pt6000 = {'pt6000':pt6000}
    savePath = os.path.join(dataRoot, 'pt6000')
    scipy.io.savemat(savePath, pt6000)

dataRoot = 'E:\\python\\faceData'
dataFolder = 'multiNum50'
cydpName = 'cylinderLbpV1'
dcydpName = 'dCylinderDepthImage'
cydpPath = os.path.join(dataRoot, dataFolder, cydpName)
dcydpPath = os.path.join(dataRoot, dataFolder, dcydpName)
f = h5py.File(cydpPath, 'r')
print(f.keys())
cydp = f.get(cydpName)[:]
f.close()

f = h5py.File(dcydpPath, 'r')
dcydp = f.get(dcydpName)[:]
f.close()

cdp1 = cydp[0, :, :]
cdp2 = dcydp[0, :, :]
cdp2 = _baseFun.depthMap_empty_interpolation_2D(cdp2)

cdp1 = np.flipud(cdp1)
cdp2 = np.flipud(cdp2)

saveRoot = 'E:\\programResult\\tinyPaper\\image'
saveName1 = 'cdp1'
saveName2 = 'cdp2'
savePath1 = os.path.join(saveRoot, saveName1)
savePath2 = os.path.join(saveRoot, saveName2)

scipy.io.savemat(savePath1, {'cdp1':cdp1})
scipy.io.savemat(savePath2, {'cdp2':cdp2})

a = 1

