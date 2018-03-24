# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:22:00 2017

@author: dxl
"""

import numpy as np
import h5py
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

#-------------------------
def load_data_zju(path):
    temp = h5py.File(path,'r')
    print(temp.keys())
    data = temp.get('Vertices')               #Vertices为浙大库，表情×人脸×坐标×点序号
    dataArr = np.array(data)
    return dataArr
    pass
#-------------------------
def save_data_zju(name, dataArr):
    f = h5py.File(name, 'w')
    f.create_dataset('p', data = dataArr)
    f.close()
    pass
#-------------------------
def find_bj_pos(arr):
    z = arr[:, 2]
    listz = list(z)
    bjInd = listz.index(max(listz))
    return bjInd
    pass

def find_k_neighbors(p, num):
    p = p.T
    bjInd = find_bj_pos(p)
    bj = p[bjInd, :]
    bj = bj.T
    NN = NearestNeighbors(n_neighbors = num)
    NN.fit(p)
    dis, ind = NN.kneighbors(bj)
    return ind
    pass
#-------------------------
def save_as_h5py(path, name, dataArr):
    f = h5py.File(name, 'w')
    f.create_dataset(name, data = dataArr)
    f.close()
    pass
#-------------------------
def load_h5py_file(path):
    f = h5py.File(path,'r')     
    ind = f.get('ind')[:]
    return ind
    pass
#-------------------------
def get_part5300():
    dataRoot = '/home/dxl/myPro/faceData'
    pathP = os.path.join(dataRoot, '3D', 'Vertices.mat')
    pathInd = os.path.join(dataRoot, 'depthImage', 'ind5300')
    pSaveName = 'part5300'
    pathSave = os.path.join(dataRoot, 'depthImage', pSaveName)
    
    allP = load_data_zju(pathP)
    ind5300 = load_h5py_file(pathInd)
    
    sp1, sp2, sp3, sp4 = np.shape(allP)
    allP = np.reshape(allP, [-1, sp3, sp4])
    numAll = np.size(allP, 0)
    
    ind5300 = ind5300.T
    numInd = np.size(ind5300, 0)
    
    depthImage = np.zeros([numAll, numInd, 3])
    
    for i in range(numAll):
        print('person ', i)
        pT = allP[i, :, :]
        pT = pT.T
        for j in range(numInd):
            depthImage[i, j, :] = pT[ind5300[j], :]
            pass
    
    #f = h5py.File(pathSave, 'w')
    #f.create_dataset(pSaveName, data = depthImage)
    #f.close()
    pass
#-------------------------
def get_detail_depthImage(part5300):
    numPers = np.size(part5300, 0)
    xMin = 0
    yMin = 0
    xMax = 0
    yMax = 0
    xDis = 0
    yDis = 0
    bjxMax = 0.0
    bjyMax = 0.0
    for i in range(numPers):
        pt = part5300[i, :, :]
        x = pt[:, 0]
        y = pt[:, 1]
        xMinT = min(x)
        yMinT = min(y)
        xMaxT = max(x)
        yMaxT = max(y)
        bjx = x[0]
        bjy = y[0]
        xDisT = xMaxT - xMinT
        yDisT = yMaxT - yMinT
        if xMinT < xMin:
            xMin = xMinT
        if yMinT < yMin:
            yMin = yMinT
        if xMaxT > xMax:
            xMax = xMaxT
        if yMaxT > yMax:
            yMax = yMaxT
        if xDisT > xDis:
            xDis = xDisT
        if yDisT > yDis:
            yDis = yDisT
        if bjx > bjxMax and bjy > bjyMax:
            bjxMax = bjx
            bjyMax = bjy
    return xMin, yMin, xMax, yMax, xDis, yDis, bjxMax, bjyMax
    pass  
#------------------------- 
def data_panning(part5300):   
    numPers = np.size(part5300, 0)
    numPoint = np.size(part5300, 1)
    
    xMin, yMin, xMax, yMax, xDis, yDis, bjxMax, bjyMax = get_detail_depthImage(part5300)
    
    part5300n = np.zeros([numPers, numPoint, 3])
    for i in range(numPers):
        pt = part5300[i, :, :]
        pt[:, 0] = pt[:, 0] + (bjxMax - pt[0, 0])
        pt[:, 1] = pt[:, 1] + (bjyMax - pt[0, 1])
        pt[:, 0] = pt[:, 0] - xMin
        pt[:, 1] = pt[:, 1] - yMin
        part5300n[i, :, :] = pt
        print('part5300n', i)
    
    name = 'part5300n'
    pathPart5300n = os.path.join(dataRoot, fileFolder, name)
    f = h5py.File(pathPart5300n, 'w')
    f.create_dataset(name, data = part5300n)
    f.close()
    pass
#------------------------- 
def get_detail_depthImage_pro(part5300n):
    numPers = np.size(part5300n, 0)
    xMin = 0
    yMin = 0
    xMax = 0
    yMax = 0
    for i in range(numPers):
        pt = part5300n[i, :, :]
        x = pt[:, 0]
        y = pt[:, 1]
        xMinT = min(x)
        yMinT = min(y)
        xMaxT = max(x)
        yMaxT = max(y)
        if xMinT < xMin:
            xMin = xMinT
        if yMinT < yMin:
            yMin = yMinT
        if xMaxT > xMax:
            xMax = xMaxT
        if yMaxT > yMax:
            yMax = yMaxT
    return xMin, yMin, xMax, yMax
    pass  
#------------------------- 
def get_depthImage(part5300n):
    numPers = np.size(part5300n, 0)
    numPoints = np.size(part5300n, 1)
    part5300nn = np.zeros([numPers, numPoints, 3])
    for i in range(numPers):
        pt = part5300n[i, :, :]
        pt[:, 0] = np.ceil(pt[:, 0] * 40)
        pt[:, 1] = np.ceil(pt[:, 1] * 40)
        part5300nn[i, :, :] = pt
        print('part5300nn', i)
        pass
    
    xMin, yMin, xMax, yMax = get_detail_depthImage_pro(part5300nn)
    
    # get depthImage
    xMax = int(xMax)
    yMax = int(yMax)
    depthImage = np.zeros([numPers, yMax+1, xMax+1])
    for k in range(numPers):
        pt = part5300nn[k, :, :]
        print('depthImage', k)
        for kk in range(numPoints):
            x = int(pt[kk, 0])
            y = int(pt[kk, 1])
            z = pt[kk, 2]
            if depthImage[k, y, x] < z:
                depthImage[k, y, x] = z
        
    name = 'depthImage'
    depthImagePath = os.path.join(dataRoot, fileFolder, name)
    f1 = h5py.File(depthImagePath, 'w')
    f1.create_dataset(name, data = depthImage)
    f1.close()
    pass
#------------------------- 
def get_labels(dataRoot, fileFolder):
    numPress = 47
    numPers = 150
    numAll = numPress * numPers
    labels = np.zeros([numAll, 1])
    k = 0
    for i in range(numPress):
        for j in range(numPers):
            labels[k, 0] = j + 1
            k = k + 1
    labels = np.int16(labels)
    lablePath = os.path.join(dataRoot, fileFolder, 'labels')
    f = h5py.File(lablePath, 'w')
    f.create_dataset('labels', data = labels)
    f.close()
    pass
#------main----------------   
dataRoot = '/home/dxl/myPro/faceData'
fileFolder = 'depthImage'
pName = 'depthImage'
lName = 'labels'

depthImage = os.path.join(dataRoot, fileFolder, pName)
f1 = h5py.File(depthImage, 'r')
depthImage = f1.get(pName)[:]
numPers = np.size(depthImage, 0)

labelPath = os.path.join(dataRoot, fileFolder, lName)
f2 = h5py.File(labelPath, 'r')
labels = f2.get(lName)[:]

pt = depthImage[0, :, :]




    

    
    
        
            
            
        
        

 





