# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:25:17 2018

@author: DXL
"""

import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import time

#-----获取同一个人的n个表情-------
def get_same_personImage():
    st = time.clock()
    dataRoot = 'E:\\python\\faceData'
    dataFolder = 'multiNum50'   
    getClass = 0
    saveFolder = str(getClass)
    dataName = 'depthImageV1'
    getNum = 47
    dataPath = os.path.join(dataRoot, dataFolder, dataName)
    
    f = h5py.File(dataPath, 'r')
    data = f.get(dataName)[:]
    f.close()
    
    numPers, h, w = np.shape(data)
    for i in range(getNum):
        ii = i*150 + getClass
        imageT = data[ii, :, :]
        saveName = str(getClass) + str(ii) + '.png'
        savePath = os.path.join(dataRoot, dataFolder, saveFolder, saveName)
        plt.imsave(savePath, imageT, format = 'png')
        print('The %d th image has been saved...' % i)
        pass
    end = time.clock()
    print(end-st)
    pass
#-----获取一小部分depthImaegV1-----
def get_test_small_dataSet():
    st = time.clock()
    dataRoot = 'E:\\python\\faceData'
    dataFolder = 'multiNum50'
    dataName = 'depthImageV1'
    saveFolder = 'test50'
    saveName = 'testData'
    dataPath = os.path.join(dataRoot, dataFolder, dataName)

    testNumPers = 40
    testNumPress = 3
    testNumTest = 1
    
    f = h5py.File(dataPath, 'r')
    data = f.get(dataName)[:]
    f.close()
    
    imageT = data[0, :, :]
    h, w = np.shape(imageT)
    trainImage = np.zeros([testNumPers*testNumPress, h, w])
    testImage = np.zeros([testNumPers*testNumTest, h, w])
    trainLabel = np.zeros([testNumPers*testNumPress])
    testLabel = np.zeros([testNumPers*testNumTest])
    k1 = 0
    k2 = 0
    
    for i in range(testNumPers):
        for j in range(testNumPress):
            ii = j*150 + i
            imageT = data[ii, :, :]
            trainImage[k1, :, :] = imageT
            trainLabel[k1] = i
            k1 = k1 + 1
        
        for k in range(testNumTest):
            ii = (testNumPress+k)*150 + i
            imageT = data[ii, :, :]
            testImage[k2, :, :] = imageT
            testLabel[k2] = i
            k2 = k2 + 1
    
    trainDataName = 'trainImage'
    testDataName = 'testName'
    trainLabelName = 'trainLabel'
    testLabelName = 'testLabel'
    
    savePath = os.path.join(dataRoot, saveFolder, saveName)
    
    if os.path.exists(savePath):
        os.remove(savePath)
        print('The old %s has been removed...' % saveName)
        
    f = h5py.File(savePath, 'w')
    f.create_dataset(trainDataName, data = trainImage)
    f.create_dataset(testDataName, data = testImage)
    f.create_dataset(trainLabelName, data = trainLabel)
    f.create_dataset(testLabelName, data = testLabel)
    f.close()
    print('The new %s has been saved...' % saveName)
    end = time.clock()
    print('cost %.4f seconds...' % (end - st))
    pass

#-----获取一小部分三维点云数据----
def get_part_pointCloud():
    st = time.clock()
    dataRoot = 'E:\\python\\faceData'
    dataFolder = 'multiNum50'
    dataName = 'frontalFace'
    saveFolder = 'test50'
    saveName = 'pointCloud'
    dataPath = os.path.join(dataRoot, dataFolder, dataName)
    savePath = os.path.join(dataRoot, saveFolder, saveName)

    partPerNum = 5
    partPressNum = 2
    
    f = h5py.File(dataPath, 'r')
    data = f.get(dataName)[:]
    f.close()
    
    numPers, numPoints, numDim = np.shape(data)
    pointCloud = np.zeros([partPerNum*partPressNum, numPoints, numDim])
    
    k = 0
    for i in range(partPerNum):
        for j in range(partPressNum):
            ii = j*150 + i
            imageT = data[ii, :, :]
            pointCloud[k, :, :] = imageT
            k = k + 1
            print('The %d th face has been saved...' % k)
    if os.path.exists(savePath):
        os.remove(savePath)  
        
    f = h5py.File(savePath, 'w')
    f.create_dataset(saveName, data = pointCloud)
    f.close()
    end = time.clock()
    print('cost %d seconds' % (end-st))    
    return pointCloud
#-----获取鼻尖点附近以上的刚性区域----
def get_part_face(data):   
    numPers, h, w = np.shape(data)
    dataPro = np.zeros([numPers, 15, 15])
    for i in range(numPers):
        pt = data[i, :, :]
        ptT = pt[67:82, 49:64]
        dataPro[i, :, :] = ptT
    return dataPro
    print('finshed..')




    
    














