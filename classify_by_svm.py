# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:34:32 2018

@author: dxl
"""

import os
import h5py
import numpy as np
import argparse
from sklearn.cross_validation import ShuffleSplit
from sklearn import svm

#--------------------
def train_test_split(args):
    rs = ShuffleSplit(n = args.numPers, n_iter = args.niter, \
      test_size = args.testRate, random_state = 1)
    trainIndex = []
    testIndex = []
    for trainI, testI in rs:
        trainIndex.append(trainI)
        testIndex.append(testI)
    return trainIndex, testIndex
    pass
#------main----------
dataRoot = '/home/dxl/myPro/faceData'
fileFolder = 'depthImage'
pName = 'depthImage'
lName = 'labels'

depthImage = os.path.join(dataRoot, fileFolder, pName)
f1 = h5py.File(depthImage, 'r')
depthImage = f1.get(pName)[:]
numPers, cols, rows = np.shape(depthImage)

labelPath = os.path.join(dataRoot, fileFolder, lName)
f2 = h5py.File(labelPath, 'r')
labels = f2.get(lName)[:]

parser = argparse.ArgumentParser(description = 'face recognition via svm')
parser.add_argument('-np', dest = 'numPers', type = int, default = numPers)
parser.add_argument('-tr', dest = 'testRate', type = float, default = 0.3)
parser.add_argument('-it', dest = 'niter', type = int, default = 1)
parser.add_argument('-in', dest = 'iterNum', type = int, default = 30)
args = parser.parse_args()


trainIndex, testIndex = train_test_split(args)
numTrain = np.size(trainIndex)
numTest = np.size(testIndex)

trainImage = np.zeros([numTrain, cols*rows])
trainLabel = np.zeros([numTrain])
testImage = np.zeros([numTest, cols*rows])
testLabel = np.zeros([numTest])

for i in range(numTrain):
    ii = trainIndex[0][i]
    pt = depthImage[ii, :, :]  
    pt = np.reshape(pt, [1, cols*rows])
    trainImage[i, :] = pt
    label = labels[ii, 0]        
    trainLabel[i] = label
    
for j in range(numTest):
    jj = testIndex[0][j]
    pt = depthImage[jj, :, :]
    pt = np.reshape(pt, [1, cols*rows])
    testImage[j, :] = pt
    label = labels[jj, 0]
    testLabel[j] = label
    
svmc = svm.LinearSVC(random_state = 1)
svmClassifier = svmc.fit(trainImage, trainLabel)

assert svmClassifier is not None, 'There is no train result'
testPic = testImage[0, :]
testLab = testLabel[0]
testPic1 = testPic.reshape(1, -1)
pre = svmClassifier.predict(testImage)

count = 0
for i in range(numTest):
    tlab = testLabel[i]
    preLab = pre[i]
    if tlab == preLab:
        count = count + 1
accuRate = count/numTest
print(accuRate)



    
    
    
    
    
    
