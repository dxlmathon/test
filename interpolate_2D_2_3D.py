# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 13:42:05 2018

对深度图进行插值（3D-2D-3D）

@author: Babideng
"""

import os
import h5py
import numpy as np
import copy
import matplotlib.pyplot as plt
import baseFun_frDepthImage as _baseFun
import baseFun_dataProcess as _baseFun_dp
from skimage import morphology
import argparse
from sklearn.cross_validation import ShuffleSplit
from sklearn import svm
import time
import winsound

#----------------
pointNum = 6000
multiNum = 50
#----------------
dataRoot = 'E:\\python\\faceData'
dataFolder = 'multiNum' + str(multiNum)
dataFolder3D = '3D'
dataFolderDM = 'depthImage'
face3DName = 'Vertices.mat'

savePathF = os.path.join(dataRoot, dataFolder, 'frontalFace')
savePathL = os.path.join(dataRoot, dataFolder, 'label')
#-------get point cloud-----
if os.path.exists(savePathF):
    temp4 = h5py.File(savePathF, 'r')
    frontalFace = temp4.get('frontalFace')[:]
    temp4.close()
else:   
    face3DPath = os.path.join(dataRoot, dataFolder3D, face3DName)
    temp1 = h5py.File(face3DPath, 'r')
    allFace = temp1.get('Vertices')[:]
    temp1.close()
    
    oneFace = allFace[0, 1, :, :]
    oneFace = oneFace.T
    
    pInd = _baseFun.get_n_neibor_ind(oneFace, pointNum)
    frontalFace = _baseFun.get_frontalFace(allFace, pInd)
       
    temp2 = h5py.File(savePathF, 'w')
    temp2.create_dataset('frontalFace', data = frontalFace)
    temp2.close()
print('The frontalFace has loaded done...')
#-------get label---------
if os.path.exists(savePathL):
    temp5 = h5py.File(savePathL, 'r')
    labelZju = temp5.get('labelZju')[:]
    temp5.close()
else:
    labelZju = _baseFun.get_label(47, 150)
    temp3 = h5py.File(savePathL, 'w')
    temp3.create_dataset('labelZju', data = labelZju)
    temp3.close()
    pass
print('The labelZju has loaded done...')
#-------get depthImage_pre-----
saveNameDM = 'depthMap'
savePathDM = os.path.join(dataRoot, dataFolder, saveNameDM)

if os.path.exists(savePathDM):
    h1 = h5py.File(savePathDM, 'r')
    depthMap = h1.get(saveNameDM)[:]
    h1.close()
else:
    ffMoveZoom = _baseFun.move_zomm_3Ddata(frontalFace, multiNum)
    fface = _baseFun.depth_data_deal(ffMoveZoom)
    depthMap = _baseFun.orthogonal_projection(fface)
    h2 = h5py.File(savePathDM, 'w')
    h2.create_dataset(saveNameDM, data = depthMap)
    h2.close()
    print('Saving depthMap has done!----------')
    pass
print('The depthMap has loaded done...')
#-------get depthImage dealed-----
saveNameDI = 'depthImageV1'
savePathDI = os.path.join(dataRoot, dataFolder, saveNameDI)

if os.path.exists(savePathDI):
    h1 = h5py.File(savePathDI, 'r')
    depthImageV1 = h1.get(saveNameDI)[:]
    h1.close()
else:
    depthImageV1 = np.zeros_like(depthMap)
    numPers, temp1, temp2 = np.shape(depthImageV1)
    for i in range(numPers):
        imageT = depthMap[i, :, :]
        imageT = _baseFun.depthMap_empty_interpolation_2D(imageT)
        depthImageV1[i, :, :] = imageT
        print('The %d depthImageV1 is produced!' % i)
    h2 = h5py.File(savePathDI, 'w')
    h2.create_dataset(saveNameDI, data = depthImageV1)
    h2.close()
    print('Saving depthImageV1 has done!-------')
    pass
print('The depthMapV1 has loaded done...')
#-------get LBP feature-----
saveNameLbp = 'depthImageLbp'
savePathLbp = os.path.join(dataRoot, dataFolder, saveNameLbp)

if os.path.exists(savePathLbp):
    f = h5py.File(savePathLbp, 'r')
    depthImageLbp = f.get(saveNameLbp)[:]
    f.close()
else:
    numPers, h, w = np.shape(depthImageV1)
    temp = depthImageV1[0, :, :]
    temp = _baseFun.circularLBP(temp, r = 1, neibor = 8)
    lbpH, lbpW = np.shape(temp)
    depthImageLbp = np.zeros([numPers, lbpH, lbpW])
    for i in range(numPers):
        picT = depthImageV1[i, :, :]
        lbpT = _baseFun.circularLBP_BLI(picT, r = 1, neibor = 8)
        depthImageLbp[i, :, :] = lbpT
    
    f = h5py.File(savePathLbp, 'w')
    f.create_dataset(saveNameLbp, data = depthImageLbp)
    f.close()
    pass
print('The depthImageLbp has loaded done...')
#-----获取曲面lbp插值版------
saveNameCLbp = 'cylinderLbpV1'
savePathCLbp = os.path.join(dataRoot, dataFolder, saveNameCLbp)

if os.path.exists(savePathCLbp):
    f = h5py.File(savePathCLbp, 'r')
    cylinderLbp = f.get(saveNameCLbp)[:]
    f.close()
else:
    temp = _baseFun.move_zomm_3Ddata(frontalFace, multiNum)
    temp = _baseFun.depth_data_deal(temp)
    cylinderLbp = _baseFun.cylinder_y_projection(temp)
    numPers, h, w = np.shape(cylinderLbp)
    cylinderLbpV1 = np.zeros_like(cylinderLbp)
    for i in range(numPers):
        clbp = cylinderLbp[i, :, :]
        clbpT = _baseFun.depthMap_empty_interpolation_2D(clbp)
        cylinderLbpV1[i, :, :] = clbpT
    f = h5py.File(savePathCLbp, 'w')
    f.create_dataset(saveNameCLbp, data = cylinderLbpV1)
    f.close()
    pass
print('The dealed cylinderLbp has been loaded...')
#-----获取双曲率曲面lbp插值版------
saveNameCLbp = 'dCylinderDepthImage'
savePathCLbp = os.path.join(dataRoot, dataFolder, saveNameCLbp)

if os.path.exists(savePathCLbp):
    f = h5py.File(savePathCLbp, 'r')
    dCylinderDepthImage = f.get(saveNameCLbp)[:]
    f.close()
else:
    temp = _baseFun.move_zomm_3Ddata(frontalFace, multiNum)
    temp = _baseFun.depth_data_deal(temp)
    dCylinderDepthImage = _baseFun.double_cylinder_y_projection(temp)
    numPers, h, w = np.shape(cylinderLbp)
    dCylinderDepthImageV1 = np.zeros_like(dCylinderDepthImage)
    for i in range(numPers):
        clbp = dCylinderDepthImage[i, :, :]
        clbpT = _baseFun.depthMap_empty_interpolation_2D(clbp)
        dCylinderDepthImageV1[i, :, :] = clbpT
    f = h5py.File(savePathCLbp, 'w')
    f.create_dataset(saveNameCLbp, data = dCylinderDepthImage)
    f.close()
    pass
print('The dealed dCylinderDepthImage has been loaded...')
#-----获取曲率曲面lbp插值-部分面片----
oriDataName = 'cylinderLbpV1'
dataPath = os.path.join(dataRoot, dataFolder, oriDataName)

f = h5py.File(dataPath, 'r')
oriData = f.get(oriDataName, 'r')[:]
f.close()

partData = _baseFun_dp.get_part_face(oriData)
#--------------------
dataT = partData
numPers, cols, rows = np.shape(dataT)
#--------------------
def train_test_split(args):
    rs = ShuffleSplit(n = args.numPers, n_iter = args.niter, \
      test_size = args.testRate, random_state = None)
    trainIndex = []
    testIndex = []
    for trainI, testI in rs:
        trainIndex.append(trainI)
        testIndex.append(testI)
    return trainIndex, testIndex
    pass
#---------------------
parser = argparse.ArgumentParser(description = 'face recognition via svm')
parser.add_argument('-np', dest = 'numPers', type = int, default = numPers)
parser.add_argument('-tr', dest = 'testRate', type = float, default = 0.3)
parser.add_argument('-it', dest = 'niter', type = int, default = 1)
parser.add_argument('-in', dest = 'iterNum', type = int, default = 1)
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
    pt = dataT[ii, :, :]  
    pt = np.reshape(pt, [1, cols*rows])
    trainImage[i, :] = pt
    labelT = labelZju[ii]      
    trainLabel[i] = labelT
    
for j in range(numTest):
    jj = testIndex[0][j]
    pt = dataT[jj, :, :]
    pt = np.reshape(pt, [1, cols*rows])
    testImage[j, :] = pt
    labelT = labelZju[jj]
    testLabel[j] = labelT

st = time.clock()   
svmc = svm.LinearSVC(random_state = 1)
svmClassifier = svmc.fit(trainImage, trainLabel)
print('fit done!')

assert svmClassifier is not None, 'There is no train result'
testPic = testImage[0, :]
testPic1 = testPic.reshape(1, -1)
pre = svmClassifier.predict(testImage)

end = time.clock()
costTime = (end - st)/60

count = 0
for i in range(numTest):
    tlab = testLabel[i]
    preLab = pre[i]
    if tlab == preLab:
        count = count + 1
accuRate = count/numTest

print(accuRate)
print('SVM cost %.2f min...' % costTime)
winsound.Beep(600, 1000)




