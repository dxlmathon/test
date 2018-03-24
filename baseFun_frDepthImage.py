# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 14:56:22 2018

@author: Babideng
"""

import numpy as np
import copy
from sklearn.neighbors import NearestNeighbors
from skimage import color
import matplotlib.pyplot as plt
from skimage import morphology
import math

#----------------------
def get_n_neibor_ind(oneFaceS, pointNum):
    """
    get pointNum neibors' index of bj point
    """
    oneFace = copy.deepcopy(oneFaceS)
    bjInd = get_bj_ind(oneFace)
    bj = oneFace[bjInd, :]
    bj = bj.reshape(1, -1)
    NN = NearestNeighbors(n_neighbors = pointNum)
    NN.fit(oneFace)
    dis, ind = NN.kneighbors(bj)
    return ind
    pass  
#----------------------
def get_bj_ind(oneFace):
    """get bj point index in the zju database"""
    z = oneFace[:, 2]
    listz = list(z)
    bjInd = listz.index(max(listz))
    return bjInd
    pass

#----------------------
def get_frontalFace(allFaceS, ind):
    """
    extract size(index) point of all face data
    """
    allFace = copy.deepcopy(allFaceS)
    numPoint = np.size(ind)
    numPress, numPers, temp1, temp2 = np.shape(allFace)
    numFaces = numPress * numPers
    frontalFace = np.zeros([numFaces, numPoint, temp1])
    c1 = 0
    for i in range(numPress):
        for j in range(numPers):
            tempF = allFace[i, j, :, :]
            tempF = tempF.T
            for k in range(numPoint):     
                indT = ind[0, k]
                frontalFace[c1, k, :] = tempF[indT, :]
            print('The %d frontalFace is produced' % (i*numPers+j+1))
            c1 = c1 + 1
    return frontalFace
#----------------------
def get_label(numPress, numPers):
    label = np.zeros([numPress*numPers])
    k = 0
    for i in range(numPress):
        for j in range(numPers):
            label[k] = j
            k = k + 1
    label = np.uint8(label)
    return label
    pass
#----------------------
def move_zomm_3Ddata(frontalFaceS, multiNum):
    """
    align all face' bj point
    make x,y coordinate a positive num
    multiple x,y coordinate with a num
    """
    frontalFace = copy.deepcopy(frontalFaceS)
    numPers, numPoints, temp1 = np.shape(frontalFace)
    xMin = 0
    yMin = 0

    for i in range(numPers):
        frontalFace[i, :, 0] = frontalFace[i, :, 0] * multiNum
        frontalFace[i, :, 1] = frontalFace[i, :, 1] * multiNum
        pt = frontalFace[i, :, :]
        x = pt[:, 0]
        y = pt[:, 1]      
        xMinT = min(x)
        yMinT = min(y)
        if xMinT < xMin:
            xMin = xMinT
        if yMinT < yMin:
            yMin = yMinT
    xMin = np.ceil(xMin)
    yMin = np.ceil(yMin)
     
    bjxMax = 0
    bjyMax = 0
    for j in range(numPers):
        frontalFace[j, :, 0] = frontalFace[j, :, 0] - xMin
        frontalFace[j, :, 1] = frontalFace[j, :, 1] - yMin
        pt = frontalFace[j, :, :]
        x = pt[:, 0]
        y = pt[:, 1]      
        bjx = x[0]
        bjy = y[0]
        if bjx > bjxMax:
            bjxMax = bjx
        if bjy > bjyMax:
            bjyMax = bjy

    for k in range(numPers):
        bjx = frontalFace[k, 0, 0]
        bjy = frontalFace[k, 0, 1]
        frontalFace[k, :, 0] = frontalFace[k, :, 0] + bjxMax - bjx
        frontalFace[k, :, 1] = frontalFace[k, :, 1] + bjyMax - bjy
        print('The %d dealed face data is produced!' % k)
    return frontalFace
    pass

#----------------------
def depth_data_deal(frontalFaceT):
    """
    move bjz to the axis 1
    """
    frontalFace = copy.deepcopy(frontalFaceT)
    numPers, numPoints, temp1 = np.shape(frontalFace)
    for i in range(numPers):
        zMax = frontalFace[i, 0, 2]
        zDis = 1 - zMax
        for j in range(numPoints): 
            frontalFace[i, j, 2] = frontalFace[i, j, 2] + zDis
    return frontalFace
    pass
#----------------------
def orthogonal_projection(frontalFaceS):
    """
    yield depth image with orthogonal projection
    """
    frontalFace = copy.deepcopy(frontalFaceS)
    numPers, numPoints, temp1 = np.shape(frontalFace)
    xMax = 0
    yMax = 0
    
    for i in range(numPers):
        x = frontalFace[i, :, 0]
        y = frontalFace[i, :, 1]
        xMaxT = max(x)
        yMaxT = max(y)
        
        if xMaxT > xMax:
            xMax = xMaxT
        if yMaxT > yMax:
            yMax = yMaxT

    xMax = np.ceil(xMax)
    xMax = int(xMax)
    yMax = np.ceil(yMax)
    yMax = int(yMax)
    
    depthMap = np.zeros([numPers, yMax+1, xMax+1])    
    for k in range(numPers):
        pt = frontalFace[k, :, :]
        for i in range(numPoints):
            x = int(round(pt[i, 0]))
            y = int(round(pt[i, 1]))        
            depth = pt[i, 2]
            depthMap[k, y, x] = depth
    
    return depthMap
    pass

#----------------------
def image_corrosion(input2Dimage, dimOpt = 3):
    """
    image corrosion use a matrix which the size is dimOpt*dimOpt
    """
    opt = np.ones([dimOpt, dimOpt], dtype = np.uint8)
    h, w = np.shape(input2Dimage)
    outputImage = np.zeros_like(input2Dimage)
    temp1 = dimOpt//2
    temp2 = dimOpt - temp1 
    for i in range(temp1, h-temp2):
        for j in range(temp1, w-temp2):
            block = input2Dimage[i-temp1:i+temp2, j-temp1:j+temp2]
            blockT = np.ceil(block)
            blockT = np.uint8(blockT)
            out = opt & blockT
            if sum(sum(out)) == dimOpt*dimOpt:
                outputImage[i, j] = input2Dimage[i, j]
            else:
                outputImage[i, j] = 0
    return outputImage
    pass
#----------------------
def image_dilation(input2Dimage, dimOpt = 3):
    """
    image dilation use a matrix which the size is dimOpt*dimOpt
    """
    opt = np.ones([dimOpt, dimOpt], dtype = np.uint8)
    h, w = np.shape(input2Dimage)
    outputImage = np.zeros_like(input2Dimage)
    temp1 = dimOpt//2
    temp2 = dimOpt - temp1
    for i in range(temp1, h-temp2):
        for j in range(temp1, w-temp2):
            block = input2Dimage[i-temp1:i+temp2, j-temp1:j+temp2]
            blockT = np.ceil(block)
            blockT = np.uint8(blockT)
            out = opt & blockT
            if sum(sum(out)) == 0:
                outputImage[i, j] = 0
            else:
                outputImage[i, j] = 1
    return outputImage
    pass
#----------------------
def depthMap_empty_interpolation_3D(depthMapT, correspond3Ddata, numNeibor = 3):
    """
    find empty in the area(binarization closing find_empty)
    find neibors, get depth
    """
    depthMap = copy.deepcopy(depthMapT)
    h, w = np.shape(depthMap)
    depthMap1 = np.ceil(depthMap)
    depthMap1 = np.uint8(depthMap1)
    binaryMap = np.ones_like(depthMap, dtype = np.uint8)
    andOut = depthMap1 & binaryMap
    andOut1 = image_dilation(andOut)
    andOut1 = image_corrosion(andOut1)
    face = correspond3Ddata[:, 0:2]
    for i in range(h):
        for j in range(w):
            t1 = depthMap[i, j]
            t2 = andOut1[i, j]
            if t1 == 0 and t2 != 0:
                basePoint = [i, j]
                basePoint = np.array(basePoint)
                basePoint = basePoint.reshape(1, -1)
                NN = NearestNeighbors(n_neighbors = numNeibor)
                NN.fit(face)
                dis, ind = NN.kneighbors(basePoint)
                depth = 0
                for k in range(numNeibor):
                    depth = correspond3Ddata[ind[0, k], 2] + depth
                depth = depth/(numNeibor)
                depthMap[i, j] = depth
                pass
    return depthMap
    pass
#----------------------
def depthMap_empty_interpolation_2D(depthMapT):
    depthMap = copy.deepcopy(depthMapT)
    h, w = np.shape(depthMap)    
    depthMap1 = np.ceil(depthMap)
    depthMap1 = np.uint8(depthMap1)
    binaryMap = np.ones_like(depthMap, dtype = np.uint8)
    andOut = depthMap1 & binaryMap
    andOut1 = image_dilation(andOut, dimOpt = 6)
    andOut1 = image_corrosion(andOut1, dimOpt = 6)
    for i in range(1, h-1):
        for j in range(1, w-1):
            t1 = depthMap[i, j]
            t2 = andOut1[i, j]
            a = np.zeros([1, 4])
            sumDepth = 0
            kk = 0
            if t1 == 0 and t2 != 0:
                a[0, 0] = depthMap[i-1, j]
                a[0, 1] = depthMap[i+1, j]
                a[0, 2] = depthMap[i, j-1]
                a[0, 3] = depthMap[i, j+1]
                for k in range(4):
                    if a[0, k] != 0:
                        sumDepth = sumDepth + a[0, k]
                        kk = kk + 1
                if kk != 0:
                    depth = sumDepth/kk
                    depthMap[i, j] = depth
    return depthMap   
    pass
#----------------------
def circularLBP(img, r = 1, neibor = 8):  
    neibor = max(min(neibor, 31), 1)
    angle = 2 * np.pi / neibor
    angles = np.arange(0, 2 * np.pi, angle)
    dis = np.array([r*np.cos(angles), -r*np.sin(angles)]).T;
    
    [m, n] = np.shape(img)
    minx = min(dis[:, 0])
    maxx = max(dis[:, 0])
    miny = min(dis[:, 1])
    maxy = max(dis[:, 1])
    blockSizeX = np.ceil(maxx) - np.floor(minx) + 1
    blockSizeY = np.ceil(maxy) - np.floor(miny) + 1
    
    lbpImageH = np.int16(m - blockSizeX + 1)
    lbpImageW = np.int16(n - blockSizeY + 1)
    lbpImage = np.zeros([lbpImageH, lbpImageW])   # 存储lbp特征
    
    for j in range(r, m-r):
        for i in range(r, n-r):
            compValue = img[j, i]       # 中心点像素值
            for k in range(8):
                samplePoint = dis[k, :]
                x = samplePoint[0]
                y = samplePoint[1]
                x = np.int8(round(x))
                y = np.int8(round(y))
                spValue = img[y+j, x+i]
                if spValue>compValue:
                    spValue = 1
                else:
                    spValue = 0
                    
                lbpImage[j-r, i-r] = lbpImage[j-r, i-r] + spValue*math.pow(2, k)
    return lbpImage
#----------------------
def circularLBP_BLI(img, r = 1, neibor = 8):  
    neibor = max(min(neibor, 31), 1)
    angle = 2 * np.pi / neibor
    angles = np.arange(0, 2 * np.pi, angle)
    dis = np.array([r*np.cos(angles), -r*np.sin(angles)]).T;
    
    [m, n] = np.shape(img)
    minx = min(dis[:, 0])
    maxx = max(dis[:, 0])
    miny = min(dis[:, 1])
    maxy = max(dis[:, 1])
    blockSizeX = np.ceil(maxx) - np.floor(minx) + 1
    blockSizeY = np.ceil(maxy) - np.floor(miny) + 1
    
    lbpImageH = np.int16(m - blockSizeX + 1)
    lbpImageW = np.int16(n - blockSizeY + 1)
    lbpImage = np.zeros([lbpImageH, lbpImageW])   # 存储lbp特征
    
    for j in range(r, m-r):
        for i in range(r, n-r):
            compValue = img[j, i]          # 中心点像素值
            for k in range(neibor):
                samplePoint = dis[k, :]
                x = samplePoint[0]
                y = samplePoint[1]
                spValue = biLinear_interpolation(img, i, j, y, x)
                if spValue>compValue:
                    spValue = 1
                else:
                    spValue = 0                 
                lbpImage[j-r, i-r] = lbpImage[j-r, i-r] + spValue*math.pow(2, k)
    return lbpImage
#----------------------
def biLinear_interpolation(image, i, j, y, x):
    yr = j + y;
    xr = i + x;
    minY = np.int16(np.floor(yr))
    maxY = np.int16(np.ceil(yr))
    minX = np.int16(np.floor(xr))
    maxX = np.int16(np.ceil(xr))  
    f00 = image[minY, minX]
    f10 = image[minY, maxX]
    f11 = image[maxY, maxX]
    f01 = image[maxY, minX]
    x = xr - minX
    y = yr - minY
    spValue = (1-y)*(1-x)*f00 + x*(1-y)*f01 + y*(1-x)*f10 + x*y*f11
    return spValue
#----------------------
def cylinder_y_projection(frontalFaceS, isOwnDefineR = False, radius = 70):
    frontalFace = copy.deepcopy(frontalFaceS)
    numPers, pointNum, temp = np.shape(frontalFace)
    #-------
    xMax = 0
    yMax = 0
    
    for i in range(numPers):
        x = frontalFace[i, :, 0]
        y = frontalFace[i, :, 1]
        xMaxT = max(x)
        yMaxT = max(y)
        
        if xMaxT > xMax:
            xMax = xMaxT
        if yMaxT > yMax:
            yMax = yMaxT

    xMax = np.ceil(xMax)
    xMax = int(xMax)
    yMax = np.ceil(yMax)
    yMax = int(yMax)   
    #-------    
    if isOwnDefineR:
        radius = radius
    else:
        radius = (xMax + 2)/2
    pt = frontalFace[0, :, :]
    bjx = int(round(pt[0, 0]))
    bjy = int(round(pt[0, 1]))
    cx = int(round(radius*np.pi/2))

    cylinderLbp = np.zeros([numPers, yMax+1, int(np.ceil(np.pi*radius))+1])
    
    for i in range(numPers):
        pt = frontalFace[i, :, :]
        depth = pt[0, 2]
        cylinderLbp[i, bjy, cx] = depth
        for j in range(1,pointNum):
            x = pt[j, 0]
            y = int(round(pt[j, 1]))
            depth = pt[j, 2]
            xd = x - bjx
            if np.abs(xd) < radius:
                angle = math.acos(xd/radius)
                arclen = (np.pi/2-angle)*radius
                xx = cx + arclen
                xx = int(round(xx))
                cylinderLbp[i, y, xx] = depth
            else:
                continue
        print('The %d th cylinderLbp has been yield...' % i)
    return cylinderLbp
    pass
#----------------------
def double_cylinder_y_projection(frontalFaceS, isOwnDefineRout = False, rIn = 10, rOut = 70):
    frontalFace = copy.deepcopy(frontalFaceS)
    numPers, numPoints, temp = np.shape(frontalFace)
    #-------
    xMax = 0
    yMax = 0
    
    for i in range(numPers):
        x = frontalFace[i, :, 0]
        y = frontalFace[i, :, 1]
        xMaxT = max(x)
        yMaxT = max(y)
        
        if xMaxT > xMax:
            xMax = xMaxT
        if yMaxT > yMax:
            yMax = yMaxT

    xMax = np.ceil(xMax)
    xMax = int(xMax)
    yMax = np.ceil(yMax)
    yMax = int(yMax)   
    #------- 
    if isOwnDefineRout:
        rOut = rOut
    else:
        rOut = (xMax + 2)/2
        
    pt = frontalFace[0, :, :]
    bjx = int(round(pt[0, 0]))
    bjy = int(round(pt[0, 1]))
    
    angle = math.acos(rIn/rOut)
    arclen = angle*rOut
    lenMax = np.pi*rIn + arclen*2
    arclenDiff = np.pi*rIn/2 - (np.pi/2-angle)*rOut    
    cx = int(round(lenMax/2))
    cylinderLbp = np.zeros([numPers, yMax+1, int(np.ceil(lenMax))+1])
    
    for i in range(numPers):
        pt = frontalFace[i, :, :]
        depth = pt[0, 2]
        cylinderLbp[i, bjy, cx] = depth
        for j in range(numPoints):
            x = pt[j, 0]
            y = int(round(pt[j, 1]))
            depth = pt[j, 2]
            xd = x - bjx
            if np.abs(xd) < rOut:
                if np.abs(xd) < rIn:
                    angle = math.acos(xd/rIn)
                    arclen = (np.pi/2-angle)*rIn
                    xx = cx + arclen
                    xx = int(round(xx))
                    cylinderLbp[i, y, xx] = depth
                else:
                    angle = math.acos(xd/rOut)
                    if xd > 0:
                        arclen = (np.pi/2-angle)*rOut + arclenDiff
                    else:
                        arclen = (np.pi/2-angle)*rOut - arclenDiff
                    xx = cx + arclen
                    xx = int(round(xx))
                    cylinderLbp[i, y, xx] = depth
            else:
                continue
        print('The %d th double curvature cylinder depthImage has yielded...' % i)
    return cylinderLbp
    pass







          
                
                