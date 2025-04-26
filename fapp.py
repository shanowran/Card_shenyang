import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import os
from PIL import Image



# 1. 特征检测和匹配
def featuresDetectAndMatch(img1, img2):
    """特征点检测和计算描述子"""
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    return keypoints1, keypoints2, matches

# 2. 比例测试
def ratio_match(knnMatches, r=0.7518):
    """比例测试筛选匹配"""
    goodMatches = []
    for m, n in knnMatches:
        if m.distance < r * n.distance:
            goodMatches.append(m)
    return goodMatches

# 3. 数据预处理
def getPreData(goodMatches, keypoints1, keypoints2):
    """提取匹配点的坐标"""
    X = np.array([keypoints1[m.queryIdx].pt for m in goodMatches])
    Y = np.array([keypoints2[m.trainIdx].pt for m in goodMatches])
    return X, Y

# 4. 预处理
def preTreat(query, train):
    """创建簇点坐标"""
    y = train[:, 1] - query[:, 1]
    xtemp = (train[:, 0] + np.max(query[:, 0])) - query[:, 0]
    dist = np.sqrt(y**2 + xtemp**2)
    sin = ((y / dist) + 1) * (np.max(dist) / 2)
    coordinate = np.vstack((sin, dist)).T
    return coordinate

# 5. 计算角度
def get_angle(x1, y1, x2, y2, x3, y3):
    """计算两线段之间的角度"""
    theta1 = np.abs(np.arctan2(y1 - y3, x1 - x3) * 180 / np.pi)
    theta2 = np.abs(np.arctan2(y2 - y3, x2 - x3) * 180 / np.pi)
    return theta1 + theta2

# 6. 计算筛选阈值
def cal_threshold(coordinate, interval):
    """计算筛选阈值"""
    gridLen = np.ceil(np.max(coordinate) / interval)
    pointsGridCoordinate = (coordinate / gridLen).astype(int)
    pointsGridInd = pointsGridCoordinate[:, 0] * interval + pointsGridCoordinate[:, 1]
    
    count = Counter(pointsGridInd)
    numList = sorted(count.values(), reverse=True)
    
    if len(numList) > 5:
        partNum = sum(numList[:5])
    else:
        return 2
    
    dataDensity = partNum / coordinate.shape[0]
    
    minAngle = 360
    threshold = 0
    x1, y1 = 0, numList[0]
    x2, y2 = len(numList) - 1, numList[-1]
    
    for i in range(len(numList) - 2):
        x3, y3 = i + 1, numList[i + 1]
        angle = get_angle(x1, y1, x2, y2, x3, y3)
        if angle < minAngle:
            threshold = numList[i + 1]
            minAngle = angle
    
    if threshold > dataDensity * 10:
        threshold = np.ceil(threshold * dataDensity)
    
    return int(threshold + 1)

# 7. 网格划分
def simple_grid(coordinate, interval, threshold):
    """划分网格并筛选正确索引"""
    gridLen = np.ceil(np.max(coordinate) / interval)
    pointsGridCoordinate = (coordinate / gridLen).astype(int)
    pointsGridInd = pointsGridCoordinate[:, 0] * interval + pointsGridCoordinate[:, 1]
    
    mapVec = defaultdict(list)
    for i, ind in enumerate(pointsGridInd):
        mapVec[ind].append(i)
    
    correctIndex = []
    for vect in mapVec.values():
        if len(vect) >= threshold:
            correctIndex.extend(vect)
    return correctIndex

# 8. 显示匹配结果
def showMatchResult(img1, img2, keypoints1, keypoints2, matches, windName):
    """显示匹配结果"""
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, 
                                 matchColor=(0, 255, 255),  # 黄色线条
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(windName)
    plt.axis('off')
    plt.show()

# 9. FAPP 方法
def FAPP(query, train, keypoints1, keypoints2, putativeMatches):
    """执行 FAPP 算法"""
    coordinate = preTreat(query, train)
    pointNum = coordinate.shape[0]
    gridNum = int(np.sqrt(pointNum) * 2)
    threshold = cal_threshold(coordinate, gridNum)
    correctInd = simple_grid(coordinate, gridNum, threshold)
    return correctInd

# 10. 运行 FAPP
def runFAPP(keypoints1, keypoints2, putativeMatches):
    """运行 FAPP 并返回最终匹配"""
    query, train = getPreData(putativeMatches, keypoints1, keypoints2)
    correctInd = FAPP(query, train, keypoints1, keypoints2, putativeMatches)
    finalMatches = [putativeMatches[i] for i in correctInd]
    return finalMatches