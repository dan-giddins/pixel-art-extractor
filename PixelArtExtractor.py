import cv2 as cv
import numpy as np
from matplotlib import pyplot as plot

def printImg(img):
    plot.imshow(img)
    plot.show()

def showFeatures(img, features):
    for feature in features:
        cv.circle(img, (int(feature[0,0]), int(feature[0,1])), 8, (0,0,255))

def showPoints(img, points):
    array = []
    for point in points:
        array.append([int(point[0]), int(point[1])])
    h, w, c = img.shape
    output_img = np.zeros((w, h))
    for point in array:
        output_img[point[1],point[0]] = 1
    printImg(output_img)

def convertToSet(features):
    return set(map(tuple, features.reshape(-1,2)))

# read source img
img = cv.imread('rotated_cat.png')

# split into channels
red = img[:,:,2] 
green = img[:,:,1]
blue = img[:,:,0]

# goodFeaturesToTrack parms
max_corners = 0
qualityLevel = 0.01
minDistance = 0

# get features
red_corners = convertToSet(cv.goodFeaturesToTrack(red,max_corners,qualityLevel,minDistance))
green_corners = convertToSet(cv.goodFeaturesToTrack(green,max_corners,qualityLevel,minDistance))
blue_corners = convertToSet(cv.goodFeaturesToTrack(blue,max_corners,qualityLevel,minDistance))

# union channel sets together
corners = red_corners.union(green_corners).union(blue_corners)

# print corners
showPoints(img, corners)

#threshold = np.where(corners > 0.01*corners.max(), 1, 0)



