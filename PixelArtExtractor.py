import cv2 as cv
import numpy as np
from matplotlib import pyplot as plot

def printImg(img):
    plot.imshow(img)
    plot.show()

def showFeatures(img, features):
    for feature in features:
        cv.circle(img, (int(feature[0,0]), int(feature[0,1])), 8, (0,0,255))

img = cv.imread('rotated_cat.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

red = img[:,:,0] 
green = img[:,:,1]
blue = img[:,:,2]

max_corners = 0
qualityLevel = 0.01
minDistance = 0


red_corners = set(map(tuple, cv.goodFeaturesToTrack(red,max_corners,qualityLevel,minDistance).reshape(-1,2)))
green_corners = set(map(tuple, cv.goodFeaturesToTrack(green,max_corners,qualityLevel,minDistance).reshape(-1,2)))
blue_corners = set(map(tuple, cv.goodFeaturesToTrack(blue,max_corners,qualityLevel,minDistance).reshape(-1,2)))

corners = red_corners.union(green_corners).union(blue_corners)

h, w, c = img.shape
corners_img = np.zeros((w, h))
corners_array = np.array(list(corners))
corners_img[corners_array] = 1

printImg(corners_img)

#threshold = np.where(corners > 0.01*corners.max(), 1, 0)



