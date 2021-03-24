import cv2
import numpy as np
from matplotlib import pyplot as plot
import math
import scipy.cluster.hierarchy as hcluster

def printImg(img):
    plot.imshow(img)
    plot.show()

def showFeatures(img, features):
    for feature in features:
        cv2.circle(img, (int(feature[0,0]), int(feature[0,1])), 8, (0,0,255))

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

def convertToArray(s):
    array = []
    for i in s:
        array.append([i[0], i[1]])
    return array

def groupPoints(corners):
    theshold_distance = 4
    grouped_corners = set()
    for c1 in corners:
        local_corners = set()
        for c2 in corners:
            if (math.hypot(c2[0] - c1[0], c2[1] - c1[1]) < theshold_distance):
                local_corners.add(c2)
        x = 0
        y = 0
        for c in local_corners:
            x += c[0]
            y += c[1]
        x /= len(local_corners)
        y /= len(local_corners)
        grouped_corners.add((x, y))
    return grouped_corners

def showPointsOnImg(img, points):
    for point in points:
        img[int(point[1]), int(point[0])] = [0, 0, 0]
    printImg(img)

def rotate(point, angle, origin = (0, 0)):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

# read source img
img = cv2.imread('rotated_cat.png')

# split into channels
red = img[:,:,2] 
green = img[:,:,1]
blue = img[:,:,0]

# goodFeaturesToTrack parms
max_corners = 0
quality_level = 0.01
min_distance = 0
block_size = 3

# get features
red_corners = convertToSet(cv2.goodFeaturesToTrack(red,max_corners,quality_level,min_distance,block_size))
green_corners = convertToSet(cv2.goodFeaturesToTrack(green,max_corners,quality_level,min_distance,block_size))
blue_corners = convertToSet(cv2.goodFeaturesToTrack(blue,max_corners,quality_level,min_distance,block_size))

# union channel sets together
corners = red_corners.union(green_corners).union(blue_corners)

#showPointsOnImg(img, corners)

# clustering
corner_array = convertToArray(corners)
corner_clusters = hcluster.fclusterdata(corner_array, 3, criterion="distance")

corner_groups = [[] for _ in range(corner_clusters.max())]
for i in range(len(corner_array)):
    corner_groups[corner_clusters[i]-1].append(corner_array[i])

grouped_corners = set()
for group in corner_groups:
    x = 0
    y = 0
    for point in group:
        x += point[0]
        y += point[1]
    x /= len(group)
    y /= len(group)
    grouped_corners.add((x, y))

showPointsOnImg(img, grouped_corners)

scale_factor = 6
scaled_corners = set()
for c in grouped_corners:
    scaled_corners.add((c[0]*scale_factor, c[1]*scale_factor))
h, w, c = img.shape
scaled_img = cv2.resize(img, (w * scale_factor, h * scale_factor))

#showPoints(scaled_img, scaled_corners)


