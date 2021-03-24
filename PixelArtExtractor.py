import cv2
import numpy
from matplotlib import pyplot

def printImg(img):
    pyplot.imshow(img)
    pyplot.show()

img = cv2.imread('rotated_cat.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

red = img[:,:,0] 
green = img[:,:,1]
blue = img[:,:,2]

block_size = 5
k_size = 3
k = 0.07

red_corners = cv2.cornerHarris(red,block_size,k_size,k)
green_corners = cv2.cornerHarris(green,block_size,k_size,k)
blue_corners = cv2.cornerHarris(blue,block_size,k_size,k)

corners = numpy.add(numpy.add(red_corners, green_corners), blue_corners)

threshold = numpy.where(corners > 0.01*corners.max(), 1, 0)

printImg(threshold)

