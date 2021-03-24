import cv2
import numpy
from matplotlib import pyplot

def printImg(img):
    pyplot.imshow(img)
    pyplot.show()

img = cv2.imread('rotated_cat.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

edges = cv2.Canny(img,20,50,L2gradient = True)

corners = cv2.cornerHarris(edges, 3, 3, 0.01)

printImg(corners)
