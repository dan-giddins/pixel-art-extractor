import cv2
import numpy
from matplotlib import pyplot as plot
import math
import scipy.cluster.hierarchy as hcluster
from collections import Counter

def printImg(in_img):
    plot.imshow(in_img)
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
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# edge detection
edges = cv2.Canny(img,20,50,L2gradient = True)

# HoughLines parms
rho_res = 1/2
theta_res = numpy.pi/(180*2**6)
acc_thresh = 100

# line detection
lines = cv2.HoughLines(edges,rho_res,theta_res,acc_thresh).reshape(-1,2).tolist()

#print(lines)

# display lines
for line in lines:
    rho = line[0]
    theta = line[1]
    a = numpy.cos(theta)
    b = numpy.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    #cv2.line(img,(x1,y1),(x2,y2),(0,0,0),1)

#printImg(img)

# get average angle
angle_sum = 0
for line in lines:
    angle_sum += line[1] % (numpy.pi/2)
avg_angle = angle_sum / len(lines)
print(avg_angle)

# get an average distance between all the lines that are 1 'pixel' apart
line_distances = []
print(len(lines))
for l1 in lines:
    for l2 in lines:
        line_distances.append(abs(abs(l1[0]) - abs(l2[0])))
sorted_line_distances = Counter(line_distances).most_common()
valid_lengths = list(filter(lambda x : x[0] > 5 and x[0] < 20 and x[1] > len(lines)/2, sorted_line_distances))
sorted_valid_lenghts =  Counter(valid_lengths).most_common()
length_sum = 0
count = 0
for length in valid_lengths:
    length_sum += length[0] * length[1]
    count +=length[1]
avg_distance = length_sum / count
print(avg_distance)

# get the average pixel offset for x and y
offset_sum_x = 0
offset_sum_y = 0
for line in lines:
    if line[1] < numpy.pi:
        offset_sum_y += line[0] % avg_distance
    else:
         offset_sum_x += line[0] % avg_distance
avg_offset_x = offset_sum_x / len(lines)
avg_offset_y = offset_sum_y / len(lines)
print(avg_offset_x)
print(avg_offset_y)

# get pixel cords
pixel_cords = []
pixel_width = 200
pixel_height = 200
pixel_image = numpy.full((pixel_width, pixel_height, 3), [255, 255, 255])
h, w, c = img.shape
cos = numpy.cos(avg_angle - numpy.pi/2)
sin = numpy.sin(avg_angle - numpy.pi/2)
pixel_offset_x = avg_offset_x / avg_distance
pixel_offset_y = avg_offset_y / avg_distance
# 0.5 as we want center of 'pixel' from original image
for pixel_y in range(pixel_height):
    for pixel_x in range(pixel_width):
        pixel_x_unit = pixel_x + 0.5 + pixel_offset_x - pixel_width/2
        pixel_y_unit = pixel_y + 0.5 + pixel_offset_y - pixel_height/2
        # get unit cords
        x_unit = (pixel_x_unit * cos) - (pixel_y_unit * sin)
        y_unit = (pixel_x_unit * sin) + (pixel_y_unit * cos)
        # scale up
        x = int(avg_distance * x_unit)
        y = int(avg_distance * y_unit)
        if (x < w and x >= 0 and y < h and y >= 0):
            pixel_cords.append((x, y))
            #print(pixel)
            pixel_image[pixel_y, pixel_x] = img[y, x]

#showPointsOnImg(img, pixel_cords)

#printImg(pixel_image)

# crop to 1 pixel more that image (assume background is white)
top = pixel_height
bottom = 0
left = pixel_width
right = 0
for y in range(pixel_height):
    for x in range(pixel_width):
        if not numpy.array_equal(pixel_image[y, x], [255, 255, 255]):
            if x < left:
                left = x
            if x > right:
                right = x
            if y < top:
                top = y
            if y > bottom:
                bottom = y
pixel_image_crop = numpy.full((bottom - top + 3, right - left + 3, 3), [255, 255, 255])
h, w, c = pixel_image_crop.shape
for y in range(h):
    for x in range(w):
        pixel_image_crop[y, x] = pixel_image[y + top - 1, x + left - 1]

# flood image to remove background
mask = numpy.zeros((h+2, w+2), numpy.uint8)
cv2.floodFill(pixel_image_crop, mask, (0,0), [0, 0, 0], loDiff=1000, upDiff=1000)
#printImg(mask)
printImg(pixel_image_crop)

#pixel_image = cv2.cvtColor(pixel_image, cv2.COLOR_RGBA2BGRA)
#print(cv2.imwrite('C:\\Users\\Proto\\OneDrive\\Pictures\\pixel_cat\\pixel_cat_fixed.png', pixel_image))

# # goodFeaturesToTrack parms
# max_corners = 0
# quality_level = 0.01
# min_distance = 0
# block_size = 3

# # get features
# red_corners = convertToSet(cv2.goodFeaturesToTrack(red,max_corners,quality_level,min_distance,block_size))
# green_corners = convertToSet(cv2.goodFeaturesToTrack(green,max_corners,quality_level,min_distance,block_size))
# blue_corners = convertToSet(cv2.goodFeaturesToTrack(blue,max_corners,quality_level,min_distance,block_size))

# # union channel sets together
# corners = red_corners.union(green_corners).union(blue_corners)

# #showPointsOnImg(img, corners)

# # clustering
# corner_array = convertToArray(corners)
# corner_clusters = hcluster.fclusterdata(corner_array, 3, criterion="distance")

# corner_groups = [[] for _ in range(corner_clusters.max())]
# for i in range(len(corner_array)):
#     corner_groups[corner_clusters[i]-1].append(corner_array[i])

# grouped_corners = set()
# for group in corner_groups:
#     x = 0
#     y = 0
#     for point in group:
#         x += point[0]
#         y += point[1]
#     x /= len(group)
#     y /= len(group)
#     grouped_corners.add((x, y))

# showPointsOnImg(img, grouped_corners)

# scale_factor = 6
# scaled_corners = set()
# for c in grouped_corners:
#     scaled_corners.add((c[0]*scale_factor, c[1]*scale_factor))
# h, w, c = img.shape
# scaled_img = cv2.resize(img, (w * scale_factor, h * scale_factor))

# #showPoints(scaled_img, scaled_corners)


