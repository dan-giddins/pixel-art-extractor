"""Extract the original pixel art from an image."""
import copy
import math
from collections import Counter

import cv2
import numpy
from matplotlib import pyplot


def main():
    """The main entrypoint for the application."""
    image = cv2.imread('rotated_cat.png')
    # print_BGR_image(image)
    edges = cv2.Canny(image, 20, 50, L2gradient=True)
    lines = get_lines(edges)
    # draw_lines(lines, image)
    average_angle_offset = get_angle_offset(lines)
    average_line_distance = get_average_line_distance(lines)
    average_pixel_offset = get_average_pixel_offset(
        lines, average_line_distance)
    pixel_image_and_coordinates = get_pixel_image_and_coordinates(
        image, average_angle_offset, average_pixel_offset, average_line_distance)
    pixel_image = pixel_image_and_coordinates[0]
    # draw_points_on_image(image, pixel_image_and_coordinates[1])
    # filepath = "C:\\Users\\Proto\\OneDrive\\Pictures\\pixel_cat\\"\
    #     "pixel_cat_lines_and_pixels.png"
    # write_image_to_file(image, filepath)
    # print_BGR_image(image)
    pixel_image_crop = crop_image(pixel_image)
    mask = get_background_mask(pixel_image_crop)
    pixel_image_transparent = make_background_transparent(
        pixel_image_crop, mask)
    # print_BGRA_image(pixel_image_transparent)
    create_border(pixel_image_transparent)
    pixel_image_scaled = scale_up(pixel_image_transparent)
    print_bgra_image(pixel_image_scaled)
    # filepath = "C:\\Users\\Proto\\OneDrive\\Pictures\\pixel_cat\\"\
    #     "pixel_cat_fixed_trans_scaled_border_thicker.png"
    # write_image_to_file(pixel_image_scaled, filepath)


def write_image_to_file(image, filepath):
    """Write an image to file."""
    written = cv2.imwrite(filepath, image)
    if written:
        print("Image written to '" + filepath + "'.")
    else:
        print("Error writing file!")


def scale_up(image):
    """Scale up an image."""
    scale = 16
    height, width = get_shape(image)
    scaled_height = height * scale
    scaled_width = width * scale
    pixel_image_scaled = numpy.full(
        (scaled_height, scaled_width, 4), [0, 0, 0, 0])
    for y_pos in range(height):
        for x_pos in range(width):
            for y_offset in range(y_pos * scale, (y_pos + 1) * scale):
                for x_offset in range(x_pos * scale, (x_pos + 1) * scale):
                    pixel_image_scaled[y_offset,
                                       x_offset] = image[y_pos, x_pos]
    return pixel_image_scaled


def create_border(image):
    """Create a border around the item in the image."""
    border_pixels = set()
    checked_pixels = set()
    find_border(image, 0, 0, border_pixels, checked_pixels)
    white_pixel = [255, 255, 255, 255]
    for border_pixel in border_pixels:
        image[border_pixel[1], border_pixel[0]] = white_pixel


def make_background_transparent(image, mask):
    """Make the background of an image transparent using a mask"""
    height, width = get_shape(image)
    pixel_image_transparent = numpy.full((height, width, 4), [0, 0, 0, 0])
    for y_pos in range(height):
        for x_pos in range(width):
            if not mask[y_pos+1, x_pos+1]:
                pixel = image[y_pos, x_pos]
                pixel_image_transparent[y_pos, x_pos] = [
                    pixel[0],
                    pixel[1],
                    pixel[2],
                    255]
    return pixel_image_transparent


def get_background_mask(image):
    """Flood an image to create background mask."""
    height, width = get_shape(image)
    mask = numpy.zeros((height+2, width+2), numpy.uint8)
    diff = 10
    diff_array = [diff, diff, diff]
    cv2.floodFill(image, mask, (0, 0), [
        0, 0, 0], loDiff=diff_array, upDiff=diff_array)
    return mask


def crop_image(image):
    """Crop an image to 1 pixel more that image, assuming the image background is white."""
    height, width = get_shape(image)
    top = height
    bottom = 0
    left = width
    right = 0
    for y_pos in range(height):
        for x_pos in range(width):
            if not numpy.array_equal(image[y_pos, x_pos], [255, 255, 255]):
                if x_pos < left:
                    left = x_pos
                if x_pos > right:
                    right = x_pos
                if y_pos < top:
                    top = y_pos
                if y_pos > bottom:
                    bottom = y_pos
    crop_h = bottom - top + 3
    crop_w = right - left + 3
    pixel_image_crop = numpy.full((crop_h, crop_w, 3), [255, 255, 255])
    for y_pos in range(crop_h):
        for x_pos in range(crop_w):
            pixel_image_crop[y_pos, x_pos] = image[y_pos +
                                                   top - 1, x_pos + left - 1]
    return pixel_image_crop


def get_shape(image):
    """Get the dimensions of an image."""
    shape = image.shape
    height = shape[0]
    width = shape[1]
    return height, width


def get_pixel_image_and_coordinates(
        image, average_angle_offset, average_pixel_offset, average_line_distance):
    """Get the new image and the coordinates of the pixels in relation to the orginal image"""
    pixel_coordinates = []
    pixel_width = 200
    pixel_height = 200
    pixel_image = numpy.full((pixel_width, pixel_height, 3), [255, 255, 255])
    height, width = get_shape(image)
    cos = numpy.cos(average_angle_offset - numpy.pi/2)
    sin = numpy.sin(average_angle_offset - numpy.pi/2)
    pixel_offset_x = (
        average_pixel_offset[0] / average_line_distance) - pixel_width/2
    pixel_offset_y = (
        average_pixel_offset[1] / average_line_distance) - pixel_height/2
    # 0.5 as we want center of 'pixel' from original image
    for pixel_y in range(pixel_height):
        for pixel_x in range(pixel_width):
            pixel_x_unit = pixel_x + 0.5 + pixel_offset_x
            pixel_y_unit = pixel_y + 0.5 + pixel_offset_y
            # get unit cords
            x_unit = (pixel_x_unit * cos) - (pixel_y_unit * sin)
            y_unit = (pixel_x_unit * sin) + (pixel_y_unit * cos)
            # scale up
            x_scaled = int(average_line_distance * x_unit)
            y_scaled = int(average_line_distance * y_unit)
            if (x_scaled < width and x_scaled >= 0 and y_scaled < height and y_scaled >= 0):
                pixel_coordinates.append((x_scaled, y_scaled))
                pixel_image[pixel_y, pixel_x] = image[y_scaled, x_scaled]
    return pixel_image, pixel_coordinates


def get_average_pixel_offset(lines, average_line_distance):
    """Get the average x and y pixel offset."""
    offset_sum_x = 0
    offset_sum_y = 0
    for line in lines:
        if line[1] < numpy.pi:
            offset_sum_y += line[0] % average_line_distance
        else:
            offset_sum_x += line[0] % average_line_distance
    avg_offset_x = offset_sum_x / len(lines)
    avg_offset_y = offset_sum_y / len(lines)
    return (avg_offset_x, avg_offset_y)


def get_average_line_distance(lines):
    """Get an average distance between all the lines that are 1 'pixel' apart."""
    line_distances = get_line_distances(lines)
    sorted_line_distances = Counter(line_distances).most_common()

    def filter_lambda(line_distances):
        return line_distances[0] > 5 and line_distances[0] < 20 and line_distances[1] > len(lines)/2
    valid_lengths = list(filter(filter_lambda, sorted_line_distances))
    length_sum = 0
    count = 0
    for length in valid_lengths:
        length_sum += length[0] * length[1]
        count += length[1]
    average_line_distance = length_sum / count
    return average_line_distance


def get_line_distances(lines):
    """Get the distances between all the lines."""
    line_distances = []
    for line_1 in lines:
        for line_2 in lines:
            line_distances.append(abs(abs(line_1[0]) - abs(line_2[0])))
    return line_distances


def get_angle_offset(lines):
    """Get the average angle offset of the lines."""
    angle_sum = 0
    for line in lines:
        angle_sum += line[1] % (numpy.pi/2)
    avg_angle = angle_sum / len(lines)
    return avg_angle


def get_lines(edges):
    """Detect all lines in the image."""
    lines = cv2.HoughLines(edges, 1/2, numpy.pi/(180*2**6), 100)
    lines = lines.reshape(-1, 2).tolist()
    return lines


def print_bgr_image(image):
    """Print a BGR image."""
    temp_image = copy.deepcopy(image)
    temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
    print_image(temp_image)


def print_bgra_image(image):
    """Print a BGRA image."""
    temp_image = copy.deepcopy(image)
    split = cv2.split(temp_image)
    temp_image[:, :, 0] = split[2]
    temp_image[:, :, 2] = split[0]
    print_image(temp_image)


def print_image(image):
    """Print an image using pyplot."""
    pyplot.imshow(image)
    pyplot.show()


def draw_points_on_image(image, points):
    """Draw a set of points on an image."""
    for point in points:
        image[int(point[1]), int(point[0])] = [0, 0, 0]


def rotate_point(point, angle, origin=(0, 0)):
    """Rotate a point counterclockwise by a given angle in radians around a given origin."""
    o_x, o_y = origin
    p_x, p_y = point
    r_x = o_x + math.cos(angle) * (p_x - o_x) - math.sin(angle) * (p_y - o_y)
    r_y = o_y + math.sin(angle) * (p_x - o_x) + math.cos(angle) * (p_y - o_y)
    return r_x, r_y


def find_border(image, x_pos, y_pos, border_pixels, checked_pixels):
    """Recursively check neighbouring pixels to see if current pixel is a border pixel."""
    stack = [(image, x_pos, y_pos, border_pixels, checked_pixels)]
    while len(stack) > 0:
        arguments = stack.pop()
        image = arguments[0]
        x_pos = arguments[1]
        y_pos = arguments[2]
        border_pixels = arguments[3]
        checked_pixels = arguments[4]
        if (x_pos, y_pos) in checked_pixels:
            continue
        else:
            checked_pixels.add((x_pos, y_pos))
        height, width = get_shape(image)
        # up
        if y_pos > 0:
            if image[y_pos - 1, x_pos][3]:
                border_pixels.add((x_pos, y_pos))
            else:
                stack.append((image, x_pos, y_pos - 1,
                              border_pixels, checked_pixels))
                # up right
                if x_pos < width - 1:
                    if image[y_pos - 1, x_pos + 1][3]:
                        border_pixels.add((x_pos, y_pos))
                    else:
                        stack.append((image, x_pos + 1, y_pos - 1,
                                      border_pixels, checked_pixels))
        # right
        if x_pos < width - 1:
            if image[y_pos, x_pos + 1][3]:
                border_pixels.add((x_pos, y_pos))
            else:
                stack.append((image, x_pos + 1, y_pos,
                              border_pixels, checked_pixels))
                # right down
                if y_pos < height - 1:
                    if image[y_pos + 1, x_pos + 1][3]:
                        border_pixels.add((x_pos, y_pos))
                    else:
                        stack.append((image, x_pos + 1, y_pos + 1,
                                      border_pixels, checked_pixels))
        # down
        if y_pos < height - 1:
            if image[y_pos + 1, x_pos][3]:
                border_pixels.add((x_pos, y_pos))
            else:
                stack.append(
                    (image, x_pos, y_pos+1, border_pixels, checked_pixels))
                # down left
                if x_pos > 0:
                    if (image[y_pos + 1, x_pos - 1][3]):
                        border_pixels.add((x_pos, y_pos))
                    else:
                        stack.append((image, x_pos - 1, y_pos + 1,
                                      border_pixels, checked_pixels))
        # left
        if x_pos > 0:
            if image[y_pos, x_pos - 1][3]:
                border_pixels.add((x_pos, y_pos))
            else:
                stack.append((image, x_pos - 1, y_pos,
                              border_pixels, checked_pixels))
                if y_pos > 0:
                    if image[y_pos - 1, x_pos - 1][3]:
                        border_pixels.add((x_pos, y_pos))
                    else:
                        stack.append((image, x_pos - 1, y_pos - 1,
                                      border_pixels, checked_pixels))


def draw_lines(lines, image):
    """Draw lines on an image."""
    for line in lines:
        rho = line[0]
        theta = line[1]
        cos = numpy.cos(theta)
        sin = numpy.sin(theta)
        x_0 = cos*rho
        y_0 = sin*rho
        x_1 = int(x_0 + 1000*(-sin))
        y_1 = int(y_0 + 1000*(cos))
        x_2 = int(x_0 - 1000*(-sin))
        y_2 = int(y_0 - 1000*(cos))
        cv2.line(image, (x_1, y_1), (x_2, y_2), (0, 0, 0), 1)


if __name__ == "__main__":
    main()
