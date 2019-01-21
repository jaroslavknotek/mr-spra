import numpy as np
import cv2
from math import sqrt
import matplotlib.pyplot as plt
import utils
import img_logger


def filter_duplicit_lines(coors, thr_distance, thr_angle):
    done = []
    to_be_processed = list(coors)

    #    def dist(a, b):
    #        diff = a - b
    #        # first coordinates is from (0,np.pi)
    #        # the second is from (-imsize,size)
    #        # this is attempt to support
    #        weights = (1, 100)
    #        x, y = diff * weights
    #        return sqrt(x ** 2 + x ** 2)

    def dist(a, b):
        x, y = a - b
        return x, np.rad2deg(y)

    while len(to_be_processed) > 0:
        point = to_be_processed[0]
        distances = map(lambda x: (x, dist(x, point)), to_be_processed[1:])

        to_be_processed = [pt for pt, (d, a) in distances if d > thr_distance or a > thr_angle]
        done.append(point)
    return done


def __is_background(layer):
    # when there is too much white color
    # increase performance

    return np.sum(layer) / 255 > layer.size / 2


def get_lines(img, config):
    # edges = cv2.Canny(img,config["canny_min"],config["canny_max"])
    min_line_length = config["hough_transform_minLineLength"]
    max_line_gap = config["hough_transform_maxLineGap"]

    lines = cv2.HoughLines(img, 1, np.pi / 180, min_line_length, max_line_gap)
    if lines is None:
        return []

    return [l[0] for l in lines]


@img_logger.decorator_apply_print_fn(img_logger.draw_lines)
def get_non_duplicit_lines(img, config):
    """
    Finds major edges represented by lines
    :param img:
    :param config: dict
        expecting following values
            hough_transform_minLineLength: minimal length of line in pixels
            hough_transform_maxLineGap: max gap between two lines to be count as one
            hough_transform_filter_threshold: degrees of an angle between two lines which if is smaller,
                the lines are considered duplicit if they have the same distance from the [0,0]
    :return: array
        array of touples (distance from center, angle in radians)
    """
    if __is_background(img):
        return []

    lines = get_lines(img, config)

    threshold_distance = config["hough_transform_filter_threshold_angle"]
    threshold_angle = config["hough_transform_filter_threshold_distance"]
    return filter_duplicit_lines(lines, threshold_distance, threshold_angle)


def __filter_vertical_lines(lines, precision_deg):
    return [(r, t) for r, t in lines
            if t >= np.deg2rad(180 - precision_deg) or t <= np.deg2rad(precision_deg)]


def __filter_horizontal_lines(lines, precision_deg):
    return [(r, t) for r, t in lines
            if np.deg2rad(90 - precision_deg) <= t <= np.deg2rad(90 + precision_deg)]


def filter_out_axes(lines, precision_deg=2):
    """
    Filter out axes if they exists. Does not preserve order.
    :param lines: array of tuples
        array of pair rho,theta representing line. Rho is distance from center and theta is its angle in radians
    :param precision_deg:
          angle tolerance of degrees e.g. a line with 89 deg is still considered a horizontal if
        this parameter is more than 1
    :return: array
        lines without axes if any existed.
    """
    axes = get_axes_XY(lines, precision_deg)
    if not axes:
        return lines

    lines_set = set([(r, t) for r, t in lines])
    lines_set.remove(axes[0])
    lines_set.remove(axes[1])
    return lines


def get_axes_XY(lines, precision_deg=2):
    """
    Finds all vertical and horizontal lines and decides which one are axis if any
    :param lines: array
        array of touples (rho,theta) of distance and angle in radians representing lines.
    :param precision_deg:
        angle tolerance of degrees e.g. a line with 89 deg is still considered a horizontal if
        this parameter is more than 1
    :return: touple
        touple with x line and y line. Both represented by rho, theta pair.

    """

    vertical = __filter_vertical_lines(lines, precision_deg)
    horizontal = __filter_horizontal_lines(lines, precision_deg)

    # there are not both axes
    if len(horizontal) == 0 or len(vertical) == 0:
        return None

    # both axes present
    if len(horizontal) == 1 and len(vertical) == 1:
        return horizontal[0], vertical[0]

    # return topdown x axis
    top_down_index = np.argmax([abs(r) for r, _ in horizontal])
    top_down = horizontal[top_down_index]

    # return topleft y axis
    top_left_index = np.argmin([abs(r) for r, _ in vertical])
    top_left = vertical[top_left_index]

    return top_left, top_down
