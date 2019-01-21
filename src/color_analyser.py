import cv2
import numpy as np
import utils
from scipy.signal import find_peaks

import logging

logger = logging.getLogger(__name__)


def get_color_count(img):
    """
    Uses election alogirthms to compute the different colors on image
    :param img:
    :return:
    """
    counts = __color_election(img)
    logger.info("color election {}".format(counts))
    median = int(np.median(counts))
    logger.info("median {}".format(median))
    return median


def __color_election(img):
    """
    Performs different algorithms recognizing amount of color used and returns them in an array
    :param img:

    :return:
    """

    hist = __calculate_gs_histogram(img)

    peaks_only = get_color_count_peaks_only(hist)
    logger.info("{} found {} color".format("peaks", peaks_only))

    peaks_kmeans = get_color_count_peaks_kmeans(hist)
    logger.info("{} found {} color".format("peaks_kmeans", peaks_kmeans))

    peaks_marginal = get_color_count_peaks_marginal(hist)
    logger.info("{} found {} color".format("marginal", peaks_marginal))

    return np.array([peaks_only,peaks_kmeans,peaks_marginal])


    # previous version tried to work with altered images
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # img_s = cv2.filter2D(img.copy(), -1, kernel)
    # cols = __get_peaks(img_s)
    # sharpened_img_color_count = __extract_significant_color_count(cols)
    # logger.info("{} found {} color".format("sharpened", sharpened_img_color_count))
    #
    # blur = cv2.GaussianBlur(img.copy(), (3, 3), 0)
    # cols = __get_peaks(blur)
    # blurred_img_color_count = __extract_significant_color_count(cols)
    # logger.info("{} found {} color".format("blur", blurred_img_color_count))
    #
    # img_sb = cv2.filter2D(blur.copy(), -1, kernel)
    # cols = __get_peaks(img_sb)
    # sharp_blurred_img_color_count = __extract_significant_color_count(cols)
    # logger.info("{} found {} color".format("sharpblur", sharp_blurred_img_color_count))

    #return np.array(
    #    [orig_img_color_count, sharpened_img_color_count, blurred_img_color_count, sharp_blurred_img_color_count])


def get_color_count_peaks_only(histr,peaks_min_distance = 10):
    peaks,_ = _get_peaks(histr,peaks_min_distance)
    return len(peaks)


def get_color_count_peaks_kmeans(histr, peaks_min_distance=10):
    peaks, histogram = _get_peaks(histr,peaks_min_distance)

    k = 2
    # exlude background
    peak_values = np.ma.array(histogram[peaks], mask=False)
    background_id = np.argmax(peak_values)
    # masking value
    peak_values.mask[background_id] = True
    peak_values = peak_values.compressed()

    peak_values = np.array(peak_values).reshape((len(peak_values), 1)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    if peak_values.size < 2:
        return 0
    compactness, labels, centers = cv2.kmeans(peak_values, k, None, criteria, 10, flags)

    # center with important colors
    amax = int(np.argmax(centers))
    return labels[labels == amax].size + 1


def get_color_count_peaks_marginal(histr,peaks_min_distance=10):
    peaks, histogram = _get_peaks(histr, peaks_min_distance)

    if (len(peaks) < 2):
        return len(peaks)

    colors = sorted(histogram[peaks], reverse=True)
    # skipping background
    bck = colors[0]
    total = np.sum(histr) - bck
    thr = .85
    sum = 0

    for i, c in enumerate(colors[1:]):
        sum = sum + c
        if sum > total * thr:
            return i + 1

    return len(peaks)


def __calculate_gs_histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    histr = cv2.calcHist([gray], [0], None, [256], [0, 256])

    return np.float32(np.array([int(x[0]) for x in histr]))  # / img.size


def _get_peaks(histr, peaks_min_distance):
    histr_padded = np.zeros(histr.shape[0] + 2)
    histr_padded[1:-1] = histr

    histogram_normalized = histr_padded / np.sum(histr_padded)
    min_height = np.sqrt(np.sum(histr)) / np.sum(histr)
    peaks, _ = find_peaks(histogram_normalized, distance=peaks_min_distance, height=min_height)
    return peaks, histogram_normalized
