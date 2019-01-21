import cv2
import numpy as np
import decode
import color_analyser as ca
import lines

import logging
logger = logging.getLogger(__name__)

def __process_center(center, layer):
    rgb_int_center = np.uint(center[2] + center[1] * (2 ** 8) + center[0] * (2 ** 16))
    if rgb_int_center == 0:
        layer[layer != 0] = 1
        # now we have array of zeros and ones repreesnting inverted mask
        layer = (np.ones(layer.shape) - layer) * 255

    else:
        layer[layer != rgb_int_center] = 0
        layer[layer == rgb_int_center] = 255

    # morphology to remove ghosts
    # kernel = np.ones((3,3),np.uint8)
    # layer = layer.astype(np.float32)
    # gg = cv2.cvtColor(layer, cv2.COLOR_GRAY2RGB)

    # layer = cv2.morphologyEx(gg, cv2.MORPH_OPEN, kernel)
    return np.array(layer).astype(np.uint8)


def __extract_texts_per_layer(layers, net):
    # net should be reloaded for each image
    rgb_layers = (cv2.cvtColor(l, cv2.COLOR_GRAY2RGB) for l in layers)
    return (get_text(rgb, net) for rgb in rgb_layers)


def extract_image_features(img, config, net):
    """

    Does not use background layer
    :param img:
    :param config:
    :param net:
    :return:
    """
    k = ca.get_color_count(img)
    logger.info("number of distinct colors: {} ".format(k))

    layers = split_to_k_layers(img, k)

    config["hough_transform_minLineLength"] = int(min(img.shape[:2]) * .8)
    logger.info("extracting texts")
    texts = list(__extract_texts_per_layer(layers, net))
    logger.info("extracting lines")

    lines_arr = [lines.get_non_duplicit_lines(gs, config) for gs in layers]

    if len(lines_arr) < k:
        # adding empty list for background that was filtered out
        lines_arr = [] + lines_arr

    if len(layers) == len(texts) == len(lines_arr):
        return zip(layers, texts, lines_arr)
    else:
        msg = "Some data is missing: layers: {}, texts: {}, lines: {}"
        logger.error(msg)
        raise Exception(msg.format(len(layers), len(texts), len(lines_arr)))


def split_to_k_layers(img, k):
    array_image = np.float32(img.reshape((-1, 3)))
    shape = img.shape
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, centers = cv2.kmeans(array_image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[label.flatten()]
    res2 = res.reshape(shape).astype(np.uint)

    rgb_int_img = res2[:, :, 2] + np.left_shift(res2[:, :, 1], 8) + np.left_shift(res2[:, :, 0], 16)
    layers = [__process_center(center, rgb_int_img.copy()) for center in centers]
    # layers = map(lambda c: process_center(c,rgb_int_img.copy() ) ,centers)

    return layers  # , ret


def get_text(img, net):
    min_sh = min(img.shape[:2]) // 32
    (newW, newH) = min_sh * 32, min_sh * 32

    # sort the results bounding box coordinates from top to bottom
    return decode.detect_text(img, net, (newW, newH), min_confidence=.1)
