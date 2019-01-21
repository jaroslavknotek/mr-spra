import feature_extractor as extractor
import numpy as np
import cv2
import lines
import logging

logger = logging.getLogger(__name__)


def analyse_image(image, config, network):
    if is_photo(image):
        logger.info("Image is skipped. It's a photo")
        return {}

    logger.info("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(network)
    logger.info("analysing image")
    layers_data = extract_and_analyse(image, config, net)
    return __put_together(layers_data)


def is_photo(img):
    """
    Decides whether image can be info gfx. Its based on an assumption that
    ifno gfx have one major color - solid background
    :param img:
    :return:
    """
    # todo implement more reliable and robust method
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    background2image_ratio = np.max(hist) / gray.size
    return background2image_ratio < .5


def extract_and_analyse(image, config, net):
    layer_texts_lines = extractor.extract_image_features(image, config, net)
    return [__analyse_layer(lr, t, l) for lr, t, l in layer_texts_lines]


def __analyse_layer(layer, texts, lines_arr):
    logger.info("analysing layer")
    logger.info("getting keywords")
    keywords = set(np.array([t for pts, t in texts]).flatten())
    logger.info("analysing axes")
    axes_data = __analyse_axes(lines_arr, texts)
    is_background = __is_background_layer(layer)
    logger.info("analysing functions")
    function_data = __analyse_functions(lines_arr)
    return {
        "keywords": list(keywords),
        "axes_data": axes_data,
        "function_data": function_data,
        "layer": layer,
        "is_background": is_background
    }


def __analyse_functions(lines_arr):
    lines_arr = lines.filter_out_axes(lines_arr)
    # using simple float since float32 is not serializable
    return [{"id": i, "angle_rad": float(rad)} for i, (rho, rad) in enumerate(lines_arr)]


def __is_background_layer(layer):
    """
    If foreground color (white) cover majority of the layer
    :param layer:
        Binary image
    :return:
        true if image is presumed to be a background
    """
    return (np.sum(layer) / 255) / layer.size > .5


def __analyse_axes(lines_arr, texts):
    """
    Extract axes from lines if any exist.
    Search in texts for its label
    :param lines_arr:
    :param texts:
    :return:
    """
    axes = lines.get_axes_XY(lines_arr, 1)
    axes_data = {}
    if axes:
        x_axis, y_axis = axes
        # locate text nearby get text closes to axis
        # y axis

        y_axis_name = "n/a"
        y_axis_x_coor, _ = y_axis

        x_axis_name = "n/a"
        x_axis_y_coor, _ = x_axis
        if len(texts) > 0:
            y_axis_dists = [abs(min(startX, endX) - y_axis_x_coor) for ((startX, _, endX, _), text) in texts]
            y_axis_name = y_axis_dists[np.argmin(y_axis_dists)]

            x_axis_dists = [abs(min(startY, endY) - x_axis_y_coor) for ((_, startY, _, endY), text) in texts]
            x_axis_name = y_axis_dists[np.argmin(x_axis_dists)]

        # remove axis from lines
        axes_data["y"] = {"label": str(y_axis_name), "position": (int(y_axis_x_coor), 0)}
        axes_data["x"] = {"label": str(x_axis_name), "position": (0, int(x_axis_y_coor))}
    return axes_data


def __put_together(layers_data):
    image_data = {}
    keywords = set()
    functions = []
    for data in layers_data:

        if data.get("is_background", False):
            # skipping since it doesn't have reliable data
            continue

        axes_data = data.get("axes_data", {})
        if "x" in axes_data and "y" in axes_data and not "axes" in image_data:
            image_data["axes"] = axes_data

        for kw in data.get("keywords", []):
            keywords.add(kw)

        fn_data = data.get("function_data", [])

        for fn in fn_data:
            functions.append(fn)

    image_data["keywords"] = list(keywords)
    image_data["functions"] = functions
    return image_data
