import numpy as np
import utils
import os
import cv2
import matplotlib.pyplot as plt
import warnings

PLOT = "PLOT"
IMG = "IMG"
__logging_type = None
__output_file_path = None


def start_logging_to_plots():
    global __logging_type
    __logging_type = PLOT


def start_logging_to_file(output_file_path: str):
    global __logging_type
    global __output_file_path
    __logging_type = IMG
    __output_file_path = output_file_path


# todo rewrite to class decorator
def log_img_results(img, img_name=None):
    if type(img) is not np.ndarray:
        # todo use __name__ or whatever
        warnings.warn("Error in log_img_result. Result of fn is not an image")
        return img

    if __logging_type == PLOT:
        __plot_image(img)
    elif __logging_type == IMG:
        __save_image(img, img_name)


def __plot_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def __save_image(img, img_name):
    if img_name is None:
        img_name = "debug"
    ts = utils.get_timestamp()
    file_name = "{}_{}.png".format(img_name, ts)
    path = os.path.join(__output_file_path, file_name)
    cv2.imwrite(path, img)


def decorator_apply_print_fn(fn, img_arg_index=0):
    def apply_debug(f):
        # This function is what we "replace" hello with
        def wrapper(*args, **kw):
            lns = f(*args, **kw)  # Call hello
            img = args[img_arg_index]
            # img = np.full(img.shape, 0)
            drawn = fn(lns, img.copy())
            log_img_results(drawn)
            return lns

        return wrapper

    return apply_debug


def decorator_log_img_result(img_name=None):
    """
    Decorator writing result of the function on the file system.
    Result is expected to be an image
    :param img_name:
    :param output_dir:
    :return:
    """
    utils.mkdir(__output_file_path)

    def debug_decorator(func):
        def func_wrapper(*args, **kwargs):
            img = func(*args, **kwargs)
            log_img_results(img, img_name)
            return img

        return func_wrapper

    return debug_decorator


def print_text(img, texts):
    for ((startX, startY, endX, endY), text) in texts:
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

        cv2.rectangle(img, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(img, text, (startX, startY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return img


def draw_lines(lines, img, color=(255, 0, 0)):
    for l in lines:
        rho, theta = l
        # print(rho,theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), color, 2)
    return img
