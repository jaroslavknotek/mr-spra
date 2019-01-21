import numpy as np
import matplotlib.pyplot as plt
import datetime
import cv2
import os


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_images(layers, output_dir):
    for i, l in enumerate(layers):
        mkdir(output_dir)
        img_path = os.path.join(output_dir, "l_{}.png".format(i))
        cv2.imwrite(img_path, l)


def get_timestamp():
    """
    return timestamp with millisecond rounded to 3 decimal places.
    :return: string
    """
    return datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')[:-3]


# This is not my code. It comes from here:
# https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def plot_hist(hist, peaks):
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.plot(peaks, hist[peaks], "x")
    plt.plot(hist, "r")


def plot_imgs(imgs, cmap_type=None):
    for i, img in enumerate(imgs):
        plt.subplot(len(imgs), 1, i + 1), plt.imshow(img, cmap=cmap_type)
        plt.title('Original'), plt.xticks([]), plt.yticks([])
