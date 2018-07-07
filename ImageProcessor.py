import cv2
import math
import libs.morphsnakes as morphsnakes
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as sig
import scipy.ndimage as ndim
from skimage import feature, morphology


COLOR_RED = np.array([([170, 55, 20],
                       [179, 255, 255]),
                      ([0, 55, 20],
                       [10, 255, 255])])

COLOR_GREEN = np.array([([30, 55, 20], [90, 255, 255])])

COLOR_YELLOW = np.array([([30, 55, 20], [90, 255, 255])])

SEED_DIAMETER = 0.3

RESIZE_FACTOR = 4

def show(src, name, size):
    """Zobrazenie obrazku"""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, size[0], size[1])
    cv2.imshow(name, src)


def load_image(path):
    """Return image stored in path location or error that image cannot be loaded"""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    else:
        print('Cannot load image')
        raise SystemExit


def resize(src):
    return cv2.resize(src, (int(src.shape[0]/RESIZE_FACTOR), int(src.shape[0]/RESIZE_FACTOR)))


def crop_image(src, top_left, bottom_right):
    xmin, ymin = top_left
    xmax, ymax = bottom_right
    return src[xmin:xmax, ymin:ymax]


def adj_contrast(src):
    """Return src with adjusted contrast by value"""
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(2, 2))
    return clahe.apply(src)


def adj_brightness(src, value):
    """Return src with adjusted brightnes by value"""
    return cv2.add(src, value)


def get_channel(src, channel):
    """Return result contains only channel required by channel other set to 0"""
    channels = np.array(cv2.split(src))
    for chan in range(0, 3):
        if chan != channel:
            channels[chan] = np.zeros(channels[chan].shape)
    result = cv2.merge(channels)
    return result


def in_color_range(src, boundaries):
    """Return result which contains only color in given boundaries range
        arg boundaries take numpy array of tuples with lower,upper boundary respectively"""
    image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = np.zeros((src.shape[0], src.shape[1]), np.uint8)

    for color in boundaries:
        lower, upper = color
        tmp_mask = cv2.inRange(image, lower, upper)
        mask = mask | tmp_mask

    mask = alg_fill_holes(mask)
    out = cv2.bitwise_and(src, src, mask=mask)
    return out


def border_ero_dil(src):
    border = cv2.dilate(src, None, iterations=5)
    border = border - cv2.erode(src, None, iterations=1)
    return border


def border_sobel(src):
    grx = cv2.Sobel(src, cv2.CV_64F, 1, 0)
    grx = np.absolute(grx)
    grx = np.uint8(grx)

    gry = cv2.Sobel(src, cv2.CV_64F, 0, 1)
    gry = np.absolute(gry)
    gry = np.uint8(gry)

    return gry, grx, cv2.bitwise_or(gry, grx)


def apply_mask(src, mask):
    return cv2.bitwise_and(src, src, mask=mask)


def show_histogram(hist):
    plt.plot(hist)
    return True


def match_histogram(src, hist):
    src_hist, bins = np.histogram(src.flatten(), 256)

    cdf = hist.cumsum()
    cdf = 255 * cdf / cdf[-1]

    new_im = np.interp(src, bins[:-1], cdf)
    new_im = np.reshape(new_im, src.shape)
    new_im = np.uint8(new_im)
    return new_im


def apply_borders_to_mask(mask, borders):
    return rearrange(negation(implication(mask, borders)))


def negation(A):
    A = A.astype(bool)
    return np.uint8(~A)


def implication(A, B):
    A = A.astype(bool)
    B = B.astype(bool)
    return np.uint8(~A | B)


def normalize(src):
    maximum = src.max()
    normalized = src/maximum
    return normalized


def rearrange(src):
    maximum = np.max(src)
    tmp = 255 / maximum
    rearranged = src * tmp
    rearranged = np.uint8(rearranged)
    return rearranged


def gradient(src):
    sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=5)
    output = np.sqrt(np.square(sobelx) + np.square(sobely))
    gradien = rearrange(output)
    return output, gradien


def noise_blob_remove(src, min_size, mode_flag="BORDER"):
    removed = np.zeros(src.shape)
    min_cell_size = min_size
    _, cnts, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        if cv2.contourArea(c) > int(min_cell_size):
            if mode_flag == "BORDER":
                cv2.drawContours(removed, c, -1, color=255, thickness=1)
            if mode_flag == "FILLED":
                cv2.drawContours(removed, contours=[c], contourIdx=-1, color=255, thickness=cv2.FILLED)
    return removed


def alg_fill_holes(src):
    """Vyplni male diery v regione"""
    _, thresh = cv2.threshold(src, 125, 255, cv2.THRESH_BINARY)
    rows, cols = src.shape
    mask = thresh.copy()
    empty = np.zeros((rows + 2, cols + 2), np.uint8)
    for i in range(cols):
        if mask[0][i] == 0:
            cv2.floodFill(mask, empty, (i, 0), 255)

        if mask[rows - 1][i] == 0:
            cv2.floodFill(mask, empty, (i, rows-1), 255)

    for i in range(rows):
        if mask[i][0] == 0:
            cv2.floodFill(mask, empty, (0, i), 255)

        if mask[i][cols-1] == 0:
            cv2.floodFill(mask, empty, (cols - 1, i), 255)

    filled = src | cv2.bitwise_not(mask)

    # filled = negation(src)
    # cv2.imshow("f", rearrange(filled))
    # cv2.waitKey(0)
    # filled = noise_blob_remove(filled, min_size, mode_flag="FILLED")
    # cv2.imshow("f", filled)
    # cv2.waitKey(0)
    # filled = negation(filled)
    # filled = rearrange(filled)

    return filled


def alg_distance_transform(src, threshold=0.7, min_distance=20, mode_flag="THRESH", mask=None, footprint=None):
    dist_trans = ndim.distance_transform_edt(src)
    dist_trans = rearrange(cv2.normalize(dist_trans, 0.0, 1.0, cv2.NORM_MINMAX))

    # show(dist_trans, "dist", [400, 400])
    # cv2.imwrite('3_dist_trans.tif', dist_trans)

    # TODO if not thresh or peaks value error
    # seeds = src.copy()
    if mode_flag == "THRESH":
        _, seeds = cv2.threshold(dist_trans, threshold * dist_trans.max(), 255, 0)
    if mode_flag == "PEAKS":
        seeds = feature.peak_local_max(dist_trans, indices=False, min_distance=min_distance,
                                       footprint=footprint, labels=mask)

    return seeds, seeds.astype(bool)


def get_distance_moving_threshold(src, minimal_cell_diameter, tmin, tmax):
    seeds = np.zeros(src.shape)
    for i in range(tmin, tmax):
        dist, _ = alg_distance_transform(src, mode_flag="THRESH", threshold=i / 100)
        _, cnts, _ = cv2.findContours(dist, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            M = cv2.moments(c, False)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            if (SEED_DIAMETER * minimal_cell_diameter) < 24:
                cv2.circle(seeds, (cX, cY), int(24 / RESIZE_FACTOR), 255, -1, 8, 0) #24 default 24 / resize factor
            else:
                cv2.circle(seeds, (cX, cY), int((SEED_DIAMETER * minimal_cell_diameter) / RESIZE_FACTOR), 255, -1, 8, 0)  # 24 default 24 / resize factor
    return seeds


def alg_fast_radial_symmetry_transform(src, radii, alpha, std_dev, mode_flag='BOTH', draw=None, grad=False):
    """https://pdfs.semanticscholar.org/425c/0cf97af87c333f5033b7e4db86fd931fe092.pdf"""
    h, w = src.shape
    if grad:
        gradx = cv2.Sobel(src, cv2.CV_64FC1, 1, 0)
        grady = cv2.Sobel(src, cv2.CV_64FC1, 0, 1)
    else:
        gradx = np.float64(src)
        grady = np.float64(src)
    S = np.zeros((src.shape[0] + 2 * radii, src.shape[1] + 2 * radii))
    On = np.zeros(S.shape, np.float64)
    Mn = On.copy()

    for i in range(h):
        for j in range(w):

                n = math.sqrt(gradx[i][j] * gradx[i][j] + grady[i][j] * grady[i][j])

                if n > 0:
                    gp = np.zeros(2,)
                    gp[0] = int(round((gradx[i][j] / n) * radii))
                    gp[1] = int(round((grady[i][j] / n) * radii))

                    if mode_flag == "BRIGHT" or mode_flag == 'BOTH':
                        ppve = (int(i + gp[0] + radii), int(j + gp[1] + radii))

                        On[ppve[0]][ppve[1]] = On[ppve[0]][ppve[1]] + 1
                        Mn[ppve[0]][ppve[1]] = Mn[ppve[0]][ppve[1]] + n

                    if mode_flag == "DARK" or mode_flag == 'BOTH':
                        pnve = (int(i - gp[0] + radii), int(j - gp[1] + radii))

                        On[pnve[0]][pnve[1]] = On[pnve[0]][pnve[1]] - 1
                        Mn[pnve[0]][pnve[1]] = Mn[pnve[0]][pnve[1]] - n

    # absOn = np.absolute(On)
    On = np.absolute(On)

    # On_min_val, On_max_val, On_min_loc, On_max_loc = cv2.minMaxLoc(absOn)
    On_min_val, On_max_val, On_min_loc, On_max_loc = cv2.minMaxLoc(On)

    # absMn = np.absolute(Mn)
    Mn = np.absolute(Mn)

    # Mn_min_val, Mn_max_val, Mn_min_loc, Mn_max_loc = cv2.minMaxLoc(absMn)
    Mn_min_val, Mn_max_val, Mn_min_loc, Mn_max_loc = cv2.minMaxLoc(Mn)

    # print(f"{On_max_val} {Mn_max_val}", file=text_file)

    On = On / On_max_val
    Mn = Mn / Mn_max_val

    # O_n = On / On_max_val
    # M_n = Mn / Mn_max_val

    S = np.power(On, alpha)

    S = np.multiply(S, Mn)

    # Fn = np.multiply(np.power(np.absolute(O_n), alpha), M_n)

    kernel_size = math.ceil(radii / 2)

    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1

    # An = cv2.GaussianBlur(Fn, (kernel_size, kernel_size), radii * stdDev)

    S = cv2.GaussianBlur(S, (kernel_size, kernel_size), radii * std_dev)

    # S = np.multiply(Fn, An)

    _output = S[radii:h+radii, radii:w+radii]
    _output = cv2.normalize(_output, 0.0, 1.0, cv2.NORM_MINMAX)
    _output = rearrange(_output)

    _, mask = cv2.threshold(_output, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struct_elem, (-1, -1), iterations=1)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    seeds = np.zeros(src.shape)

    for c in contours:
        M = cv2.moments(c, False)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        seeds[cX][cY] = 1
        if draw:
            cv2.circle(draw, (cX, cY), 2, 255, -1, 8, 0)

    return seeds, seeds.astype(bool)


def alg_segmentation(src, n=24, w=29):
    rearranged = rearrange(src)
    histogram = cv2.calcHist([rearranged], [0], None, [256], [0, 256])
    norm_hist = normalize(histogram)

    dft_hist = cv2.dft(norm_hist)
    low_dft_hist = dft_hist.copy()
    for i in range(w + 1, len(dft_hist) - w - 1):
        low_dft_hist[i] = 0
    filtered_hist = abs(cv2.idft(low_dft_hist))

    slope_distribution = np.zeros(filtered_hist.shape)
    valleys = slope_distribution.copy()
    peaks = slope_distribution.copy()
    for val in range(0+n+1, len(filtered_hist)-n+1):
        YL = np.matrix(filtered_hist[val-n-1: val-1])
        YR = np.matrix(filtered_hist[val: val + n])
        BL = np.zeros((n, 2))
        BR = np.zeros((n, 2))
        for i, l, r in zip(range(0, n + 1), range(val - n, val), range(val + 1, val + 1 + n)):
            BL[i] = np.array([l, 1])
            BR[i] = np.array([r, 1])

        a, b = np.dot(np.dot(np.matrix(np.dot(np.transpose(BL), BL)).getI(), np.transpose(BL)), YL)
        left_slope = a

        XRT = np.transpose(BR)
        tmpr = np.dot(XRT, BR)
        tmpr = np.matrix(tmpr)
        tmpr = tmpr.getI()
        tmpr = np.dot(tmpr, XRT)
        tmpr = np.dot(tmpr, YR)
        right_slope = tmpr[0]

        slope_difference = right_slope - left_slope
        if slope_difference <= 0:
            valleys[val] = slope_difference
            peaks[val] = 0
        else:
            valleys[val] = 0
            peaks[val] = slope_difference
        slope_distribution[val] = slope_difference

    mod_image = match_histogram(src, peaks)
    # 5 Threshold options calculation
    # Vyhladenie signalu
    tmpslope = np.reshape(slope_distribution, (1, -1))[0]
    tmpslope = sig.savgol_filter(tmpslope, window_length=9, polyorder=3)
    loc_maxima = sig.find_peaks_cwt(tmpslope, np.arange(1, 20))
    # loc_minima = sig.find_peaks_cwt(-tmpslope, np.arange(1, 20))
    slope_distribution = np.reshape(tmpslope, slope_distribution.shape)

    maxis = np.array((0, 0))
    for i in loc_maxima:
        if slope_distribution[i] > slope_distribution[maxis[0]]:
            maxis[0] = i
        elif slope_distribution[i] < slope_distribution[maxis[0]] \
                and slope_distribution[i] > slope_distribution[maxis[1]]:
            maxis[1] = i

    maxis = np.sort(maxis)
    if maxis[0] != maxis[1]:
        tmp = np.argmin(slope_distribution[maxis[0]:maxis[1]]) + maxis[0]
        threshold = tmp
    else:
        threshold = 127

    _, binary_image = cv2.threshold(mod_image, threshold, 255, cv2.THRESH_BINARY)

    return mod_image, binary_image


def alg_snakes_acwe(src, levelset=None, smoothing=1, lambda1=1.0, lambda2=1.5, iterations=100, animate=False):
    if levelset is None:
        levelset = circles_levelset(src.shape, r=10)
    else:
        levelset = levelset
    snake = morphsnakes.MorphACWE(src, smoothing=smoothing, lambda1=lambda1, lambda2=lambda2)
    snake.levelset = levelset
    if animate:
        mask = morphsnakes.evolve_visual(snake, num_iters=iterations)
    else:
        snake.run(iterations)
        mask = snake.levelset
    mask = rearrange(mask)
    return mask


def alg_watershed(mask, minimal_cell_diameter, tmin, tmax):
    mask = rearrange(mask)
    seeds = get_distance_moving_threshold(mask, minimal_cell_diameter, tmin, tmax)

    seeds = seeds.astype(bool)

    markers, _ = ndim.label(seeds)

    # watershed
    labels = morphology.watershed(~seeds, markers, mask=mask)
    # count = draw_final_contours(src, labels)
    return labels


def draw_final_contours(src, lbls, mode_flag="RECT", min_size=600.0, out=None):
    count = 0
    for label in np.unique(lbls):
        if label == 0:
            continue
        mr = np.zeros((src.shape[0], src.shape[1]), dtype=np.uint8)
        mr[lbls == label] = 255

        cnts = cv2.findContours(mr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > int(min_size):
            if mode_flag == "RECT" or mode_flag == "RECT_CENTER":
                rect = cv2.boundingRect(c)
                x, y, w, h = rect
                cv2.rectangle(src, (x, y), (x+w, y+h), 255, 1)
            if mode_flag == "CENTER" or mode_flag == "RECT_CENTER":
                M = cv2.moments(c, False)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                cv2.circle(src, (cX, cY), int(24 / RESIZE_FACTOR), 0, -1, 8, 0)
                if out is not None:
                    cv2.circle(out, (cX*RESIZE_FACTOR, cY*RESIZE_FACTOR), int(24), 0, -1, 8, 0)
            if mode_flag == "CONTOUR":
                cv2.drawContours(src, c, -1, 0, thickness=4)
            count = count + 1

    return count


def square_levelset(shape):
    levelset = np.zeros(shape)
    x = shape[1]
    y = shape[0]
    dist_x = np.uint8(shape[1]*0.01)
    dist_y = np.uint8(shape[0] * 0.01)
    levelset = cv2.rectangle(levelset, (dist_x, dist_y), (x-dist_x, y-dist_y), 1, -1)
    return levelset


def circles_levelset(shape, r):
    levelset = np.zeros(shape)



    for i in range(r, shape[1], 2*r):
        for j in range(r, shape[0], 2*r):
            lelveset = cv2.circle(levelset, (i, j), r, 1, -1)
    return levelset


def nothing(x):
    pass


def analyze(source, grey, n=24, w=29, min_cell_diameter=60 ,s=1, l1=1.0, l2=1.5, iterations=1000, tmin=25, tmax=80,
            draw_type="CENTER"):
    # grey = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    # grey = cv2.resize(grey, (int(grey.shape[0] / RESIZE_FACTOR), int(grey.shape[1] / RESIZE_FACTOR)))

    after_seg, bin_after_seg = alg_segmentation(grey, n=n, w=w)
    snake_mask = alg_snakes_acwe(after_seg, circles_levelset(after_seg.shape, r=10), smoothing=s,
                                 lambda1=l1,
                                 lambda2=l2, iterations=iterations, animate=False)
    snake_mask = alg_fill_holes(snake_mask)

    min_cell_size = (min_cell_diameter / RESIZE_FACTOR) * (min_cell_diameter / RESIZE_FACTOR) * math.pi
    bin_after_seg = noise_blob_remove(bin_after_seg, min_cell_size / RESIZE_FACTOR)
    snake_mask = noise_blob_remove(snake_mask, min_cell_size, mode_flag="FILLED")

    final_mask = apply_borders_to_mask(snake_mask, bin_after_seg)

    labels = alg_watershed(final_mask, min_cell_diameter, tmin, tmax)
    count = draw_final_contours(source, labels, mode_flag=draw_type, min_size=min_cell_size / RESIZE_FACTOR)

    return source, grey, snake_mask, final_mask, bin_after_seg, after_seg, count


class ImageProcessor(object):

    def __init__(self):
        #Parameters
        self.resize_param = RESIZE_FACTOR
        self.param_n = 24
        self.param_w = 28
        self.min_cell_diameter = 100
        self.min_cell_size = 0
        self.smoothing = 1
        self.lambda1 = 1.0
        self.lambda2 = 1.2
        self.iterations = 100
        self.tmin = 25
        self.tmax = 80
        self.draw_type = "CENTER"
        self.red_boundary = COLOR_RED
        self.green_boundary = COLOR_GREEN
        self.yellow_boundary = COLOR_YELLOW

        #outputs
        self.origin = None
        self.source = None
        self.grey = None
        self.after_seg = None
        self.bin_snake = None
        self.bin_after_seg = None
        self.final_mask = None
        self.count_all = 0
        self.count_red = 0
        self.count_green = 0
        self.output = None
        self.full_output = None

    def load_origin(self, file_name):
        self.origin = load_image(file_name)
        self.full_output = self.origin.copy()
        self.source = resize(self.origin)
        self.output = self.source.copy()
        self.grey = self.source[:, :, 0]
        self.analyze()
        return self.origin

    def make_after_seg(self, n, w):
        self.after_seg, bin_after_seg = alg_segmentation(self.grey, n=self.param_n, w=self.param_w)

    def analyze(self):
        self.after_seg, bin_after_seg = alg_segmentation(self.grey, self.param_n, self.param_w)

        self.min_cell_size = (self.min_cell_diameter / self.resize_param) * \
                             (self.min_cell_diameter / self.resize_param) * math.pi

        self.bin_after_seg = noise_blob_remove(bin_after_seg, self.min_cell_size)
        snake_mask = alg_snakes_acwe(self.after_seg, circles_levelset(self.after_seg.shape, r=10),
                                         self.smoothing, self.lambda1, self.lambda2, self.iterations, animate=False)

        snake_mask = alg_fill_holes(snake_mask)
        self.bin_snake = noise_blob_remove(snake_mask, self.min_cell_size, mode_flag="FILLED")
        self.final_mask = apply_borders_to_mask(self.bin_snake, self.bin_after_seg)
        labels = alg_watershed(self.final_mask, self.min_cell_diameter, self.tmin, self.tmax)
        draw_final_contours(self.output, labels, mode_flag=self.draw_type,
                            min_size=self.min_cell_size / self.resize_param, out=self.full_output)

    def calibrate_color(self, boundary):
        return in_color_range(self.source, boundary)

    def calibrate_segmentation(self):
        self.after_seg, bin_after_seg = alg_segmentation(self.grey, self.param_n, self.param_w)

        self.min_cell_size = (self.min_cell_diameter / self.resize_param) * \
                             (self.min_cell_diameter / self.resize_param) * math.pi

        self.bin_after_seg = noise_blob_remove(bin_after_seg, self.min_cell_size)
        return rearrange(self.bin_after_seg.astype(int))

    def calibrate_snake(self):
        self.bin_snake = alg_snakes_acwe(self.after_seg, circles_levelset(self.after_seg.shape, r=10),
                                         self.smoothing, self.lambda1, self.lambda2, self.iterations, animate=False)
        return self.bin_snake

    #TODO make it work
    def set_final_mask(self):
        snake_mask = alg_fill_holes(self.bin_snake)
        self.bin_snake = noise_blob_remove(snake_mask, self.min_cell_size, mode_flag="FILLED")
        self.final_mask = apply_borders_to_mask(self.bin_snake, self.bin_after_seg)

    def calibrate_threshold(self):
        if self.bin_snake is not None:
            if self.final_mask is None:
                self.set_final_mask()
                labels = alg_watershed(self.final_mask, self.min_cell_diameter, self.tmin, self.tmax)
                draw_final_contours(self.output, labels, mode_flag=self.draw_type,
                                    min_size=self.min_cell_size / self.resize_param)
            else:
                labels = alg_watershed(self.final_mask, self.min_cell_diameter, self.tmin, self.tmax)
                draw_final_contours(self.output, labels, mode_flag=self.draw_type,
                                    min_size=self.min_cell_size / self.resize_param)

        return self.output


































# ###################################################MAIN###############################################################

# PARAM_N = 24
#
# PARAM_W = 28
#
# RESIZE_FACTOR = 4
#
# MINIMAL_CELL_DIAMETER = 500
#
# SEED_DIAMETER = 0.3
#
# source = load_image('input_data/2-9-2014_Cocomyxa3_ch01.tif')
# type = ["CENTER", "RECT", "RECT_CENTER"]
# for style in type:
#     grey, snake_mask, final_mask, bin_after_seg, after_seg, count = analyze(source, PARAM_N, PARAM_W,
#                                                                  min_cell_diameter=MINIMAL_CELL_DIAMETER,
#                                                                  s=4, l1=1, l2=1.4, iterations=500,
#                                                                  tmin=30, tmax=80, draw_type=style)
#     cv2.imwrite('3_vystup' + style + '.tif', grey)
#     cv2.imwrite('3_final_mask.tif', final_mask)
#     cv2.imwrite('3_snake_mask.tif', snake_mask)
#     cv2.imwrite('3_bin_after_seg.tif', bin_after_seg)
#     cv2.imwrite('3_after_seg.tif', after_seg)
#
#     # show(grey, 'grey_both', [512, 512])
#     # show(final_mask, 'final', [512, 512])
#     # show(snake_mask, 'snake', [512, 512])
#     # show(bin_after_seg, 'bin', [512, 512])
#     # print(count)
#     # cv2.waitKey(0)
#     #
#     # print(count)
#
# # grey, snake_mask, final_mask, bin_after_seg, after_seg, count = analyze(source, PARAM_N, PARAM_W,
# #                                                              min_cell_diameter=MINIMAL_CELL_DIAMETER,
# #                                                              s=4, l1=1, l2=1.4, iterations=500,
# #                                                              tmin=30, tmax=80, draw_type="CENTER")
#
#
# print(count)
#
# show(grey, 'grey_both', [512, 512])
# show(final_mask, 'final', [512, 512])
# show(snake_mask, 'snake', [512, 512])
# show(bin_after_seg, 'bin', [512, 512])
#
#
# cv2.waitKey(0)
