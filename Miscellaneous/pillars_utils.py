from skimage.filters import rank
from skimage.filters.thresholding import threshold_mean, try_all_threshold
from skimage.morphology import disk

from Miscellaneous.global_parameters import *
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np


# masks = []
# images = io.imread(path)
# for img in images:
#     plt.imshow(img)
#     masks.append(np.zeros_like(img))
#     # cv2.imshow(masks[i])
# cv2.imshow(masks)


def find_edges(path: str):
    images = io.imread(path)
    top_row = len(images[0])
    bottom_row = 0
    left_col = len(images[1])
    right_col = 0

    for img in images:
        threshold_level = 121
        mask = img >= threshold_level

        rows = np.where(np.any(mask == 1, axis=1))
        if top_row > rows[0][0]:
            top_row = rows[0][0]
        if bottom_row < rows[-1][-1]:
            bottom_row = rows[-1][-1]

        cols = np.where(np.any(mask == 1, axis=0))
        if left_col > cols[0][0]:
            left_col = cols[0][0]
        if right_col < cols[-1][-1]:
            right_col = cols[-1][-1]

    return top_row, bottom_row, left_col, right_col


def crop_image(path, top_row, bottom_row, left_col, right_col):
    images = io.imread(path)
    cropped_images = images[0:len(images), top_row:bottom_row + 1, left_col:right_col + 1]
    np.save('cropped images', cropped_images)


def filter_image(path_to_image):
    with open(path_to_image, 'rb') as f:
        images = np.load(f)
    kernel = np.ones((10, 10), np.uint8)
    # box_blur = cv2.filter2D(src=images[0], ddepth=-1, kernel=kernel)

    # fig, ax = try_all_threshold(box_blur, figsize=(10, 8), verbose=False)
    # plt.show()

    normal_result = rank.mean(images[0], footprint=kernel)
    plt.imshow(normal_result, cmap=plt.cm.gray)
    # plt.show()

    thresh = threshold_mean(normal_result)
    binary = normal_result > thresh
    fig, axes = plt.subplots(ncols=2, figsize=(8, 3))
    ax = axes.ravel()
    ax[0].imshow(normal_result, cmap=plt.cm.gray)
    ax[0].set_title('Original image')
    ax[1].imshow(binary, cmap=plt.cm.gray)
    ax[1].set_title('Result')
    for a in ax:
        a.axis('off')
    # plt.show()

    img = binary.astype(np.uint8)
    img *= 255
    # cv2.imshow('binary', img)
    # cv2.waitKey()

    img_erosion = cv2.erode(img, kernel)
    cv2.imshow('Input', img)
    cv2.imshow('Erosion', img_erosion)
    img_dilation = cv2.dilate(img_erosion, kernel)
    cv2.imshow('Dilation', img_dilation)
    cv2.waitKey(0)

    x=1


# path = PILLARS + '\\New-06-Airyscan Processing-04-actin_drift_corrected_13.2.tif'
# # Get edges
# x1, x2, x3, x4 = find_edges(path)
# crop_image(path, x1, x2, x3, x4)
filter_image('cropped images.npy')

