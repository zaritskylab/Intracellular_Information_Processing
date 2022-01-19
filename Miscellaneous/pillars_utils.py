from skimage.filters import rank
from skimage.filters.thresholding import threshold_mean, try_all_threshold
from skimage.morphology import disk
from collections import deque as queue

from Miscellaneous.global_parameters import *
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np
from datetime import datetime
import numpy.ma as ma
from collections import deque


# masks = []
# images = io.imread(path)
# for img in images:
#     plt.imshow(img)
#     masks.append(np.zeros_like(img))
#     # cv2.imshow(masks[i])
# cv2.imshow(masks)


# cv2.imshow("c", circle)
# cv2.waitKey(0)


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


def set_background_color(img, from_color, to_color):
    # Direction vectors
    dRow = [-1, 0, 1, 0]
    dCol = [0, 1, 0, -1]

    # Declare the visited array
    visited = [[False for i in range(img.shape[1])] for i in range(img.shape[0])]
    row = 0
    col = 0

    # Stores indices of the matrix cells
    q = queue()

    # Mark the starting cell as visited
    # and push it into the queue
    q.append((row, col))
    visited[row][col] = True

    # Iterate while the queue
    # is not empty
    while (len(q) > 0):
        cell = q.popleft()
        x = cell[0]
        y = cell[1]
        img[x][y] = to_color

        # Go to the adjacent cells
        for i in range(4):
            adjx = x + dRow[i]
            adjy = y + dCol[i]
            if (isValid(visited, adjx, adjy, img, from_color)):
                q.append((adjx, adjy))
                visited[adjx][adjy] = True


def isValid(visited, row, col, img, from_color):
    # If cell lies out of bounds
    if (row < 0 or col < 0 or row >= len(img) or col >= len(img[0])):
        return False

    # If cell is already visited
    if (visited[row][col]):
        return False

    # Otherwise
    return img[row][col] == from_color


def filter_image(path_to_image):
    print("start: ", datetime.now().time())
    with open(path_to_image, 'rb') as f:
        images = np.load(f)
    kernel = np.ones((10, 10), np.uint8) / 100
    # box_blur = cv2.filter2D(src=images[0], ddepth=-1, kernel=kernel)

    # fig, ax = try_all_threshold(box_blur, figsize=(10, 8), verbose=False)
    # plt.show()

    for i in range(len(images)):
        normal_result = rank.mean(images[180], footprint=kernel)
        plt.imshow(normal_result, cmap=plt.cm.gray)
        plt.show()

        thresh = threshold_mean(normal_result)
        binary = normal_result > thresh
        # fig, axes = plt.subplots(ncols=2, figsize=(8, 3))
        # ax = axes.ravel()
        # ax[0].imshow(normal_result, cmap=plt.cm.gray)
        # ax[0].set_title('Original image')
        # ax[1].imshow(binary, cmap=plt.cm.gray)
        # ax[1].set_title('Result')
        # for a in ax:
        #     a.axis('off')
        # plt.show()

        img = binary.astype(np.uint8)
        img *= 255

        set_background_color(img, 0, 255)

        img_erosion = cv2.erode(img, kernel)

        # cv2.imshow('Input', img)
        # TODO: here
        cv2.imshow('Erosion', img_erosion)
        mx = ma.masked_array(normal_result, img_erosion)
        plt.imshow(mx, cmap=plt.cm.gray)
        plt.show()
        # img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('Opening', img_opening)
        # img_dilation = cv2.dilate(img_erosion, kernel)
        # cv2.imshow('Dilation', img_dilation)
        cv2.waitKey(0)

        # countIslands(img_erosion.copy())

        # example_pillar_row = 334
        # example_pillar_col = 519
        #
        # starting_row_point = 30
        # starting_col_point = 117
        #
        # starting_row2_point = 70
        # starting_col2_point = 49
        #
        # row_jump = 76
        # double_col_jump = 134
        # pillar_row_length = 41
        # pillar_col_length = 36
        #
        # pillar_example = img_erosion[
        #                  example_pillar_row: example_pillar_row + pillar_row_length + 1,
        #                  example_pillar_col: example_pillar_col + pillar_col_length + 1
        #                  ]
        #
        # mask = np.zeros((1000, 1000), np.uint8)
        # mask += 255
        #
        # for row in range(starting_row_point, 1000, row_jump):
        #     for col in range(starting_col_point, 1000, double_col_jump):
        #         mask[
        #         row:row + pillar_row_length + 1,
        #         col:col + pillar_col_length + 1
        #         ] = pillar_example
        #
        # for row in range(starting_row2_point, 1000 - pillar_row_length, row_jump):
        #     for col in range(starting_col2_point, 1000 - pillar_col_length, double_col_jump):
        #         mask[
        #         row:row + pillar_row_length + 1,
        #         col:col + pillar_col_length + 1
        #         ] = pillar_example

        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)
        x = 1

    print("end: ", datetime.now().time())


def create_mask(radius: int, centers: list):
    mask = np.zeros((1000, 1000), np.uint8)
    mask += 255
    color = 0
    thickness = -1

    for center in centers:
        cv2.circle(mask, (center[1], center[0]), radius, color, thickness)

    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    return mask


def find_centers():
    start_row_1 = 49
    start_col_1 = 3

    row_jump = 77
    col_jump = 134

    centers = []

    for i in range(start_row_1, 1000, row_jump):
        for j in range(start_col_1, 1000, col_jump):
            centers.append((i, j))

    start_row_2 = 9
    start_col_2 = 70

    for i in range(start_row_2, 1000, row_jump):
        for j in range(start_col_2, 1000, col_jump):
            centers.append((i, j))

    return centers


# def create_pillar_mask(img, center, radius):
#     mask = np.zeros((1000, 1000), np.uint8)
#     mask += 255
#
#     top_left_corner = (center[0] - radius, center[1] - radius)
#     bottom_right_corner = (center[0] + radius, center[1] + radius)
#     zizi = img[top_left_corner[0]: bottom_right_corner[0], top_left_corner[1]: bottom_right_corner[1]]
#     mask[top_left_corner[0]: bottom_right_corner[0], top_left_corner[1]: bottom_right_corner[1]] = zizi
#     cv2.imshow("mask", mask)
#     cv2.waitKey(0)
#     return mask


# def create_small_circles(img_erosion):
#     example_pillar_row = 490
#     example_pillar_col = 664
#
#     starting_row_point = 30 + 22
#     starting_col_point = 117 + 17
#
#     starting_row2_point = 70 + 22
#     starting_col2_point = 49 + 17
#
#     row_jump = 76 + 22
#     double_col_jump = 134 + 17
#     pillar_row_length = 41 - 22
#     pillar_col_length = 36 - 17
#
#     pillar_example = img_erosion[
#                      example_pillar_row: example_pillar_row + pillar_row_length + 1,
#                      example_pillar_col: example_pillar_col + pillar_col_length + 1
#                      ]
#
#     mask = np.zeros((1000, 1000), np.uint8)
#     mask += 255
#
#     for row in range(starting_row_point, 1000, row_jump):
#         for col in range(starting_col_point, 1000, double_col_jump):
#             mask[
#             row:row + pillar_row_length + 1,
#             col:col + pillar_col_length + 1
#             ] = pillar_example
#
#     for row in range(starting_row2_point, 1000 - pillar_row_length, row_jump):
#         for col in range(starting_col2_point, 1000 - pillar_col_length, double_col_jump):
#             mask[
#             row:row + pillar_row_length + 1,
#             col:col + pillar_col_length + 1
#             ] = pillar_example
#
#     cv2.imshow('mask', mask)
#     cv2.waitKey(0)
#
#     circle = np.zeros((20, 20), np.uint8)
#     color = (255)
#     cv2.circle(circle, (10, 10), 10, color)
#     x = 1


# def isSafe(img, i, j, vis):
#     row = len(img)
#     col = len(img[0])
#     return ((i >= 0) and (i < row) and
#             (j >= 0) and (j < col) and
#             (img[i][j] == 0 and (not vis[i][j])))
#
#
# def BFS(img, vis, si, sj):
#     # These arrays are used to get row and
#     # column numbers of 8 neighbours of
#     # a given cell
#     size = 0
#     row = [-1, -1, -1, 0, 0, 1, 1, 1]
#     col = [-1, 0, 1, -1, 1, -1, 0, 1]
#
#     # Simple BFS first step, we enqueue
#     # source and mark it as visited
#     q = deque()
#     q.append([si, sj])
#     vis[si][sj] = True
#
#     # Next step of BFS. We take out
#     # items one by one from queue and
#     # enqueue their univisited adjacent
#     while (len(q) > 0):
#         temp = q.popleft()
#
#         i = temp[0]
#         j = temp[1]
#
#         # Go through all 8 adjacent
#         for k in range(8):
#             if (isSafe(img, i + row[k], j + col[k], vis)):
#                 vis[i + row[k]][j + col[k]] = True
#                 q.append([i + row[k], j + col[k]])
#                 size += 1
#     return size
#
#
# # This function returns number islands (connected
# # components) in a graph. It simply works as
# # BFS for disconnected graph and returns count
# # of BFS calls.
# def countIslands(img):
#     row = len(img)
#     col = len(img[0])
#     # Mark all cells as not visited
#     vis = [[False for i in range(row)]
#            for i in range(col)]
#     # memset(vis, 0, sizeof(vis));
#
#     # 5all BFS for every unvisited vertex
#     # Whenever we see an univisted vertex,
#     # we increment res (number of islands)
#     # also.
#     res = 0
#
#     for i in range(row):
#         for j in range(col):
#             if (img[i][j] == 0 and not vis[i][j]):
#                 island_size = BFS(img, vis, i, j)
#                 print(i, j, "island size ", island_size)
#                 res += 1
#
#     return res



# # Get edges
# x1, x2, x3, x4 = find_edges(path)
# crop_image(path, x1, x2, x3, x4)



def get_last_image(path_to_image):
    with open(path_to_image, 'rb') as f:
        images = np.load(f)
    kernel = np.ones((10, 10), np.uint8) / 100
    # box_blur = cv2.filter2D(src=images[0], ddepth=-1, kernel=kernel)

    # fig, ax = try_all_threshold(box_blur, figsize=(10, 8), verbose=False)
    # plt.show()
    return rank.mean(images[180], footprint=kernel)


def show_last_image_masked():
    last_img = get_last_image('../SavedData/not_cropped_images.npy')
    plt.imshow(last_img, cmap=plt.cm.gray)
    plt.show()

    with open('../SavedData/sub_mask.npy', 'rb') as f:
        pillars_mask = np.load(f)
        pillars_mask = 255 - pillars_mask
        mx = ma.masked_array(last_img, pillars_mask)
        plt.imshow(mx, cmap=plt.cm.gray)
        plt.show()
        x=1


def build_pillars_mask():
    centers = find_centers()
    small_mask = create_mask(10, centers)
    large_mask = create_mask(40, centers)
    pillars_mask = large_mask - small_mask
    pillars_mask *= 255

    # cv2.imshow('sub mask', sub_mask)
    # with open('../SavedData/sub_mask.npy', 'wb') as f:
    #     np.save(f, pillars_mask)
    # cv2.waitKey(0)
    return pillars_mask

def get_mask_for_each_pillar():
    centers = find_centers()
    thickness = -1
    pillar_to_mask_dict = {}
    for center in centers:
        small_mask_template = np.zeros((1000, 1000), np.uint8)
        cv2.circle(small_mask_template, (center[1], center[0]), 10, 255, thickness)

        large_mask_template = np.zeros((1000, 1000), np.uint8)
        cv2.circle(large_mask_template, (center[1], center[0]), 50, 255, thickness)

        mask = large_mask_template - small_mask_template
        pillar_to_mask_dict[center] = mask

    return pillar_to_mask_dict


def get_masked_frames_by_pillar(path_to_images):
    pillar2mask = get_mask_for_each_pillar()
    pillar2frames = {}
    pillar2frame_intensity = {}

    images = io.imread(path_to_images)

    for pillar_item in pillar2mask.items():
        pillar_id = pillar_item[0]
        pillar_mask = pillar_item[1]
        curr_pillar_intensity = []
        # curr_pillar_masked_frames = []
        for frame in images:
            frame_masked = np.where(pillar_mask, frame, 0)
            curr_pillar_intensity.append(np.sum(frame_masked))
            # curr_pillar_masked_frames.append(frame_masked)
        # pillar2frames[pillar_id] = curr_pillar_masked_frames
        pillar2frame_intensity[pillar_id] = curr_pillar_intensity

    return pillar2frame_intensity


path = PILLARS + '\\New-06-Airyscan Processing-04-actin_drift_corrected_13.2.tif'
full_mask = build_pillars_mask()
cv2.imshow("full_mask", full_mask)
cv2.waitKey(0)
last_img_mask = show_last_image_masked()

get_masked_frames_by_pillar(path)
