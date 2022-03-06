import math
import pickle
from pandocfilters import Math
from skimage.filters import rank
from skimage.filters.thresholding import threshold_mean, try_all_threshold
from skimage.morphology import disk
from collections import deque as queue, Counter
from Miscellaneous.global_parameters import *
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import numpy.ma as ma
from Miscellaneous.consts import *
from Miscellaneous.pillars_graph import *
import networkx as nx
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR

_image_path = LAST_IMG_VIDEO_06
_pillar_to_intensities_path = '../SavedPillarsData/SavedPillarsData_06/pillar_to_intensities_cached.pickle'
_frame2pillar_path = '../SavedPillarsData/SavedPillarsData_06/frames2pillars_cached.pickle'
_correlation_alive_normalized_path = '../SavedPillarsData/SavedPillarsData_06/alive_pillar_correlation_normalized_cached.pickle'
_correlation_alive_not_normalized_path = '../SavedPillarsData/SavedPillarsData_06/alive_pillar_correlation_cached.pickle'
_all_pillars_correlation_normalized_path = '../SavedPillarsData/SavedPillarsData_06/all_pillar_correlation_normalized_cached.pickle'
_all_pillars_correlation_not_normalized_path = '../SavedPillarsData/SavedPillarsData_06/all_pillar_correlation_cached.pickle'
_normalized = True


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
        image = np.load(f)
    kernel = np.ones((10, 10), np.uint8) / 100
    # box_blur = cv2.filter2D(src=image[0], ddepth=-1, kernel=kernel)

    # fig, ax = try_all_threshold(box_blur, figsize=(10, 8), verbose=False)
    # plt.show()

    for i in range(len(image)):
        normal_result = rank.mean(image, footprint=kernel)
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

    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    return mask


# def find_centers(start_row_1=START_ROW_1_06_VIDEO,
#                  start_col_1=START_COL_1_06_VIDEO,
#                  start_row_2=START_ROW_2_06_VIDEO,
#                  start_col_2=START_COL_2_06_VIDEO,
#                  row_jump=ROW_JUMP_06_VIDEO,
#                  col_jump=COL_JUMP_06_VIDEO):
#     centers = []
#
#     for i in range(start_row_1, 1000, row_jump):
#         for j in range(start_col_1, 1000, col_jump):
#             centers.append((i, j))
#
#     for i in range(start_row_2, 1000, row_jump):
#         for j in range(start_col_2, 1000, col_jump):
#             centers.append((i, j))
#
#     return centers


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


def isSafe(img, i, j, vis):
    row = len(img)
    col = len(img[0])
    return ((i >= 0) and (i < row) and
            (j >= 0) and (j < col) and
            (img[i][j] == 0 and (not vis[i][j])))


def BFS(img, vis, si, sj):
    # These arrays are used to get row and
    # column numbers of 8 neighbours of
    # a given cell
    if (img[si][sj] != 0):
        return []
    size = 0
    row = [-1, 0, 0, 1]
    col = [0, -1, 1, 0]

    locations = [(si, sj)]

    # Simple BFS first step, we enqueue
    # source and mark it as visited
    q = queue()
    q.append([si, sj])
    vis[si][sj] = True

    # Next step of BFS. We take out
    # items one by one from queue and
    # enqueue their univisited adjacent
    while (len(q) > 0):
        temp = q.popleft()

        i = temp[0]
        j = temp[1]

        # Go through all adjacents
        for k in range(len(row)):
            if (isSafe(img, i + row[k], j + col[k], vis)):
                vis[i + row[k]][j + col[k]] = True
                q.append([i + row[k], j + col[k]])
                locations.append((i + row[k], j + col[k]))
                size += 1
                if size >= CIRCLE_AREA * 1.5:
                    return []
    return locations


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

# Accepting tif path
def get_images(path):
    images = io.imread(path)
    return images


def get_last_image(path_to_image):
    with open(path_to_image, 'rb') as f:
        image = np.load(f)
    kernel = np.ones((10, 10), np.uint8) / 100
    # box_blur = cv2.filter2D(src=image[0], ddepth=-1, kernel=kernel)

    # fig, ax = try_all_threshold(box_blur, figsize=(10, 8), verbose=False)
    # plt.show()
    # return rank.mean(image, footprint=kernel)
    if len(image.shape) == 3:
        return image[-1]
    return image


def show_last_image_masked(mask_path=PATH_MASKS_VIDEO_06_15_35):
    last_img = get_last_image(_image_path)
    plt.imshow(last_img, cmap=plt.cm.gray)
    plt.show()

    with open(mask_path, 'rb') as f:
        pillars_mask = np.load(f)
        pillars_mask = 255 - pillars_mask
        mx = ma.masked_array(last_img, pillars_mask)
        plt.imshow(mx, cmap=plt.cm.gray)
        # add the centers location on the image
        # centers = find_centers()
        # for center in centers:
        #     s = '(' + str(center[0]) + ',' + str(center[1]) + ')'
        #     plt.text(center[VIDEO_06_LENGTH], center[0], s=s, fontsize=7, color='red')

        plt.show()


def build_pillars_mask(masks_path=PATH_MASKS_VIDEO_06_15_35,
                       logic_centers=True):
    if logic_centers:
        centers = find_centers_with_logic()
    # else:
    #     centers = find_centers()
    small_mask = create_mask(SMALL_MASK_RADIUS_06, centers)
    large_mask = create_mask(LARGE_MASK_RADIUS_06, centers)
    pillars_mask = large_mask - small_mask
    pillars_mask *= 255

    cv2.imshow('pillars_mask', pillars_mask)
    with open(masks_path, 'wb') as f:
        np.save(f, pillars_mask)
    cv2.waitKey(0)
    return pillars_mask


def find_centers_with_logic():
    last_img = get_last_image(_image_path)
    alive_centers = get_alive_centers(last_img)
    return generate_centers_from_live_centers(alive_centers, len(last_img))


def get_alive_centers(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    max_pixel = img.max()
    # find otsu's threshold value with OpenCV function
    ret, otsu = cv2.threshold(img, 0, max_pixel, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img[img < ret] = 0
    img[img > 0] = 1
    # plt.imshow(last_img, cmap=plt.cm.gray)
    # plt.show()
    alive_centers = set()
    e1 = int(len(img) / 5)
    e4 = e1 * 4
    visited = set()
    for row in range(e1, e4):
        col = e1
        while col < e4:
            if not (row, col) in visited and kinda_center(img, row, col):
                circle_area, center = get_center(img, row, col)
                if len(circle_area) == 0:
                    col += 1
                else:
                    visited.update(circle_area)
                    alive_centers.add(center)
                    col += CIRCLE_SEARCH_JUMP_SIZE
            else:
                col += 1
    return alive_centers


def generate_centers_from_live_centers(live_centers, matrix_size):
    return generate_centers_and_rules_from_live_centers(live_centers, matrix_size)[0]


def generate_centers_and_rules_from_live_centers(alive_centers, matrix_size):
    points = list(alive_centers)
    target = (matrix_size / 2, matrix_size / 2)
    closest_to_middle = min(points, key=lambda point: math.hypot(target[1] - point[1], target[0] - point[0]))
    points.remove(closest_to_middle)
    closest1 = min(points,
                   key=lambda point: math.hypot(closest_to_middle[1] - point[1], closest_to_middle[0] - point[0]))
    points.remove(closest1)
    closest2 = min(points,
                   key=lambda point: math.hypot(closest_to_middle[1] - point[1], closest_to_middle[0] - point[0]))

    rule1 = (closest_to_middle[0] - closest1[0], closest_to_middle[1] - closest1[1])
    rule2 = (closest_to_middle[0] - closest2[0], closest_to_middle[1] - closest2[1])

    generated_centers_in_line = {closest_to_middle}
    row = closest_to_middle[0]
    col = closest_to_middle[1]

    while -matrix_size <= row < matrix_size * 2 and -matrix_size <= col < matrix_size * 2:
        generated_centers_in_line.add((row, col))
        row += rule1[0]
        col += rule1[1]

    row = closest_to_middle[0]
    col = closest_to_middle[1]

    while -matrix_size <= row < matrix_size * 2 and col >= -matrix_size and col < matrix_size * 2:
        generated_centers_in_line.add((row, col))
        row -= rule1[0]
        col -= rule1[1]

    generated_centers = set(generated_centers_in_line)

    for center in generated_centers_in_line:
        row = center[0]
        col = center[1]
        while -matrix_size <= row < matrix_size * 2 and col >= -matrix_size and col < matrix_size * 2:
            generated_centers.add((row, col))
            row += rule2[0]
            col += rule2[1]
    for center in generated_centers_in_line:
        row = center[0]
        col = center[1]
        while -matrix_size <= row < matrix_size * 2 and col >= -matrix_size and col < matrix_size * 2:
            generated_centers.add((row, col))
            row -= rule2[0]
            col -= rule2[1]
    centers_in_range = [center for center in list(generated_centers) if
                        0 <= center[0] < matrix_size and 0 <= center[1] < matrix_size]
    return centers_in_range, rule1, rule2


def get_pillar_to_neighbors():
    last_img = get_last_image(_image_path)
    alive_centers = get_alive_centers(last_img)
    centers_lst, rule_jump_1, rule_jump_2 = generate_centers_and_rules_from_live_centers(alive_centers, len(last_img))
    pillar_to_neighbors = {}
    for p in centers_lst:
        neighbors_lst = set()

        n1 = (p[0] - rule_jump_1[0], p[1] - rule_jump_1[1])
        if n1 in centers_lst:
            neighbors_lst.add(n1)

        n2 = (p[0] + rule_jump_1[0], p[1] + rule_jump_1[1])
        if n2 in centers_lst:
            neighbors_lst.add(n2)

        n3 = (p[0] - rule_jump_2[0], p[1] - rule_jump_2[1])
        if n3 in centers_lst:
            neighbors_lst.add(n3)

        n4 = (p[0] + rule_jump_2[0], p[1] + rule_jump_2[1])
        if n4 in centers_lst:
            neighbors_lst.add(n4)

        n_minus1_minus2 = (n1[0] - rule_jump_2[0], n1[1] - rule_jump_2[1])
        if n_minus1_minus2 in centers_lst:
            neighbors_lst.add(n_minus1_minus2)

        n_minus1_plus2 = (n1[0] + rule_jump_2[0], n1[1] + rule_jump_2[1])
        if n_minus1_plus2 in centers_lst:
            neighbors_lst.add(n_minus1_plus2)

        n_plus1_minus2 = (n2[0] - rule_jump_2[0], n2[1] - rule_jump_2[1])
        if n_plus1_minus2 in centers_lst:
            neighbors_lst.add(n_plus1_minus2)

        n_plus1_plus2 = (n2[0] + rule_jump_2[0], n2[1] + rule_jump_2[1])
        if n_plus1_plus2 in centers_lst:
            neighbors_lst.add(n_plus1_plus2)

        pillar_to_neighbors[p] = list(neighbors_lst)

    return pillar_to_neighbors


def kinda_center(img, row, col):
    col_zeros = img[
                row:row + 1,
                col - CIRCLE_ZERO_VALIDATE_SEARCH_LENGTH: col + CIRCLE_ZERO_VALIDATE_SEARCH_LENGTH]
    if np.any(col_zeros):
        return False
    row_zeros = img[
                row - CIRCLE_ZERO_VALIDATE_SEARCH_LENGTH: row + CIRCLE_ZERO_VALIDATE_SEARCH_LENGTH,
                col: col + 1]
    if np.any(row_zeros):
        return False

    return img[row][col + CIRCLE_ONE_VALIDATE_SEARCH_LENGTH] == 1 \
           and img[row][col - CIRCLE_ONE_VALIDATE_SEARCH_LENGTH] == 1 \
           and img[row - CIRCLE_ONE_VALIDATE_SEARCH_LENGTH][col] == 1 \
           and img[row + CIRCLE_ONE_VALIDATE_SEARCH_LENGTH][col] == 1


def get_center(img, row, col):
    rows = len(img)
    cols = len(img[0])
    # Mark all cells as not visited
    vis = [[False for i in range(rows)]
           for i in range(cols)]
    circle_area = BFS(img, vis, row, col)
    if len(circle_area) == 0:
        return [], (0, 0)
    return circle_area, get_circle_center(circle_area)


def get_circle_center(circle_area):
    X = [tup[0] for tup in circle_area]
    Y = [tup[1] for tup in circle_area]
    avg_X = sum(X) / len(X)
    avg_Y = sum(Y) / len(Y)

    return (int(avg_X), int(avg_Y))


def get_mask_for_each_pillar():
    centers = find_centers_with_logic()
    thickness = -1
    pillar_to_mask_dict = {}
    for center in centers:
        small_mask_template = np.zeros((1000, 1000), np.uint8)
        cv2.circle(small_mask_template, (center[1], center[0]), SMALL_MASK_RADIUS_06, 255, thickness)

        large_mask_template = np.zeros((1000, 1000), np.uint8)
        cv2.circle(large_mask_template, (center[1], center[0]), LARGE_MASK_RADIUS_06, 255, thickness)

        mask = large_mask_template - small_mask_template

        pillar_to_mask_dict[center] = mask

    return pillar_to_mask_dict


def get_pillar_to_intensities(path):
    if os.path.isfile(_pillar_to_intensities_path):
        with open(_pillar_to_intensities_path, 'rb') as handle:
            pillar2frame_intensity = pickle.load(handle)
            return pillar2frame_intensity

    pillar2mask = get_mask_for_each_pillar()
    pillar2frames = {}
    pillar2frame_intensity = {}

    images = get_images(path)

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

    with open(_pillar_to_intensities_path, 'wb') as handle:
        pickle.dump(pillar2frame_intensity, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pillar2frame_intensity


# def get_pillar_to_neighbors():
#     pillar_to_neighbors = {}
#     all_pillars = find_centers()
#
#     # "direct" neighbors
#     for pillar in all_pillars:
#         pillar_to_neighbors[pillar] = []
#
#         if (pillar[0] + ROW_JUMP_06_VIDEO, pillar[1]) in all_pillars:
#             pillar_to_neighbors[pillar].append((pillar[0] + ROW_JUMP_06_VIDEO, pillar[1]))
#
#         if (pillar[0] - ROW_JUMP_06_VIDEO, pillar[1]) in all_pillars:
#             pillar_to_neighbors[pillar].append((pillar[0] - ROW_JUMP_06_VIDEO, pillar[1]))
#
#         if (pillar[0], pillar[1] + COL_JUMP_06_VIDEO) in all_pillars:
#             pillar_to_neighbors[pillar].append((pillar[0], pillar[1] + COL_JUMP_06_VIDEO))
#
#         if (pillar[0], pillar[1] - COL_JUMP_06_VIDEO) in all_pillars:
#             pillar_to_neighbors[pillar].append((pillar[0], pillar[1] - COL_JUMP_06_VIDEO))
#
#     pillar_to_cross_neighbors = {}
#     # cross neighbors
#     for pillar in all_pillars:
#         pillar_to_cross_neighbors[pillar] = []
#
#         if (pillar[0] + JUMP_ROW_CROSS_1_06, pillar[1] + JUMP_COL_CROSS_06_VIDEO) in all_pillars:
#             pillar_to_cross_neighbors[pillar].append(
#                 (pillar[0] + JUMP_ROW_CROSS_1_06, pillar[1] + JUMP_COL_CROSS_06_VIDEO))
#
#         if (pillar[0] - JUMP_ROW_CROSS_1_06, pillar[1] + JUMP_COL_CROSS_06_VIDEO) in all_pillars:
#             pillar_to_cross_neighbors[pillar].append(
#                 (pillar[0] - JUMP_ROW_CROSS_1_06, pillar[1] + JUMP_COL_CROSS_06_VIDEO))
#
#         if (pillar[0] - JUMP_ROW_CROSS_1_06, pillar[1] - JUMP_COL_CROSS_06_VIDEO) in all_pillars:
#             pillar_to_cross_neighbors[pillar].append(
#                 (pillar[0] - JUMP_ROW_CROSS_1_06, pillar[1] - JUMP_COL_CROSS_06_VIDEO))
#
#         if (pillar[0] + JUMP_ROW_CROSS_1_06, pillar[1] - JUMP_COL_CROSS_06_VIDEO) in all_pillars:
#             pillar_to_cross_neighbors[pillar].append(
#                 (pillar[0] + JUMP_ROW_CROSS_1_06, pillar[1] - JUMP_COL_CROSS_06_VIDEO))
#
#         # different number of row jump
#         if (pillar[0] + JUMP_ROW_CROSS_2_06, pillar[1] + JUMP_COL_CROSS_06_VIDEO) in all_pillars:
#             pillar_to_cross_neighbors[pillar].append(
#                 (pillar[0] + JUMP_ROW_CROSS_2_06, pillar[1] + JUMP_COL_CROSS_06_VIDEO))
#
#         if (pillar[0] - JUMP_ROW_CROSS_2_06, pillar[1] + JUMP_COL_CROSS_06_VIDEO) in all_pillars:
#             pillar_to_cross_neighbors[pillar].append(
#                 (pillar[0] - JUMP_ROW_CROSS_2_06, pillar[1] + JUMP_COL_CROSS_06_VIDEO))
#
#         if (pillar[0] - JUMP_ROW_CROSS_2_06, pillar[1] - JUMP_COL_CROSS_06_VIDEO) in all_pillars:
#             pillar_to_cross_neighbors[pillar].append(
#                 (pillar[0] - JUMP_ROW_CROSS_2_06, pillar[1] - JUMP_COL_CROSS_06_VIDEO))
#
#         if (pillar[0] + JUMP_ROW_CROSS_2_06, pillar[1] - JUMP_COL_CROSS_06_VIDEO) in all_pillars:
#             pillar_to_cross_neighbors[pillar].append(
#                 (pillar[0] + JUMP_ROW_CROSS_2_06, pillar[1] - JUMP_COL_CROSS_06_VIDEO))
#
#     return pillar_to_neighbors, pillar_to_cross_neighbors


def get_frame_to_graph():
    path = get_images_path()
    pillar_to_neighbors, pillar_to_cross_neighbors = get_pillar_to_neighbors()
    pillar_frame_intensity_dict = get_pillar_to_intensities(path)
    images = get_images(path)
    frame_to_graph_dict = {}
    for i in range(len(images)):
        pillars_graph = PillarsGraph()
        # fill the graph with pillar nodes
        for pillar in pillar_frame_intensity_dict.items():
            pillar_id = pillar[0]
            pillar_intensity = pillar[1][i]
            pillar_neighbors = pillar_to_neighbors[pillar_id]
            pillar_node = PillarNode(pillar_id, pillar_intensity, i)
            pillars_graph.add_pillar_node(pillar_id, pillar_node)
        # fill each node with his neighbors nodes
        for pillar_item in pillars_graph.pillar_id_to_node.items():
            pillar_id = pillar_item[0]
            pillar_node = pillar_item[1]
            pillar_to_node = pillars_graph.pillar_id_to_node
            for neighbor in pillar_to_neighbors[pillar_id]:
                pillar_node.add_neighbor(pillar_to_node[neighbor])
            # TODO: also on cross neighbors?

        frame_to_graph_dict[i] = pillars_graph

    return frame_to_graph_dict


def get_images_path():
    return PILLARS + VIDEO_06_TIF_PATH


def intensity_histogram():
    # histogram to find intensity threshold
    all_intensities = []
    pillar2frame_intensity_dict = get_pillar_to_intensities(get_images_path())
    for val in pillar2frame_intensity_dict.values():
        all_intensities.extend(val)
    all_intensities = np.asarray(all_intensities)
    intensity_to_count = [(item, count) for item, count in Counter(all_intensities).items() if count > 1]
    n = all_intensities.size
    rrange = all_intensities.max() - all_intensities.min()
    num_of_intervals = math.sqrt(n)
    width_of_intervals = rrange / num_of_intervals
    # bins = [i for i in range(all_intensities.min(), all_intensities.max() + 1, int(width_of_intervals))]
    bins = [i for i in range(all_intensities.min(), all_intensities.max() + 1, 500000)]
    plt.hist(all_intensities, bins=bins, density=True)
    plt.show()
    x = 1


# full_mask = build_pillars_mask()
# # cv2.imshow("full_mask", full_mask)
# # cv2.waitKey(0)
# last_img_mask = show_last_image_masked()
#
# get_frame_to_graph()
# intensity_histogram()

# plot pillars time series
# pillar2frame_intensity_dict = get_pillar_to_intensities(get_images_path())
# pillars_items = list(pillar2frame_intensity_dict.items())
# for pillar_item in pillars_items:
#     intensities = pillar_item[1]
#     for i in range(len(intensities)):
#         if intensities[i] < INTENSITY_THRESHOLD:
#             intensities[i] = 0
#
#     x = [i + 1 for i in range(len(intensities))]
#     plt.plot(x, intensities)
#     plt.xlabel('Time')
#     plt.ylabel('Intensity')
#     plt.title('Pillar ' + str(pillar_item[0]))
#     plt.show()
#     x = 1


def get_frame_to_alive_pillars():
    if os.path.isfile(_frame2pillar_path):
        with open(_frame2pillar_path, 'rb') as handle:
            frame_to_alive_pillars = pickle.load(handle)
            return frame_to_alive_pillars
    frame_to_alive_pillars = {}
    # frame_to_background_pillars = {}
    images = get_images(get_images_path())
    pillar2mask = get_mask_for_each_pillar()
    frame_num = 1
    for frame in images:
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        max_pixel = blur.max()
        # find otsu's threshold value with OpenCV function
        ret, otsu = cv2.threshold(blur, 0, max_pixel, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur[blur < ret] = 0

        relevant_pillars_in_frame = []
        # background_pillars = []
        for pillar_item in pillar2mask.items():
            curr_pillar = blur * pillar_item[1]
            is_pillar_alive = np.sum(curr_pillar)
            if is_pillar_alive > 0:
                relevant_pillars_in_frame.append(pillar_item[0])
            # else:
            #     background_pillars.append(pillar_item[0])
        frame_to_alive_pillars[frame_num] = relevant_pillars_in_frame
        # frame_to_background_pillars[frame_num] = background_pillars
        frame_num += 1
    # with open('../SavedPillarsData_05/background_gray_scale_pillars.pickle', 'wb') as handle:
    #     pickle.dump(frame_to_background_pillars, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(_frame2pillar_path, 'wb') as handle:
        pickle.dump(frame_to_alive_pillars, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return frame_to_alive_pillars


def get_alive_pillars_correlation():
    path = get_alive_pillars_corr_path()

    if os.path.isfile(path):
        with open(path, 'rb') as handle:
            correlation = pickle.load(handle)
            return correlation

    relevant_pillars_dict = get_alive_pillars_to_intensities()

    pillar_intensity_df = pd.DataFrame({str(k): v for k, v in relevant_pillars_dict.items()})
    alive_pillars_corr = pillar_intensity_df.corr()

    with open(path, 'wb') as handle:
        pickle.dump(alive_pillars_corr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return alive_pillars_corr


def get_alive_pillars_corr_path():
    if _normalized:
        path = _correlation_alive_normalized_path
    else:
        path = _correlation_alive_not_normalized_path

    return path


def get_alive_pillars_lst():
    frame_to_pillars = get_frame_to_alive_pillars()
    any_time_live_pillars = set()
    for pillars in frame_to_pillars.values():
        any_time_live_pillars.update(pillars)

    return list(any_time_live_pillars)


def get_alive_pillars_to_intensities():
    if _normalized:
        pillar_intensity_dict = normalized_intensities_by_mean_background_intensity()
    else:
        pillar_intensity_dict = get_pillar_to_intensities(get_images_path())

    alive_pillars = get_alive_pillars_lst()

    alive_pillars_dict = {pillar: pillar_intensity_dict[pillar] for pillar in alive_pillars}

    return alive_pillars_dict


def get_all_pillars_correlation():
    path = get_all_pillars_corr_path()

    if os.path.isfile(path):
        with open(path, 'rb') as handle:
            correlation = pickle.load(handle)
            return correlation

    if _normalized:
        pillar_intensity_dict = normalized_intensities_by_mean_background_intensity()
    else:
        pillar_intensity_dict = get_pillar_to_intensities(get_images_path())

    all_pillar_intensity_df = pd.DataFrame({str(k): v for k, v in pillar_intensity_dict.items()})
    all_pillars_corr = all_pillar_intensity_df.corr()

    with open(path, 'wb') as handle:
        pickle.dump(all_pillars_corr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_pillars_corr


def get_all_pillars_corr_path():
    if _normalized:
        path = _all_pillars_correlation_normalized_path
    else:
        path = _all_pillars_correlation_not_normalized_path

    return path


# def correlation_plot(only_alive=True):
#     my_G = nx.Graph()
#     nodes_loc = find_centers()
#     neighbors1, neighbors2 = get_pillar_to_neighbors()
#     node_loc2index = {}
#     for i in range(len(nodes_loc)):
#         node_loc2index[nodes_loc[i]] = i
#         my_G.add_node(i)
#     alive_pillars_correlation = get_alive_pillars_correlation(normalized=True)
#     all_pillars_corr = get_all_pillars_correlation(normalized=True)
#
#     if only_alive:
#         correlation = alive_pillars_correlation
#     else:
#         correlation = all_pillars_corr
#
#     for n1 in neighbors1.keys():
#         for n2 in neighbors1[n1]:
#             my_G.add_edge(node_loc2index[n1], node_loc2index[n2])
#             try:
#                 my_G[node_loc2index[n1]][node_loc2index[n2]]['weight'] = correlation[str(n1)][str(n2)]
#             except:
#                 x = 1
#     for n1 in neighbors2.keys():
#         for n2 in neighbors2[n1]:
#             my_G.add_edge(node_loc2index[n1], node_loc2index[n2])
#             try:
#                 my_G[node_loc2index[n1]][node_loc2index[n2]]['weight'] = correlation[str(n1)][str(n2)]
#
#             except:
#                 x = 1
#
#     edges, weights = zip(*nx.get_edge_attributes(my_G, 'weight').items())
#     cmap = plt.cm.seismic
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
#     frame2pillars = get_frame_to_alive_pillars()
#     nodes_index2size = [10] * len(nodes_loc)
#     for node in nodes_loc:
#         for i in range(len(frame2pillars)):
#             if node in frame2pillars[i + 1]:
#                 nodes_index2size[node_loc2index[node]] = len(frame2pillars) - ((i // 13) * 13)
#                 break
#     nodes_loc_y_inverse = [(loc[1], 1000 - loc[0]) for loc in nodes_loc]
#
#     nx.draw(my_G, nodes_loc_y_inverse, with_labels=True, node_color='gray', edgelist=edges, edge_color=weights,
#             width=3.0,
#             edge_cmap=cmap,
#             node_size=nodes_index2size)
#     plt.colorbar(sm)
#     plt.show()
#     x = 1

def correlation_plot(only_alive=True):
    my_G = nx.Graph()
    last_img = get_last_image(_image_path)
    alive_centers = get_alive_centers(last_img)
    nodes_loc = generate_centers_from_live_centers(alive_centers, len(last_img))
    neighbors = get_pillar_to_neighbors()
    node_loc2index = {}
    for i in range(len(nodes_loc)):
        node_loc2index[nodes_loc[i]] = i
        my_G.add_node(i)
    alive_pillars_correlation = get_alive_pillars_correlation()
    all_pillars_corr = get_all_pillars_correlation()

    if only_alive:
        correlation = alive_pillars_correlation
    else:
        correlation = all_pillars_corr

    for n1 in neighbors.keys():
        for n2 in neighbors[n1]:
            my_G.add_edge(node_loc2index[n1], node_loc2index[n2])
            try:
                my_G[node_loc2index[n1]][node_loc2index[n2]]['weight'] = correlation[str(n1)][str(n2)]
            except:
                x = 1

    edges, weights = zip(*nx.get_edge_attributes(my_G, 'weight').items())
    cmap = plt.cm.seismic
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    frame2pillars = get_frame_to_alive_pillars()
    nodes_index2size = [10] * len(nodes_loc)
    for node in nodes_loc:
        for i in range(len(frame2pillars)):
            if node in frame2pillars[i + 1]:
                nodes_index2size[node_loc2index[node]] = len(frame2pillars) - ((i // 13) * 13)
                break
    nodes_loc_y_inverse = [(loc[1], 1000 - loc[0]) for loc in nodes_loc]

    nx.draw(my_G, nodes_loc_y_inverse, with_labels=True, node_color='gray', edgelist=edges, edge_color=weights,
            width=3.0,
            edge_cmap=cmap,
            node_size=nodes_index2size)
    plt.colorbar(sm)
    plt.show()
    x = 1


def indirect_alive_neighbors_correlation_plot(pillar_location, only_alive=True):
    my_G = nx.Graph()
    nodes_loc = find_centers_with_logic()
    node_loc2index = {}
    for i in range(len(nodes_loc)):
        node_loc2index[nodes_loc[i]] = i
        my_G.add_node(i)

    if only_alive:
        pillars = get_alive_pillars_to_intensities()
    else:
        # pillars = get_pillar_to_intensities(get_images_path())
        pillars = normalized_intensities_by_mean_background_intensity()

    pillar_loc = pillar_location
    indirect_neighbors_dict = get_pillar_indirect_neighbors_dict(pillar_location)
    # alive_pillars = get_alive_pillars_to_intensities()
    directed_neighbors = get_pillar_directed_neighbors(pillar_loc)
    indirect_alive_neighbors = {pillar: indirect_neighbors_dict[pillar] for pillar in pillars.keys() if
                                pillar not in directed_neighbors}
    pillars_corr = get_indirect_neighbors_correlation(pillar_loc, only_alive)
    for no_n1 in indirect_alive_neighbors.keys():
        my_G.add_edge(node_loc2index[pillar_loc], node_loc2index[no_n1])
        try:
            my_G[node_loc2index[pillar_loc]][node_loc2index[no_n1]]['weight'] = pillars_corr[str(pillar_loc)][
                str(no_n1)]
        except:
            x = -1

    edges, weights = zip(*nx.get_edge_attributes(my_G, 'weight').items())

    cmap = plt.cm.seismic
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    frame2pillars = get_frame_to_alive_pillars()
    nodes_index2size = [10] * len(nodes_loc)
    for node in nodes_loc:
        for i in range(len(frame2pillars)):
            if node in frame2pillars[i + 1]:
                nodes_index2size[node_loc2index[node]] = len(frame2pillars) - ((i // 13) * 13)
                break
    nodes_loc_y_inverse = [(loc[1], 1000 - loc[0]) for loc in nodes_loc]
    nx.draw(my_G, nodes_loc_y_inverse, with_labels=True, node_color='gray', edgelist=edges, edge_color=weights,
            width=3.0,
            edge_cmap=cmap,
            node_size=nodes_index2size)
    plt.colorbar(sm)
    plt.show()
    x = 1


def gc_plot(gc_df, only_alive=True):
    my_G = nx.Graph().to_directed()
    nodes_loc = find_centers_with_logic()
    # neighbors1, neighbors2 = get_pillar_to_neighbors()
    node_loc2index = {}
    for i in range(len(nodes_loc)):
        node_loc2index[str(nodes_loc[i])] = i
        my_G.add_node(i)
    alive_pillars_correlation = get_alive_pillars_correlation()
    all_pillars_corr = get_all_pillars_correlation()

    if only_alive:
        correlation = alive_pillars_correlation
    else:
        correlation = all_pillars_corr

    for col in gc_df.keys():
        for row, _ in gc_df.iterrows():
            if gc_df[col][row] < 0.05:
                my_G.add_edge(node_loc2index[col], node_loc2index[row])
                try:
                    my_G[node_loc2index[col]][node_loc2index[row]]['weight'] = correlation[col][row]
                except:
                    x = 1

    cmap = plt.cm.seismic
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    frame2pillars = get_frame_to_alive_pillars()
    nodes_index2size = [10] * len(nodes_loc)
    for node in nodes_loc:
        for i in range(len(frame2pillars)):
            if node in frame2pillars[i + 1]:
                nodes_index2size[node_loc2index[str(node)]] = len(frame2pillars) - ((i // 13) * 13)
                break
    nodes_loc_y_inverse = [(loc[1], 1000 - loc[0]) for loc in nodes_loc]

    edges, weights = zip(*nx.get_edge_attributes(my_G, 'weight').items())
    # edges = list(filter(lambda x: x[0] == 52, edges))
    nx.draw(my_G, nodes_loc_y_inverse, with_labels=True, node_color='gray', edgelist=edges, edge_color=weights,
            width=3.0,
            edge_cmap=cmap,
            node_size=nodes_index2size)
    plt.colorbar(sm)
    plt.show()
    x = 1


def get_indirect_neighbors_correlation(pillar_location, only_alive=True):
    if only_alive:
        pillars_corr = get_alive_pillars_correlation()
    else:
        pillars_corr = get_all_pillars_correlation()

    pillar_directed_neighbors = get_pillar_directed_neighbors(pillar_location)

    pillar_directed_neighbors_str = []
    for tup in pillar_directed_neighbors:
        if tup != pillar_location:
            pillar_directed_neighbors_str.append(str(tup))
    pillars_corr = pillars_corr.drop(pillar_directed_neighbors_str, axis=0)
    pillars_corr = pillars_corr.drop(pillar_directed_neighbors_str, axis=1)

    return pillars_corr


def get_pillar_indirect_neighbors_dict(pillar_location):
    pillar_directed_neighbors = get_pillar_directed_neighbors(pillar_location)
    neighbors1, neighbors2 = get_pillar_to_neighbors()
    indirect_neighbors_dict = {}
    for n in neighbors1.keys():
        if n not in pillar_directed_neighbors:
            indirect_neighbors_dict[n] = neighbors1[n]
    for n in neighbors2.keys():
        if n not in pillar_directed_neighbors:
            indirect_neighbors_dict[n] = neighbors2[n]

    return indirect_neighbors_dict


def get_pillar_directed_neighbors(pillar_location):
    neighbors1, neighbors2 = get_pillar_to_neighbors()
    pillar_directed_neighbors = []
    pillar_directed_neighbors.extend(neighbors1[pillar_location])
    pillar_directed_neighbors.extend(neighbors2[pillar_location])
    pillar_directed_neighbors.append(pillar_location)

    return pillar_directed_neighbors


def correlation_histogram(correlations_df):
    corr = set()
    correlations = correlations_df
    for i in correlations:
        for j in correlations:
            if i != j:
                corr.add(correlations[i][j])
    corr_array = np.array(list(corr))
    ax = sns.histplot(data=corr_array, kde=True)
    plt.xlabel("Correlation")
    plt.show()


def plot_pillar_time_series(pillar_location):
    if _normalized:
        pillar2intens = normalized_intensities_by_mean_background_intensity()
    else:
        pillar2intens = get_pillar_to_intensities(get_images_path())
    pillar_loc = pillar_location
    intensities = pillar2intens[pillar_loc]
    x = [i * 19.87 for i in range(len(intensities))]
    intensities = [i * 0.0519938 for i in intensities]
    plt.plot(x, intensities)
    plt.xlabel('Time (sec)')
    plt.ylabel('Intensity (micron)')
    plt.title('Pillar ' + str(pillar_loc))
    plt.show()


def normalized_intensities_by_max_background_intensity():
    alive_pillars = get_alive_pillars_to_intensities()
    all_pillars = get_pillar_to_intensities(get_images_path())
    background_pillars_intensities = {pillar: all_pillars[pillar] for pillar in all_pillars.keys() if
                                      pillar not in alive_pillars}
    intensity_values_lst = list(background_pillars_intensities.values())
    max_int = max(max(intensity_values_lst, key=max))
    for pillar_item in all_pillars.items():
        for i, intensity in enumerate(pillar_item[1]):
            sub_int = int(intensity) - int(max_int)
            if sub_int > 0:
                all_pillars[pillar_item[0]][i] = sub_int
            else:
                all_pillars[pillar_item[0]][i] = 0

    return all_pillars


def normalized_intensities_by_mean_background_intensity():
    alive_pillars = get_alive_pillars_lst()
    all_pillars = get_pillar_to_intensities(get_images_path())
    background_pillars_intensities = {pillar: all_pillars[pillar] for pillar in all_pillars.keys() if
                                      pillar not in alive_pillars}

    background_intensity_values_lst = list(background_pillars_intensities.values())
    avg_intensity_in_frame = np.mean(background_intensity_values_lst, axis=0)
    for pillar_item in all_pillars.items():
        for i, intensity in enumerate(pillar_item[1]):
            sub_int = int(intensity) - avg_intensity_in_frame[i]
            all_pillars[pillar_item[0]][i] = sub_int

    return all_pillars


# correlation_plot(only_alive=False)
# indirect_alive_neighbors_correlation_plot((896, 807), False)
# correlation_histogram(get_alive_pillars_correlation(normalized=True))
# correlation_histogram(get_all_pillars_correlation(normalized=True))
# correlation_histogram(get_indirect_neighbors_correlation((511, 539)))
# plot_pillar_time_series((856, 740), normalized=True)
# build_pillars_mask()
# show_last_image_masked()
# normalized_intensities_by_max_background_intensity()
# normalized_intensities_by_mean_background_intensity()


def adf_test(df):
    result = adfuller(df.values)
    print('ADF Statistics: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def grangers_causation_matrix(data, variables, test='ssr_chi2test'):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=4, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(4)]
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


# p2i = get_pillar_to_intensities(get_images_path())
# p2i = normalized_intensities_by_max_background_intensity()
# p2i = get_alive_pillars_to_intensities(normalized_intensities=True)
# p2i_df = pd.DataFrame({str(k): v for k, v in p2i.items()})
# adf_test(p2i_df['(625, 740)'])

# data stationary
# p_vals = []
# for i in range(5):
#     for col in p2i_df:
#         res = adfuller(p2i_df[col])
#         p_vals.append(res[1])
#     p_vals = np.array(p_vals)
#     if p_vals.any() > 0.05:
#         p2i_df = p2i_df.diff().dropna()
#         p_vals = list(p_vals)
#     else:
#         break

# Find the lag for gc test. according to the lag where min of aic
# var_model = VAR(p2i_df)
# for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
#     result = var_model.fit(i)
#     try:
#         print('Lag Order =', i)
#         print('AIC : ', result.aic)
#         print('BIC : ', result.bic)
#         print('FPE : ', result.fpe)
#         print('HQIC: ', result.hqic, '\n')
#     except:
#         continue

# correlation_plot()
# gc_df = grangers_causation_matrix(p2i_df, p2i_df.columns)
# gc_plot(gc_df)
# _, pillar_adf_p_value, _, _, _, _ = adfuller(p2i_df['(625, 740)'])
# p2i_df_transformed = p2i_df.diff().dropna()
# adf_test(p2i_df_transformed['(625, 740)'])
# _, pillar_adf_p_value_transformed, _, _, _, _ = adfuller(p2i_df_transformed['(625, 740)'])
# results = model.fit(maxlags=3)

# gc_df = grangers_causation_matrix(p2i_df_transformed, p2i_df_transformed.columns)

# def granger_cause_plot(granger_cause_df, only_alive=True):
#     my_G = nx.Graph()
#     nodes_loc = find_centers()
#     node_loc2index = {}
#     for i in range(len(nodes_loc)):
#         node_loc2index[nodes_loc[i]] = i
#         my_G.add_node(i)
#     alive_pillars_correlation = granger_cause_df
#     # all_pillars_corr = get_all_pillars_correlation(normalized=True)
#             my_G.add_edge(node_loc2index[n1], node_loc2index[n2])
#             try:
#                 if only_alive and alive_pillars_correlation[str(n1) + '_x'][str(n2) + '_y'] < 0.05:
#                     my_G[node_loc2index[n1]][node_loc2index[n2]]['weight'] = alive_pillars_correlation[str(n1) + '_x'][str(n2) + '_y']
#                 # else:
#                 #     my_G[node_loc2index[n1]][node_loc2index[n2]]['weight'] = all_pillars_corr[str(n1)][str(n2)]
#
#             except:
#                 x = 1
#     for n1 in neighbors2.keys():
#         for n2 in neighbors2[n1]:
#             my_G.add_edge(node_loc2index[n1], node_loc2index[n2])
#             try:
#                 if only_alive and alive_pillars_correlation[str(n1) + '_x'][str(n2) + '_y'] < 0.05:
#                     my_G[node_loc2index[n1]][node_loc2index[n2]]['weight'] = alive_pillars_correlation[str(n1) + '_x'][str(n2) + '_y']
#                 # else:
#                 #     my_G[node_loc2index[n1]][node_loc2index[n2]]['weight'] = all_pillars_corr[str(n1)][str(n2)]
#
#             except:
#                 x = 1
#
#     edges, weights = zip(*nx.get_edge_attributes(my_G, 'weight').items())
#     cmap = plt.cm.seismic
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
#     frame2pillars = get_frame_to_alive_pillars()
#     nodes_index2size = [10] * len(nodes_loc)
#     for node in nodes_loc:
#         for i in range(len(frame2pillars)):
#             if node in frame2pillars[i + 1]:
#                 nodes_index2size[node_loc2index[node]] = len(frame2pillars) - ((i // 13) * 13)
#                 break
#     nodes_loc_y_inverse = [(loc[0], 1000 - loc[1]) for loc in nodes_loc]
#     nx.draw(my_G, nodes_loc_y_inverse, with_labels=True, node_color='gray', edgelist=edges, edge_color=weights,
#             width=3.0,
#             edge_cmap=cmap,
#             node_size=nodes_index2size)
#     plt.colorbar(sm)
#     plt.show()
#     x = 1


# for pillar in p2i:
#     # find derivative for stationary
#     for derivative in range(10):
#         alive_pillars_intensities_derivative = list(np.diff(p2i[pillar], n=derivative))
#         _, pillar_adf_p_value, _, _, _, _ = adfuller(alive_pillars_intensities_derivative)
#         if pillar_adf_p_value > 0.05:
#             continue
#         p2i[pillar] = alive_pillars_intensities_derivative
#         break
# p2i_df = pd.DataFrame({str(k): v for k, v in p2i.items()})
# var model to retrieve lag
# var_model = VAR(p2i_df_transformed)
# lag_order_results = var_model.select_order()
# estimators_lags = [
#     lag_order_results.aic,
#     lag_order_results.bic,
#     lag_order_results.fpe,
#     lag_order_results.hqic
# ]
# min_estimator_lag = min(estimators_lags)

# x=1

if __name__ == '__main__':
    # images = get_images(get_images_path())
    # with open('../SavedPillarsData/SavedPillarsData_06/last_image_06.npy', 'wb') as f:
    #     np.save(f, images[-1])
    # masks_path = PATH_MASKS_VIDEO_06_15_35
    # build_pillars_mask(
    #     masks_path=masks_path,
    #     logic_centers=True
    # )
    # show_last_image_masked(masks_path)
    correlation_plot(only_alive=False)
