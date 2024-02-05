import csv
import math
from Pillars.pillars_mask import *
import pickle
from skimage.filters import rank
from skimage.filters.thresholding import threshold_mean
from collections import deque as queue
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import numpy.ma as ma
from collections import defaultdict
from Pillars.consts import *
import json
from datetime import datetime
from sklearn.cluster import KMeans


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
        if Consts.SHOW_GRAPH:
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
        if Consts.SHOW_GRAPH:
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
    while len(q) > 0:
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
                if size >= Consts.MAX_CIRCLE_AREA:
                    return []
    return locations


# Accepting tif path
def get_images(path):
    images = io.imread(path)

    return images


def get_last_image():
    with open(Consts.last_image_path, 'rb') as f:
        image = np.load(f)
    # kernel = np.ones((10, 10), np.uint8) / 100
    # image = rank.mean(image, footprint=kernel)
    if len(image.shape) == 3:
        return image[-1]

    return image


def get_last_image_whiten(build_image=False):
    if build_image:
        img = create_image_by_max_value()
    else:
        img = get_last_image()

    return get_image_whiten(img)


def get_image_whiten(img):
    img = np.interp(img, (img.min(), img.max()), (0, 255)).astype('uint8')

    img[(img < Consts.pixel_to_whiten)] = 0
    img[(img >= Consts.pixel_to_whiten)] = 255

    return img


def get_mask_radiuses(mask_radius):
    small_mask_radius_ratio = mask_radius['small_radius'] / 20
    large_mask_radius_ratio = mask_radius['large_radius'] / 20
    small = math.floor(Consts.CIRCLE_RADIUS_FOR_MASK_CALCULATION * small_mask_radius_ratio)
    large = math.floor(Consts.CIRCLE_RADIUS_FOR_MASK_CALCULATION * large_mask_radius_ratio)
    return {'small': small, 'large': large}


def get_all_center_generated_ids():
    if Consts.USE_CACHE and os.path.isfile(Consts.centers_cache_path):
        with open(Consts.centers_cache_path, 'rb') as handle:
            centers = pickle.load(handle)
            return centers

    alive_centers = get_seen_centers_for_mask()

    if Consts.SHOW_GRAPH:
        plt.imshow(get_last_image(), cmap=plt.cm.gray)
        y = [center[0] for center in alive_centers]
        x = [center[1] for center in alive_centers]
        scatter_size = [3 for center in alive_centers]
        plt.scatter(x, y, s=scatter_size)
        plt.show()

    if len(alive_centers) == 0:
        raise Exception("Failed to found alive centers")

    centers = generate_centers_from_alive_centers(alive_centers, Consts.IMAGE_SIZE_ROWS, Consts.IMAGE_SIZE_COLS)

    if Consts.SHOW_GRAPH:
        plt.imshow(get_last_image(), cmap=plt.cm.gray)
        y = [center[0] for center in centers]
        x = [center[1] for center in centers]
        scatter_size = [3 for center in centers]

        plt.scatter(x, y, s=scatter_size)
        plt.show()

    if Consts.USE_CACHE:
        with open(Consts.centers_cache_path, 'wb') as handle:
            pickle.dump(centers, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return centers


def get_image_size():
    img = get_last_image()
    img_size = img.shape
    return img_size


def get_image_by_threshold(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    max_pixel = img.max()
    # find otsu's threshold value with OpenCV function
    ret, otsu = cv2.threshold(img, 0, max_pixel, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if ret is None or (not type(ret) == int and not type(ret) == float):
        ret = 500
    ret = min(ret, 500)
    img[img < ret] = 0
    img[img > 0] = 1
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()
    return img


# We use the get_seen_centers_for_mask so we won't run BFS for all possible location but only on given centers
def get_seen_centers_for_mask(img=None, all_center_ids=None):
    original_image = img
    if original_image is None and all_center_ids is None:
        if Consts.USE_CACHE and os.path.isfile(Consts.last_img_alive_centers_cache_path):
            with open(Consts.last_img_alive_centers_cache_path, 'rb') as handle:
                alive_centers = pickle.load(handle)
                return alive_centers
        img = get_last_image_whiten()

        plt.imshow(img, cmap=plt.cm.gray)
        if Consts.RESULT_FOLDER_PATH is not None:
            plt.savefig(Consts.RESULT_FOLDER_PATH + "/last_image_whiten.png")
            plt.close()  # close the figure window
            print("saved last_image_whiten.png")
    alive_centers = set()
    if all_center_ids is None:
        print("Finding seen centers for last frame")

        if Consts.tagged_centers is not None:
            alive_centers = Consts.tagged_centers
        else:
            visited = set()
            whiten_img = get_image_by_threshold(img)

            for row in range(0, len(whiten_img)):
                col = 0
                while col < len(whiten_img):
                    if not (row, col) in visited and kinda_center(whiten_img, row, col):
                        circle_area, center = get_center(whiten_img, row, col)
                        if len(circle_area) == 0:
                            col += 1
                        else:
                            visited.update(circle_area)
                            alive_centers.add(center)
                            col += Consts.CIRCLE_RADIUS
                    else:
                        col += 1
    else:
        img_avg = np.average(img)  # was const of 150
        img_median = np.median(img)  # was const of 350

        for possible_location_to_search_alive_center in all_center_ids:
            repositioned_alive_center = get_center_fixed_by_circle_mask_reposition(
                possible_location_to_search_alive_center, img)
            outer_circle_mask = get_mask_for_center(repositioned_alive_center)
            outer_circle_mask[outer_circle_mask == 255] = 1
            # If valid location
            if 0 <= repositioned_alive_center[0] < len(img) and 0 < repositioned_alive_center[1] <= len(img[0]):
                inner_circle_mask = get_pillar_circle_mask_for_movement(repositioned_alive_center)
                inner_circle_masked = img * inner_circle_mask
                inner_circle_mask_intens = np.sum(inner_circle_masked)
                avg_inner_pixel_intens = inner_circle_mask_intens / len(inner_circle_mask[inner_circle_mask != 0])

                outer_circle_masked = img * outer_circle_mask
                outer_circle_mask_intens = np.sum(outer_circle_masked)
                avg_outer_pixel_intens = outer_circle_mask_intens / len(outer_circle_mask[outer_circle_mask != 0])

                median_outer_circle = np.median(outer_circle_masked[(outer_circle_mask == 1).nonzero()])

                white_pixels_in_outer_circle = len(outer_circle_masked[(outer_circle_masked > img_avg)])
                total_pixels_in_outer_circle = len(outer_circle_masked[(outer_circle_mask == 1).nonzero()])

                if avg_inner_pixel_intens * Consts.INTENSITIES_RATIO_OUTER_INNER < avg_outer_pixel_intens and avg_outer_pixel_intens >= img_avg and median_outer_circle >= img_median:
                    if white_pixels_in_outer_circle / total_pixels_in_outer_circle >= 0.5:
                        alive_centers.add(repositioned_alive_center)

    if Consts.tagged_centers is not None and all_center_ids is None:
        alive_centers_fixed = alive_centers
    else:
        alive_centers_fixed = get_centers_fixed_by_circle_mask_reposition(alive_centers, img)

    # If boolean is true, need to add all tagged centers to alive_centers_fixed (only if not in list)
    if Consts.ALL_TAGGED_ALWAYS_ALIVE:
        tagged_centers_fixed = get_centers_fixed_by_circle_mask_reposition(Consts.tagged_centers, img)
        for fixed_tagged_center in tagged_centers_fixed:
            closest_center_to_tagged = closest_to_point(alive_centers_fixed, fixed_tagged_center)
            # If there is no seen center to tagged -> add the tagged to the list.
            if closest_center_to_tagged is None or \
                    not np.linalg.norm(np.array(fixed_tagged_center) - np.array(closest_center_to_tagged)) < \
                        Consts.MAX_DISTANCE_PILLAR_FIXED:
                alive_centers_fixed.append(fixed_tagged_center)

    if original_image is None and all_center_ids is None and Consts.USE_CACHE:

        plt.imshow(img, cmap=plt.cm.gray)
        if Consts.RESULT_FOLDER_PATH is not None:
            plt.imshow(get_last_image(), cmap=plt.cm.gray)
            y = [center[0] for center in alive_centers_fixed]
            x = [center[1] for center in alive_centers_fixed]
            scatter_size = [3 for center in alive_centers_fixed]

            plt.scatter(x, y, s=scatter_size)
            plt.text(10, 10, str(len(alive_centers_fixed)), fontsize=8)

            plt.savefig(Consts.RESULT_FOLDER_PATH + "/last_frame_alive_pillars_seen_v2.png")
            plt.close()  # close the figure window
            print("last_frame_alive_pillars_seen.png")

        with open(Consts.last_img_alive_centers_cache_path, 'wb') as handle:
            pickle.dump(alive_centers_fixed, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return alive_centers_fixed


def generate_centers_from_alive_centers(alive_centers, matrix_row_size, matrix_col_size):
    return generate_centers_and_rules_from_alive_centers(alive_centers, matrix_row_size, matrix_col_size)[0]


def generate_centers_and_rules_from_alive_centers(alive_centers, matrix_row_size, matrix_col_size):
    closest_to_middle, rule1, rule2 = get_center_generation_rules(alive_centers, matrix_row_size, matrix_col_size)

    generated_center_ids = generate_centers_by_pillar_loc(closest_to_middle,
                                                          matrix_col_size,
                                                          matrix_row_size,
                                                          rule1,
                                                          rule2)

    # should_use_cluster = False
    # if should_use_cluster:
    #     generated_centers_better_loc_by_alive_centers = generate_centers_by_closest_alive_centers(alive_centers,
    #                                                                                               matrix_col_size,
    #                                                                                               matrix_row_size,
    #                                                                                               rule1,
    #                                                                                               rule2)
    #
    #     alive_centers = generated_centers_better_loc_by_alive_centers

    # Replaced center IDs with better found locations
    generated_location2real_pillar_loc = {}
    for better_location in alive_centers:
        closest_generated_center = min(generated_center_ids,
                                       key=lambda point: math.hypot(better_location[1] - point[1],
                                                                    better_location[0] - point[0]))
        if np.linalg.norm(
                np.array(closest_generated_center) - np.array(
                    better_location)) < Consts.MAX_DISTANCE_PILLAR_FIXED or Consts.tagged_centers is not None:
            generated_center_ids.remove(closest_generated_center)
            generated_center_ids.append(better_location)
            generated_location2real_pillar_loc[closest_generated_center] = better_location
        else:
            far = True
    # Generated center = pillar id.

    # TODO: consider use get_centers_fixed_by_circle_mask_reposition also for generated location - to have better position for them,
    #  if we'll do that - don't forget to update mapping of generated_location2real_pillar_loc
    return generated_center_ids, rule1, rule2, generated_location2real_pillar_loc


def generate_centers_by_closest_alive_centers(alive_centers,
                                              matrix_col_size,
                                              matrix_row_size,
                                              rule1,
                                              rule2):
    final_list_generated_locations = []
    alive_pillar2generated_centers = {}
    all_generated_centers = []
    for alive_pillar in alive_centers:
        generated_centers = generate_centers_by_pillar_loc(alive_pillar, matrix_col_size, matrix_row_size, rule1, rule2)
        alive_pillar2generated_centers[alive_pillar] = generated_centers
        all_generated_centers.extend(generated_centers)

    kmeans = KMeans(n_clusters=max([len(v) for v in alive_pillar2generated_centers.values()]))
    kmeans.fit(all_generated_centers)
    # clusters = {i:[] for i in range(len(alive_centers))}
    # for i in range(len(kmeans.labels_)):
    #     cluster_id = kmeans.labels_[i]
    #     clusters[cluster_id].add(all_generated_centers[i])

    if Consts.SHOW_GRAPH:
        plt.imshow(get_last_image(), cmap=plt.cm.gray)
        y = [center[0] for center in all_generated_centers]
        x = [center[1] for center in all_generated_centers]
        scatter_size = [1 for center in all_generated_centers]

        plt.scatter(x, y, s=scatter_size)
        plt.show()

    for cluster_center in kmeans.cluster_centers_:
        closest_alive = closest_to_point(alive_centers, cluster_center)
        closest_generated_center_from_alive = closest_to_point(alive_pillar2generated_centers[closest_alive],
                                                               cluster_center)
        final_list_generated_locations.append(closest_generated_center_from_alive)

    return final_list_generated_locations


def generate_centers_by_pillar_loc(pillar_loc, matrix_col_size, matrix_row_size, rule1, rule2):
    generated_centers_in_line = {pillar_loc}
    row = pillar_loc[0]
    col = pillar_loc[1]
    while -2 * matrix_row_size <= row < matrix_row_size * 2 and -2 * matrix_col_size <= col < matrix_col_size * 2:
        generated_centers_in_line.add((row, col))
        row += rule1[0]
        col += rule1[1]
    row = pillar_loc[0]
    col = pillar_loc[1]
    while matrix_row_size * 2 > row >= -2 * matrix_row_size and -2 * matrix_col_size <= col < matrix_col_size * 2:
        generated_centers_in_line.add((row, col))
        row -= rule1[0]
        col -= rule1[1]
    generated_centers = set(generated_centers_in_line)
    for center in generated_centers_in_line:
        row = center[0]
        col = center[1]
        while matrix_row_size * 2 > row >= - 2 * matrix_row_size and -2 * matrix_col_size <= col < matrix_col_size * 2:
            generated_centers.add((row, col))
            row += rule2[0]
            col += rule2[1]
    for center in generated_centers_in_line:
        row = center[0]
        col = center[1]
        while matrix_row_size * 2 > row >= -2 * matrix_row_size and -2 * matrix_col_size <= col < matrix_col_size * 2:
            generated_centers.add((row, col))
            row -= rule2[0]
            col -= rule2[1]
    actual_centers_in_range = [center for center in list(generated_centers) if
                               0 <= center[0] < matrix_row_size and 0 <= center[1] < matrix_col_size]
    return actual_centers_in_range


def get_center_generation_rules(alive_centers, matrix_row_size, matrix_col_size):
    try:
        target = get_middle_of_dense_centers(alive_centers)
    except:
        target = (matrix_row_size // 2, matrix_col_size // 2)
    closest_to_middle = min(alive_centers, key=lambda point: math.hypot(target[1] - point[1], target[0] - point[0]))
    rules = get_rules_by_all_centers(alive_centers)
    if rules is not None:
        rule1, rule2 = rules
    else:
        rule1, rule2 = get_rules_by_middle_centers(alive_centers, closest_to_middle)
    return closest_to_middle, rule1, rule2


def get_rules_by_middle_centers(alive_centers, closest_to_middle):
    points = list(alive_centers)
    points.remove(closest_to_middle)
    closest1 = min(points,
                   key=lambda point: math.hypot(closest_to_middle[1] - point[1], closest_to_middle[0] - point[0]))
    points.remove(closest1)
    closest2 = min(points,
                   key=lambda point: math.hypot(closest_to_middle[1] - point[1], closest_to_middle[0] - point[0]))
    rule1 = (closest_to_middle[0] - closest1[0], closest_to_middle[1] - closest1[1])
    rule2 = (closest_to_middle[0] - closest2[0], closest_to_middle[1] - closest2[1])
    # If both rules are pretty much the same rule, meaning we are on the same line.
    while -5 <= abs(rule1[0]) - abs(rule2[0]) <= 5 and -5 <= abs(rule1[1]) - abs(rule2[1]) <= 5:
        points.remove(closest2)
        closest2 = min(points,
                       key=lambda point: math.hypot(closest_to_middle[1] - point[1], closest_to_middle[0] - point[0]))
        rule2 = (closest_to_middle[0] - closest2[0], closest_to_middle[1] - closest2[1])
    return rule1, rule2


def get_middle_of_dense_centers(alive_centers):
    def gauss(x1, x2, y1, y2, radius):
        """
        Apply a Gaussian kernel estimation (2-sigma) to distance between points.

        Effectively, this applies a Gaussian kernel with a fixed radius to one
        of the points and evaluates it at the value of the euclidean distance
        between the two points (x1, y1) and (x2, y2).
        The Gaussian is transformed to roughly (!) yield 1.0 for distance 0 and
        have the 2-sigma located at radius distance.
        """
        return (
                (1.0 / (2.0 * math.pi))
                * math.exp(
            -1 * (3.0 * math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / radius)) ** 2
                / 0.4)

    def _kde(x, y):
        """
        Estimate the kernel density at a given position.

        Simply sums up all the Gaussian kernel values towards all points
        (pts_x, pts_y) from position (x, y).
        """
        return sum([
            gauss(x, px, y, py, radius)
            # math.sqrt((x - px)**2 + (y - py)**2)
            for px, py in zip(pts_x, pts_y)
        ])

    pts_x = np.array([center[0] for center in alive_centers])
    pts_y = np.array([center[1] for center in alive_centers])

    RESOLUTION = 50
    LOCALITY = 2.0

    dx = max(pts_x) - min(pts_x)
    dy = max(pts_y) - min(pts_y)

    delta = min(dx, dy) / RESOLUTION
    nx = int(dx / delta)
    ny = int(dy / delta)
    radius = (1 / LOCALITY) * min(dx, dy)

    grid_x = np.linspace(min(pts_x), max(pts_x), num=nx)
    grid_y = np.linspace(min(pts_y), max(pts_y), num=ny)

    x, y = np.meshgrid(grid_x, grid_y)

    kde = np.vectorize(_kde)  # Let numpy care for applying our kde to a vector
    z = kde(x, y)

    xi, yi = np.where(z == np.amax(z))
    max_x = grid_x[xi][0]
    max_y = grid_y[yi][0]
    # print(f"{max_x:.4f}, {max_y:.4f}")
    #
    # fig, ax = plt.subplots()
    # ax.pcolormesh(x, y, z, cmap='inferno', vmin=np.min(z), vmax=np.max(z))
    # fig.set_size_inches(4, 4)
    # fig.savefig('density.png', bbox_inches='tight')
    #
    # fig, ax = plt.subplots()
    # ax.scatter(pts_x, pts_y, marker='+', color='blue')
    # ax.scatter(grid_x[xi], grid_y[yi], marker='+', color='red', s=200)
    # fig.set_size_inches(4, 4)
    # fig.savefig('marked.png', bbox_inches='tight')

    return (max_x, max_y)


def get_rules_by_all_centers(alive_centers):
    if len(alive_centers) < 4:
        return None
    distance_counters = defaultdict(int)
    for center in alive_centers:
        points = list(alive_centers)
        points.remove(center)
        points.sort(key=lambda K: (K[0] - center[0]) ** 2 + (K[1] - center[1]) ** 2)
        top4centers = points[:4]

        distance1 = (top4centers[0][0] - center[0], top4centers[0][1] - center[1])
        distance2 = (top4centers[1][0] - center[0], top4centers[1][1] - center[1])
        distance3 = (top4centers[2][0] - center[0], top4centers[2][1] - center[1])
        distance4 = (top4centers[3][0] - center[0], top4centers[3][1] - center[1])
        distance1_op = (-(top4centers[0][0] - center[0]), -(top4centers[0][1] - center[1]))
        distance2_op = (-(top4centers[1][0] - center[0]), -(top4centers[1][1] - center[1]))
        distance3_op = (-(top4centers[2][0] - center[0]), -(top4centers[2][1] - center[1]))
        distance4_op = (-(top4centers[3][0] - center[0]), -(top4centers[3][1] - center[1]))

        if distance1_op not in distance_counters:
            distance_counters[distance1] += 1
        else:
            distance_counters[distance1_op] += 1

        if distance2_op not in distance_counters:
            distance_counters[distance2] += 1
        else:
            distance_counters[distance2_op] += 1

        if distance3_op not in distance_counters:
            distance_counters[distance3] += 1
        else:
            distance_counters[distance3_op] += 1

        if distance4_op not in distance_counters:
            distance_counters[distance4] += 1
        else:
            distance_counters[distance4_op] += 1

    rules = []
    for distance_counter in distance_counters.items():
        rule = distance_counter[0]
        rules.extend([rule] * distance_counter[1])

    rule_groups = group_points(rules)

    sorted_rules = sorted(rule_groups, key=lambda x: len(x), reverse=True)
    max_rules = sorted_rules[0]
    # rule1 = max(max_rules, key=max_rules.count) # TODO

    rule1 = np.average(max_rules, axis=0)
    rule1 = (int(rule1[0]), int(rule1[1]))

    for rule2 in sorted_rules[1:]:
        # rule2 = max(rule2, key=rule2.count)  # TODO

        rule2 = np.average(rule2, axis=0)
        rule2 = (int(rule2[0]), int(rule2[1]))

        if rule2[0] != 0 and rule2[1] != 0 and (rule1[1] / rule2[1]) != 0:
            # Check if rules are multiplication of one another (meaning - same rule)
            if 0.95 <= (rule1[0] / rule2[0]) / (rule1[1] / rule2[1]) <= 1.05:
                continue

        # If both rules different, meaning we are not on the same line.
        if -5 <= abs(rule2[0]) - abs(rule1[0]) <= 5 and -5 <= abs(rule2[1]) - abs(rule1[1]) <= 5:
            continue
        return rule1, rule2
    return None


def group_points(points):
    groups = []
    while points:
        far_points = []
        ref = points.pop()
        groups.append([ref])
        for point in points:
            d = get_distance(ref, point)
            if d < 5:
                groups[-1].append(point)
            else:
                far_points.append(point)

        points = far_points

    # perform average operation on each group
    return groups


def get_distance(ref, point):
    # print('ref: {} , point: {}'.format(ref, point))
    x1, y1 = ref
    x2, y2 = point
    return math.hypot(x2 - x1, y2 - y1)


def kinda_center(img, row, col):
    col_zeros = img[
                row:row + 1,
                col - Consts.CIRCLE_INSIDE_VALIDATE_SEARCH_LENGTH: col + Consts.CIRCLE_INSIDE_VALIDATE_SEARCH_LENGTH]
    if np.any(col_zeros):
        return False
    row_zeros = img[
                row - Consts.CIRCLE_INSIDE_VALIDATE_SEARCH_LENGTH: row + Consts.CIRCLE_INSIDE_VALIDATE_SEARCH_LENGTH,
                col: col + 1]
    if np.any(row_zeros):
        return False

    SAFETY_DISTANCE_FROM_CIRCLE = 6

    white_pixels_in_surronding = 0
    total_pixles_in_surronging = SAFETY_DISTANCE_FROM_CIRCLE * 4

    for safety_distance in range(Consts.CIRCLE_OUTSIDE_VALIDATE_SEARCH_LENGTH,
                                 Consts.CIRCLE_OUTSIDE_VALIDATE_SEARCH_LENGTH + SAFETY_DISTANCE_FROM_CIRCLE):

        # If not out of image (if we reached out of image, this is not a pillar)
        if col - safety_distance >= 0 and img.shape[1] > col + safety_distance and row - safety_distance >= 0 and \
                img.shape[0] > row + safety_distance:
            if img[row][col + safety_distance] == 1:
                white_pixels_in_surronding += 1
            if img[row][col - safety_distance] == 1:
                white_pixels_in_surronding += 1
            if img[row - safety_distance][col] == 1:
                white_pixels_in_surronding += 1
            if img[row + safety_distance][col] == 1:
                white_pixels_in_surronding += 1
        else:
            return False

    return white_pixels_in_surronding >= total_pixles_in_surronging * 0.5


def get_center(img, row, col):
    rows = len(img)
    cols = len(img[0])
    # Mark all cells as not visited
    vis = [[False for i in range(cols)]
           for i in range(rows)]
    circle_area = BFS(img, vis, row, col)
    if len(circle_area) == 0:
        return [], (0, 0)
    maybe_center = get_center_of_points(circle_area)
    centralized_center = centralize_center(img, maybe_center)
    return circle_area, centralized_center


def get_center_of_points(circle_area_points):
    X = [tup[0] for tup in circle_area_points]
    Y = [tup[1] for tup in circle_area_points]
    avg_X = sum(X) / len(X)
    avg_Y = sum(Y) / len(Y)

    return (int(avg_X), int(avg_Y))


# The center we got is based on average, we want to make sure this is not caused by noise, so we try
# normalize it if needed
def centralize_center(img, center):
    FIND_NEW_CENTER_IN_DISTANCE = Consts.CIRCLE_RADIUS // 2
    best_center = center
    max_gap_center = float('inf')
    for row_movement in range(-FIND_NEW_CENTER_IN_DISTANCE, FIND_NEW_CENTER_IN_DISTANCE + 1):
        for col_movement in range(-FIND_NEW_CENTER_IN_DISTANCE, FIND_NEW_CENTER_IN_DISTANCE + 1):
            optional_center = (center[0] + row_movement, center[1] + col_movement)

            if img[optional_center[0], optional_center[1]] == 1:
                break

            current_loc = (center[0] + row_movement, center[1] + col_movement)

            # Check distance to left until end of circle
            while img[current_loc[0], current_loc[1]] == 0:
                current_loc = (current_loc[0], current_loc[1] - 1)
            distance_to_end_of_circle_left = abs(optional_center[1] - current_loc[1])

            # Check distance to right until end of circle
            current_loc = (center[0] + row_movement, center[1] + col_movement)
            while img[current_loc[0], current_loc[1]] == 0:
                current_loc = (current_loc[0], current_loc[1] + 1)
            distance_to_end_of_circle_right = abs(optional_center[1] - current_loc[1])

            current_loc = (center[0] + row_movement, center[1] + col_movement)

            # Check distance to top until end of circle
            while img[current_loc[0], current_loc[1]] == 0:
                current_loc = (current_loc[0] - 1, current_loc[1])
            distance_to_end_of_circle_top = abs(optional_center[0] - current_loc[0])

            # Check distance to down until end of circle
            current_loc = (center[0] + row_movement, center[1] + col_movement)
            while img[current_loc[0], current_loc[1]] == 0:
                current_loc = (current_loc[0] + 1, current_loc[1])
            distance_to_end_of_circle_down = abs(optional_center[0] - current_loc[0])

            horizontal_diff = abs(distance_to_end_of_circle_left - distance_to_end_of_circle_right)
            vertical_diff = abs(distance_to_end_of_circle_top - distance_to_end_of_circle_down)
            # If center might be valid
            if horizontal_diff <= 1 and vertical_diff <= 1:
                # If better than current
                if horizontal_diff + vertical_diff < max_gap_center:
                    max_gap_center = horizontal_diff + vertical_diff
                    best_center = optional_center

    return best_center


def get_images_path():
    if Consts.fixed:
        return Consts.fixed_images_path
    else:
        return Consts.images_path


# Checks for alive centers by all centers from last image only
# so if center is valid but it's not in same position as in the last frame - it won't be found
#
def deprecated_get_frame_to_alive_pillars_by_same_mask(pillar2last_mask):
    if Consts.USE_CACHE and os.path.isfile(Consts.frame2pillar_cache_path):
        with open(Consts.frame2pillar_cache_path, 'rb') as handle:
            frame_to_alive_pillars = pickle.load(handle)
            return frame_to_alive_pillars
    frame_to_alive_pillars = {}
    images = get_images(get_images_path())
    frame_num = 1
    for frame in images:

        relevant_pillars_in_frame = []
        # TODO: [not critical to be fixed ATM] can't check if alive by last mask, need to check for each frame
        for pillar_item in pillar2last_mask.items():
            pillar = pillar_item[0]
            pillar_mask = pillar_item[1]
            if _considered_as_alive_pillar(pillar_mask, get_image_whiten(frame), pillar):
                relevant_pillars_in_frame.append(pillar)
        frame_to_alive_pillars[frame_num] = relevant_pillars_in_frame
        frame_num += 1

    if Consts.USE_CACHE:
        with open(Consts.frame2pillar_cache_path, 'wb') as handle:
            pickle.dump(frame_to_alive_pillars, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return frame_to_alive_pillars


def get_alive_pillar_ids_in_last_frame_v3():
    if Consts.inner_cell:
        return get_alive_pillar_ids_overall_v3()
    else:
        return _get_alive_pillar_ids_lst()


def _get_alive_pillar_ids_lst():
    frame_to_pillars = get_alive_center_ids_by_frame_v3()  # get_frame_to_alive_pillars_by_same_mask(pillar2mask)
    any_time_live_pillars = {}
    for pillars in frame_to_pillars.values():
        any_time_live_pillars.update(pillars)

    return list(any_time_live_pillars)


def deprecated_get_fully_alive_pillars_lst(pillar2mask_dict):
    if Consts.USE_CACHE and os.path.isfile(Consts.alive_pillars_overall):
        with open(Consts.alive_pillars_overall, 'rb') as handle:
            fully_alive_pillars_lst = pickle.load(handle)
            return fully_alive_pillars_lst

    img = get_last_image_whiten()
    fully_alive_pillars_lst = []
    for p, mask in pillar2mask_dict.items():
        if _considered_as_alive_pillar(mask, img, p):
            fully_alive_pillars_lst.append(p)

    plt.imshow(img, cmap=plt.cm.gray)
    y = [center[0] for center in fully_alive_pillars_lst]
    x = [center[1] for center in fully_alive_pillars_lst]
    scatter_size = [3 for center in fully_alive_pillars_lst]

    plt.scatter(x, y, s=scatter_size)
    plt.text(10, 10, str(len(fully_alive_pillars_lst)), fontsize=8)

    if Consts.RESULT_FOLDER_PATH is not None:
        plt.savefig(Consts.RESULT_FOLDER_PATH + "/deprecated_get_fully_alive_pillars.png")
        plt.close()  # close the figure window
        print("saved alive_pillar_last_frame.png")
    if Consts.SHOW_GRAPH:
        plt.show()

    if Consts.USE_CACHE:
        with open(Consts.alive_pillars_overall, 'wb') as handle:
            pickle.dump(fully_alive_pillars_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return fully_alive_pillars_lst


def _considered_as_alive_pillar(mask, img, pillar):
    # Set both mask and img as [0/255] dfs
    mask[mask > 0] = 255

    max_intensity_in_mask = mask.sum() * Consts.percentage_from_perfect_circle_mask
    frame_masked = np.where(mask, img, 0)
    is_rim_white = (np.sum(frame_masked) / max_intensity_in_mask) * 100 > Consts.consider_as_full_circle_percentage

    return is_rim_white

    #
    # pillar_center_mask = get_pillar_circle_mask_for_movement(pillar)
    # image_masked = img * pillar_center_mask
    # pillar_intens = np.sum(image_masked)
    # is_inside_black = pillar_intens < 12000
    # return is_rim_white and is_inside_black


def create_image_by_max_value():
    images = get_images(get_images_path())
    new_img = np.max(np.array(images), axis=0)

    return new_img


def get_alive_pillar_ids_overall_v3():
    if Consts.USE_CACHE and os.path.isfile(Consts.alive_pillars_overall):
        with open(Consts.alive_pillars_overall, 'rb') as handle:
            all_alive_pillar_ids_overall = pickle.load(handle)
            return all_alive_pillar_ids_overall

    alive_center_ids_by_frame = get_alive_center_ids_by_frame_v3()
    all_alive_pillar_ids_overall = []
    for centers_in_frame in alive_center_ids_by_frame.values():
        all_alive_pillar_ids_overall.extend(centers_in_frame)

    plt.imshow(get_last_image(), cmap=plt.cm.gray)
    y = [center[0] for center in all_alive_pillar_ids_overall]
    x = [center[1] for center in all_alive_pillar_ids_overall]
    scatter_size = [3 for center in all_alive_pillar_ids_overall]

    plt.scatter(x, y, s=scatter_size)
    plt.text(10, 10, str(len(all_alive_pillar_ids_overall)), fontsize=8)

    if Consts.RESULT_FOLDER_PATH is not None:
        plt.savefig(Consts.RESULT_FOLDER_PATH + "/alive_pillar_overall_ids_v3.png")
        plt.close()  # close the figure window
    if Consts.SHOW_GRAPH:
        plt.show()

    all_alive_pillar_ids_overall = set(all_alive_pillar_ids_overall)

    if Consts.USE_CACHE:
        with open(Consts.alive_pillars_overall, 'wb') as handle:
            pickle.dump(all_alive_pillar_ids_overall, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_alive_pillar_ids_overall


def get_alive_center_ids_by_frame_v3():
    if Consts.USE_CACHE and os.path.isfile(Consts.alive_center_ids_by_frame_cache_path):
        with open(Consts.alive_center_ids_by_frame_cache_path, 'rb') as handle:
            alive_center_ids_by_frame = pickle.load(handle)
            return alive_center_ids_by_frame

    center_ids = get_all_center_generated_ids()
    center_location_by_frame = get_alive_centers_real_location_by_frame_v2()
    alive_center_ids_by_frame = {}

    # Fitting centers to their ids
    for curr_frame, curr_frame_center_real_locations in center_location_by_frame.items():
        frame_center_ids = []
        for center_real_location in curr_frame_center_real_locations:
            # Find closest id
            closest_center_id = min(center_ids,
                                    key=lambda center_id:
                                    math.hypot(center_id[1] - center_real_location[1],
                                               center_id[0] - center_real_location[0]))

            distance = math.hypot(closest_center_id[1] - center_real_location[1],
                                  closest_center_id[0] - center_real_location[0])
            if 0 <= distance <= math.sqrt(2 * pow(Consts.FIND_BETTER_CENTER_IN_RANGE, 2)):
                frame_center_ids.append(closest_center_id)
            else:
                center_is_far_away = True
        alive_center_ids_by_frame[curr_frame] = frame_center_ids

    # Join all center ids found in all frames, and set them as alive centers for all frames.
    if not Consts.is_spreading:
        centers_list_of_lists = list(alive_center_ids_by_frame.values())
        centers_list = [item for sublist in centers_list_of_lists for item in sublist]
        unique_centers = set(centers_list)
        alive_center_ids_by_frame = {k: list(unique_centers) for k in alive_center_ids_by_frame.keys()}

    with open(Consts.RESULT_FOLDER_PATH + "/amount_pillars_each_frame.json", 'w') as f:
        key_to_len_val = {str(k): len(v) for k, v in alive_center_ids_by_frame.items()}
        json.dump(key_to_len_val, f)

    if Consts.USE_CACHE:
        with open(Consts.alive_center_ids_by_frame_cache_path, 'wb') as handle:
            pickle.dump(alive_center_ids_by_frame, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return alive_center_ids_by_frame


# Checks for alive centers by all generated centers -
# so if center is valid but isn't alive in last frame - it will be found
def get_alive_centers_real_location_by_frame_v2(images=None):  #
    if Consts.USE_CACHE and os.path.isfile(Consts.alive_center_real_locations_by_frame_cache_path):
        with open(Consts.alive_center_real_locations_by_frame_cache_path, 'rb') as handle:
            frame_to_alive_pillars_real_location = pickle.load(handle)
            return frame_to_alive_pillars_real_location

    if images is None:
        images = get_images(get_images_path())

    center_ids = get_all_center_generated_ids()

    frame_to_alive_pillars_real_location = {}
    for frame_index, frame in enumerate(images):
        frame_to_alive_pillars_real_location[frame_index] = list(get_seen_centers_for_mask(frame, center_ids))
        print("finished finding centers", str(len(frame_to_alive_pillars_real_location[frame_index])),
              "for frame " + str(frame_index) + " / " + str(len(images)), str(datetime.now()))

    # Print only
    last_frame_alive_pillars = list(frame_to_alive_pillars_real_location.values())[-1]
    plt.imshow(get_last_image(), cmap=plt.cm.gray)
    y = [center[0] for center in last_frame_alive_pillars]
    x = [center[1] for center in last_frame_alive_pillars]
    scatter_size = [3 for center in last_frame_alive_pillars]

    plt.scatter(x, y, s=scatter_size)
    plt.text(10, 10, str(len(last_frame_alive_pillars)), fontsize=8)

    if Consts.RESULT_FOLDER_PATH is not None:
        plt.savefig(Consts.RESULT_FOLDER_PATH + "/last_frame_alive_pillars_locs_v2.png")
        plt.close()  # close the figure window
        print("last_frame_alive_pillars_locs.png")

    if Consts.USE_CACHE:
        with open(Consts.alive_center_real_locations_by_frame_cache_path, 'wb') as handle:
            pickle.dump(frame_to_alive_pillars_real_location, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return frame_to_alive_pillars_real_location


# In case a center moved too much, meaning it is a whiten issue or something - log it
def log_major_center_movements(frame_to_alive_pillars):
    if Consts.RESULT_FOLDER_PATH is None:
        return

    movements = {}
    movements["empty_frames"] = []
    for curr_frame in range(0, len(frame_to_alive_pillars) - 1):
        centers_curr_frame = frame_to_alive_pillars[curr_frame]
        if len(centers_curr_frame) == 0:
            movements["empty_frames"].append(curr_frame)
            continue
        next_frame = curr_frame + 1
        centers_next_frame = frame_to_alive_pillars[next_frame]

        for center_next_frame in centers_next_frame:
            # Find closest one in curr_frame -> if very close, all good, if very far - probably not the same.
            # if in between - bug.
            curr_frame_alive_closest = min(centers_curr_frame,
                                           key=lambda center_curr_frame:
                                           math.hypot(center_next_frame[1] - center_curr_frame[1],
                                                      center_next_frame[0] - center_curr_frame[0]))

            distance = math.hypot(center_next_frame[1] - curr_frame_alive_closest[1],
                                  center_next_frame[0] - curr_frame_alive_closest[0])
            # TODO: 3 < distance < 15 to change? also to consts
            # TODO: not saving properly to json
            if 3 < distance < 15:
                if next_frame not in movements:
                    movements[next_frame] = {}
                movements[next_frame][str(center_next_frame)] = {'closest': curr_frame_alive_closest,
                                                                 'distance': distance}

    with open(Consts.RESULT_FOLDER_PATH + "/major_movements_bugs.json", 'w') as f:
        json.dump(movements, f)


def get_pillar_circle_mask_for_movement(center):
    thickness = -1
    pillar_circle_mask = np.zeros((Consts.IMAGE_SIZE_ROWS, Consts.IMAGE_SIZE_COLS), np.uint8)
    cv2.circle(pillar_circle_mask, (center[1], center[0]), Consts.CIRCLE_RADIUS, 1, thickness)
    return pillar_circle_mask


# This method doesn't return actual center Ids, but their locations. (used in movements)
def get_frame_to_alive_pillars_reposition_v2():
    if Consts.USE_CACHE and os.path.isfile(Consts.alive_pillars_by_frame_reposition_cache_path):
        with open(Consts.alive_pillars_by_frame_reposition_cache_path, 'rb') as handle:
            frame_to_alive_pillars_reposition = pickle.load(handle)
            return frame_to_alive_pillars_reposition

    # Those centers are not final location, we want to find real centers locations
    frame_images = get_images(get_images_path())
    frame_to_alive_pillars = get_alive_centers_real_location_by_frame_v2(frame_images)

    frame_to_alive_pillars_reposition = {}

    for frame in frame_to_alive_pillars:
        curr_frame_img = frame_images[frame]
        fixed_centers = get_centers_fixed_by_circle_mask_reposition(frame_to_alive_pillars[frame], curr_frame_img)
        frame_to_alive_pillars_reposition[frame] = fixed_centers

    print("find fixed centers by frames")

    frame_to_num_of_alive_pillars = {}
    for k, v in frame_to_alive_pillars_reposition.items():
        frame_to_num_of_alive_pillars[k] = len(v)
    with open(Consts.RESULT_FOLDER_PATH + "/frame_to_num_of_alive_pillars_v2.txt", 'w') as f:
        json.dump(frame_to_num_of_alive_pillars, f)

    if Consts.USE_CACHE:
        with open(Consts.alive_pillars_by_frame_reposition_cache_path, 'wb') as handle:
            pickle.dump(frame_to_alive_pillars_reposition, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return frame_to_alive_pillars_reposition


def get_centers_fixed_by_circle_mask_reposition(centers, img):
    fixed_centers = []
    for center in centers:
        repositioned_center = get_center_fixed_by_circle_mask_reposition(center, img)
        fixed_centers.append(repositioned_center)
    return fixed_centers


def get_center_fixed_by_circle_mask_reposition(center, img, opt_fixing_range=None):
    if opt_fixing_range is None:
        fixing_range = Consts.FIND_BETTER_CENTER_IN_RANGE
    else:
        fixing_range = opt_fixing_range
    center_with_min_intensities = center
    min_intensities = float('inf')
    for row in range(center[0] - fixing_range, center[0] + fixing_range):
        for col in range(center[1] - fixing_range, center[1] + fixing_range):
            optional_center = (row, col)
            optional_center_mask = get_pillar_circle_mask_for_movement(optional_center)
            image_masked = img * optional_center_mask
            optional_center_intens = np.sum(image_masked)

            if center == optional_center:
                original_center_intens = optional_center_intens

            if min_intensities > optional_center_intens:
                min_intensities = optional_center_intens
                center_with_min_intensities = optional_center
    return center_with_min_intensities


def get_alive_centers_movements_v2():
    frame_to_alive_pillars_real_loc = get_frame_to_alive_pillars_reposition_v2()
    movements = {}

    all_centers = get_alive_pillar_ids_overall_v3()

    for center_for_key in all_centers:
        movements[center_for_key] = []
        next_center_frame = center_for_key
        for curr_frame in range(len(frame_to_alive_pillars_real_loc) - 1, 0, -1):
            centers_curr_frame = frame_to_alive_pillars_real_loc[curr_frame]
            curr_frame_alive_closest = min(centers_curr_frame,
                                           key=lambda center_curr_frame:
                                           math.hypot(next_center_frame[1] - center_curr_frame[1],
                                                      next_center_frame[0] - center_curr_frame[0]), default=None)
            if curr_frame_alive_closest is None:
                raise Exception("No alive pillars in frame " + str(curr_frame))

            print("found " + str(len(curr_frame_alive_closest)) + " alive centers")

            distance = math.hypot(next_center_frame[1] - curr_frame_alive_closest[1],
                                  next_center_frame[0] - curr_frame_alive_closest[0])

            movement_vector = (next_center_frame[1] - curr_frame_alive_closest[1],
                               next_center_frame[0] - curr_frame_alive_closest[0])

            # TODO: distance?
            if distance < 8:
                angle = (360 + math.degrees(math.atan2(next_center_frame[1] - curr_frame_alive_closest[1],
                                                       next_center_frame[0] - curr_frame_alive_closest[0]))) % 360
                movements[center_for_key].insert(0, {"distance": distance, "angle": angle,
                                                     "movement_vector": movement_vector})
                next_center_frame = curr_frame_alive_closest
            else:
                movements[center_for_key].insert(0, {"distance": 0, "angle": 0, "movement_vector": (0, 0)})

    return movements


def get_list_of_frame_df_pillars_movement_correlation(pillars_movements_dict):
    pillars = list(pillars_movements_dict.keys())
    pillars = [str(p) for p in pillars]

    corr_of_movements_dfs_list = []
    for frame in range(len(list(pillars_movements_dict.values())[0])):
        pillars_movements_corr_df = pd.DataFrame(0.0, index=pillars, columns=pillars)
        for p1, p1_moves in pillars_movements_dict.items():
            p1_vec = p1_moves[frame]['movement_vector']
            p1_magnitude = math.sqrt(pow(p1_vec[0], 2) + pow(p1_vec[1], 2))
            for p2, p2_moves in pillars_movements_dict.items():
                p2_vec = p2_moves[frame]['movement_vector']
                p2_magnitude = math.sqrt(pow(p2_vec[0], 2) + pow(p2_vec[1], 2))

                dot_product = (p1_vec[0] * p2_vec[0]) + (p1_vec[1] * p2_vec[1])
                if p1 == p2:
                    corr = 1
                elif p1_magnitude == 0 and p2_magnitude == 0:
                    corr = None
                elif p1_magnitude == 0 or p2_magnitude == 0:
                    corr = None
                else:
                    corr = dot_product / (p1_magnitude * p2_magnitude)
                pillars_movements_corr_df.loc[str(p1), str(p2)] = corr
        corr_of_movements_dfs_list.append(pillars_movements_corr_df)

    return corr_of_movements_dfs_list


def get_experiment_results_data(file_path, result_values: list) -> list:
    results_file = csv.DictReader(open(file_path))
    all_rows = []
    for row in results_file:
        all_rows.append(row)
    relevant_results_row = all_rows[-1]
    results = []
    for val in result_values:
        results.append(relevant_results_row[val])

    return results


def get_alive_pillar_to_frame_v3():
    frame_to_alive_pillars = get_alive_center_ids_by_frame_v3()
    alive_pillars_to_frame = {}
    for frame, alive_pillars_in_frame in frame_to_alive_pillars.items():
        for alive_pillar in alive_pillars_in_frame:
            if alive_pillar not in alive_pillars_to_frame:
                alive_pillars_to_frame[alive_pillar] = frame

    return alive_pillars_to_frame


def closest_to_point(points, target):
    if len(points) == 0:
        return None
    return min(points, key=lambda point: math.hypot(target[1] - point[1], target[0] - point[0]))
