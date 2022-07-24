import math
import pickle
from skimage.filters import rank
from skimage.filters.thresholding import threshold_mean
from collections import deque as queue
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np
from datetime import datetime
import numpy.ma as ma
from Pillars.consts import *


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


def get_image_whiten(build_image=False):
    if build_image:
        img = create_image_by_max_value()
    else:
        img = get_last_image()

    img = np.interp(img, (img.min(), img.max()), (0, 255)).astype('uint8')
    # img[(img >= 25) & (img <= 50)] = 50
    # # img[(img >= 21) & (img <= 50)] = 50
    # img[(img >= 81) & (img <= 100)] = 100
    # img[(img >= 101) & (img <= 150)] = 150
    # img[(img >= 151) & (img <= 200)] = 200

    img[(img < Consts.pixel_to_whiten)] = 0
    img[(img >= Consts.pixel_to_whiten)] = 255

    return img


def find_centers_with_logic():
    if Consts.USE_CACHE and os.path.isfile(Consts.centers_cache_path):
        with open(Consts.centers_cache_path, 'rb') as handle:
            centers = pickle.load(handle)
            return centers

    last_img = get_last_image()
    alive_centers = get_alive_centers(last_img)
    centers = generate_centers_from_alive_centers(alive_centers, len(last_img))

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
    # TODO: if otsu not work properly
    ret = 500
    img[img < ret] = 0
    img[img > 0] = 1
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()
    return img


def get_alive_centers(img):
    # cache
    if Consts.USE_CACHE and os.path.isfile(Consts.alive_centers_cache_path):
        with open(Consts.alive_centers_cache_path, 'rb') as handle:
            alive_centers = pickle.load(handle)
            return alive_centers

    img = get_image_by_threshold(img)

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
                    col += Consts.CIRCLE_SEARCH_JUMP_SIZE
                    # col += 1
            else:
                col += 1

    if Consts.USE_CACHE:
        with open(Consts.alive_centers_cache_path, 'wb') as handle:
            pickle.dump(alive_centers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return alive_centers


def generate_centers_from_alive_centers(alive_centers, matrix_size):
    return generate_centers_and_rules_from_alive_centers(alive_centers, matrix_size)[0]


def generate_centers_and_rules_from_alive_centers(alive_centers, matrix_size):
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

    if -5 <= abs(rule1[0]) - abs(rule2[0]) <= 5 and -5 <= abs(rule1[1]) - abs(rule2[1]) <= 5:
        points.remove(closest2)
        closest2 = min(points,
                       key=lambda point: math.hypot(closest_to_middle[1] - point[1], closest_to_middle[0] - point[0]))
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

    # Replaced live generated centers with their actual positions
    generated_location2original_pillar_loc = {}
    for actual_alive_center in alive_centers:
        closest_generated_center = min(centers_in_range,
            key=lambda point: math.hypot(actual_alive_center[1] - point[1], actual_alive_center[0] - point[0]))

        centers_in_range.remove(closest_generated_center)
        centers_in_range.append(actual_alive_center)
        generated_location2original_pillar_loc[closest_generated_center] = actual_alive_center
    return centers_in_range, rule1, rule2, generated_location2original_pillar_loc


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

    right_is_valid = False
    left_is_valid = False
    up_is_valid = False
    down_is_valid = False

    for safety_distance in range(Consts.CIRCLE_OUTSIDE_VALIDATE_SEARCH_LENGTH,
                                 Consts.CIRCLE_OUTSIDE_VALIDATE_SEARCH_LENGTH + 6):
        if img[row][col + safety_distance] == 1:
            right_is_valid = True
        if img[row][col - safety_distance] == 1:
            left_is_valid = True
        if img[row - safety_distance][col] == 1:
            up_is_valid = True
        if img[row + safety_distance][col] == 1:
            down_is_valid = True

    return down_is_valid and up_is_valid and right_is_valid and left_is_valid


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


def get_images_path():
    if Consts.fixed:
        return Consts.fixed_images_path
    else:
        return Consts.images_path


def get_frame_to_alive_pillars(pillar2mask):
    if Consts.USE_CACHE and os.path.isfile(Consts.frame2pillar_cache_path):
        with open(Consts.frame2pillar_cache_path, 'rb') as handle:
            frame_to_alive_pillars = pickle.load(handle)
            return frame_to_alive_pillars
    frame_to_alive_pillars = {}
    # frame_to_background_pillars = {}
    images = get_images(get_images_path())
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

    if Consts.USE_CACHE:
        with open(Consts.frame2pillar_cache_path, 'wb') as handle:
            pickle.dump(frame_to_alive_pillars, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return frame_to_alive_pillars


def get_alive_pillars(pillar2mask):
    if Consts.inner_cell:
        return _get_fully_alive_pillars_lst(pillar2mask)
    else:
        return _get_alive_pillars_lst(pillar2mask)


def _get_alive_pillars_lst(pillar2mask):
    frame_to_pillars = get_frame_to_alive_pillars(pillar2mask)
    any_time_live_pillars = set()
    for pillars in frame_to_pillars.values():
        any_time_live_pillars.update(pillars)

    return list(any_time_live_pillars)


def _get_fully_alive_pillars_lst(pillar2mask_dict):
    p_mask = list(pillar2mask_dict.values())[0]
    perfect_circle = p_mask.sum() * Consts.percentage_from_perfect_circle_mask

    img = get_image_whiten(build_image=Consts.build_image)

    fully_alive_pillars_lst = []
    for p, mask in pillar2mask_dict.items():
        frame_masked = np.where(mask, img, 0)
        # plt.imshow(frame_masked, cmap=plt.cm.gray)
        # plt.show()
        if (np.sum(frame_masked) / perfect_circle) * 100 > Consts.consider_as_full_circle_percentage:
            fully_alive_pillars_lst.append(p)
            # plt.imshow(frame_masked, cmap=plt.cm.gray)
            # plt.show()

    return fully_alive_pillars_lst


def create_image_by_max_value():
    images = get_images(get_images_path())
    new_img = np.max(np.array(images), axis=0)

    return new_img

# TODO: delete? needed?
# def get_all_pillars_in_radius_df(gc_df, pillar2mask_dict, radius):
#     img_size = get_image_size()
#     center = (img_size[0] / 2, img_size[1] / 2)
#
#     fully_pillars = get_fully_alive_pillars_lst(pillar2mask_dict)
#     pillars_to_drop = []
#     for p in gc_df.columns:
#         if math.dist(eval(p), center) > radius or eval(p) not in fully_pillars:
#             pillars_to_drop.append(p)
#
#     gc_df = gc_df.drop(pillars_to_drop, axis=0)
#     gc_df = gc_df.drop(pillars_to_drop, axis=1)
#
#     return gc_df


# def alive2alive_alive2back_edge2l1_t_test():
#     global alive_pillars_unique_corr
#     alive_pillars = get_alive_pillars_to_intensities()
#     alive_pillars_first_frame = get_frame_to_alive_pillars()[1]
#     all_pillars = get_pillar_to_intensities(get_images_path())
#     background_pillars_intensities = {pillar: all_pillars[pillar] for pillar in all_pillars.keys() if
#                                       pillar not in alive_pillars}
#     alive_pillars_first_frame_intens = {pillar: all_pillars[pillar] for pillar in alive_pillars.keys() if
#                                         pillar in alive_pillars_first_frame}
#     alive_pillars_df = pd.DataFrame({str(k): v for k, v in alive_pillars_first_frame_intens.items()})
#     alive_pillars_corr = alive_pillars_df.corr()
#     pillars = {}
#     if _normalized:
#         all_pillars_to_intens = normalized_intensities_by_mean_background_intensity()
#     else:
#         all_pillars_to_intens = get_pillar_to_intensities(get_images_path())
#     for p in list(background_pillars_intensities.keys()):
#         pillars[p] = all_pillars_to_intens[p]
#     for p in list(alive_pillars_first_frame_intens.keys()):
#         pillars[p] = all_pillars_to_intens[p]
#     pillar_intensity_df = pd.DataFrame({str(k): v for k, v in pillars.items()})
#     pillars_corr = pillar_intensity_df.corr()
#     alive_back_corr = []
#     for p in pillars_corr.columns:
#         for p2 in pillars_corr:
#             if (eval(p) in background_pillars_intensities and eval(p2) in background_pillars_intensities) or (
#                     eval(p) in alive_pillars_first_frame and eval(p2) in alive_pillars_first_frame):
#                 continue
#             else:
#                 alive_back_corr.append(pillars_corr[p][p2])
#     alive_pillars_unique_corr = alive_pillars_corr.stack().loc[
#         lambda x: x.index.get_level_values(0) < x.index.get_level_values(1)]
#     alive_pillars_unique_corr.index = alive_pillars_unique_corr.index.map('_'.join)
#     alive_pillars_unique_corr = alive_pillars_unique_corr.to_frame().T
#     alive_pillars_unique_corr_lst = alive_pillars_unique_corr.transpose()[0].tolist()
#     edge_to_l1, _, _ = get_alive_pillars_in_edges_to_l1_neighbors()
#     corr_alive_edge_to_back_l1 = get_correlations_between_neighboring_pillars(edge_to_l1)
#
#     t_alive2alive_and_alive2back, p_alive2alive_and_alive2back = ttest_ind(alive_pillars_unique_corr_lst,
#                                                                            alive_back_corr, equal_var=False)
#     t_alive2alive_and_edge2l1, p_alive2alive_and_edge2l1 = ttest_ind(alive_pillars_unique_corr_lst,
#                                                                      corr_alive_edge_to_back_l1,
#                                                                      equal_var=False)
#     t_alive2back_and_edge2l1, p_alive2back_and_edge2l1 = ttest_ind(alive_back_corr, corr_alive_edge_to_back_l1,
#                                                                    equal_var=False)
#     return (t_alive2alive_and_alive2back, p_alive2alive_and_alive2back), \
#            (t_alive2alive_and_edge2l1, p_alive2alive_and_edge2l1), (t_alive2back_and_edge2l1, p_alive2back_and_edge2l1)

#
