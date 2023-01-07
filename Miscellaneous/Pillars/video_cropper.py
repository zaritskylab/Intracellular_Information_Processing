from skimage import io
import numpy as np
from tifffile import imsave
import cv2


def crop_video_rect(
        video_path: str,
        top_left: tuple,
        bottom_right: tuple):
    full_img_stack = io.imread(video_path)
    template = np.zeros(full_img_stack.shape, np.uint16)
    template_height = len(template[1])
    template_width = len(template[2])

    bottom_right = (bottom_right[0] + 1, bottom_right[1] + 1)
    partial_image_height = bottom_right[0] - top_left[0]
    partial_image_width = bottom_right[1] - top_left[1]

    for img_idx in range(len(full_img_stack)):
        img = full_img_stack[img_idx]
        partial_img = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        template[img_idx][
        int((template_height - partial_image_height) / 2): int((template_height + partial_image_height) / 2),
        int((template_width - partial_image_width) / 2): int((template_width + partial_image_width) / 2)] = partial_img
    imsave("C:\\Users\\Sarit Hollander\\Desktop\\Study\\MSc\\Research\\Project\\Cell2CellComunicationAnalyzer\\Data\\Pillars\\FixedImages\\Fixed_REF5.3\\new_fixed_37.41.1.tif",
           template)


def crop_video_circle(
        video_path: str,
        circle_center: tuple,
        radius: int):
    full_img_stack = io.imread(video_path)
    template = np.zeros(full_img_stack.shape, np.uint16)

    template_height = len(template[1])
    template_width = len(template[2])

    partial_image_height = radius * 2
    partial_image_width = radius * 2

    mask = np.ones((template_height, template_width), np.uint8)
    mask += 255
    color = 1
    thickness = -1

    cv2.circle(mask, (circle_center[1], circle_center[0]), radius, color, thickness)
    for img_idx in range(len(full_img_stack)):
        img = full_img_stack[img_idx]
        masked_img = np.where(mask, img, 0)
        partial_img = masked_img[circle_center[0] - radius: circle_center[0] + radius,
                      circle_center[1] - radius: circle_center[1] + radius]

        template[img_idx][
        int((template_height - partial_image_height) / 2): int((template_height + partial_image_height) / 2),
        int((template_width - partial_image_width) / 2): int((template_width + partial_image_width) / 2)] = partial_img

    imsave(
        "C:\\Users\\Sarit Hollander\\Desktop\\Study\\MSc\\Research\\Project\\Cell2CellComunicationAnalyzer\\Data\\Pillars\\9.4\\New-11.2-Airyscan Processing-79 MEF9.4.tif",
        template)


crop_video_rect(
    'C:\\Users\\Sarit Hollander\\Desktop\\Study\\MSc\\Research\\Project\\Cell2CellComunicationAnalyzer\\Data\\Pillars\\FixedImages\\Fixed_REF5.3\\new_fixed_37.41.1.tif',
    (100, 0),
    (500, 280))

# crop_video_circle(
#     'C:\\Users\\Sarit Hollander\\Desktop\\Study\\MSc\\Research\\Project\\Cell2CellComunicationAnalyzer\\Data\\Pillars\\9.4\\New-11-Airyscan Processing-79 MEF9.4.tif',
#     (397, 744),
#     185)
