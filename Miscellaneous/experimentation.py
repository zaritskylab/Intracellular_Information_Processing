import os
import numpy as np
import skimage.io
from skimage import io
import matplotlib.pyplot as plt
import seaborn as sns
from Miscellaneous.consts import *
from Miscellaneous.global_parameters import *
from tifffile import imsave
import sys
import pandas as pd
from skimage import data, io
from skimage.registration import phase_cross_correlation
# from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
from image_registration import chi2_shift

vid_path = PILLARS + VIDEO_01_TIF_PATH
full_img_stack = io.imread(vid_path)
n_frames_to_test = 1000
# subpixel precision
#Upsample factor 100 = images will be registered to within 1/100th of a pixel.
#Default is 1 which means no upsampling.
full_img_stack = full_img_stack[:n_frames_to_test, :, :]
# full_img_stack = full_img_stack[1:n_frames_to_test, :, :]
n_frames, im_h, im_w = full_img_stack.shape
prev_image, curr_image = None, full_img_stack[0]

img_id2diffs = {}

for img_idx in range(len(full_img_stack) - 1):
    if prev_image is None:
        prev_image = curr_image.copy()
        continue
    prev_image, curr_image = curr_image.copy(), full_img_stack[img_idx].copy()
    _, error_before, _ = phase_cross_correlation(prev_image, curr_image, upsample_factor=100, normalization=None)
    img_id2diffs[img_idx] = {}
    img_id2diffs[img_idx]['original'] = error_before


prev_image, curr_image = None, full_img_stack[0]
for img_idx in range(len(full_img_stack) - 1):
    if prev_image is None:
        prev_image = curr_image.copy()
        continue
    # Get curr and prev images
    prev_image, curr_image = curr_image.copy(), full_img_stack[img_idx].copy()
    assert not np.array_equal(prev_image, curr_image)
    shifted, error_before, diffphase = phase_cross_correlation(prev_image, curr_image, upsample_factor=100, normalization=None)
    img_id2diffs[img_idx]['before_fix'] = error_before

    # noise = 0.1
    # xoff, yoff, exoff, eyoff = chi2_shift(prev_image, curr_image, noise,
    #                                       return_error=True, upsample_factor='auto')
    # corrected_image = shift(curr_image, shift=(xoff, yoff), mode='constant')
    # print("Pixels shifted by: ", xoff, yoff)
    # print(img_idx, f": Detected subpixel offset (y, x): {shifted}")
    corrected_image = shift(curr_image, shift=(shifted[0], shifted[1]), mode='grid-constant')
    curr_image = corrected_image.copy()
    full_img_stack[img_idx] = corrected_image.copy()

    _, error_after, _ = phase_cross_correlation(prev_image, corrected_image, upsample_factor=100, normalization=None)

    img_id2diffs[img_idx]['after_fix'] = error_after

# x = 1
# Save full_img_stack
imsave('../Data/Pillars/FixedImages/new_fixed_01' + '.tif', full_img_stack)

# fig = plt.figure(figsize=(10, 10))
# ax1 = fig.add_subplot(2,2,1)
# ax1.imshow(prev_image, cmap='gray')
# ax1.title.set_text('Input Image')
# ax2 = fig.add_subplot(2,2,2)
# ax2.imshow(curr_image, cmap='gray')
# ax2.title.set_text('Offset image')
# ax3 = fig.add_subplot(2,2,3)
# ax3.imshow(corrected_image, cmap='gray')
# ax3.title.set_text('Corrected')
# plt.show()







# vid_path = PILLARS + VIDEO_20_TIF_PATH
# n_frames_to_test = 1000
# # full_img_stack = io.imread('../Data/Pillars/FixedImages/fixed_01.tif')
# full_img_stack = io.imread(vid_path)
#
# full_img_stack = full_img_stack[:n_frames_to_test, :, :]
# n_frames, im_h, im_w = full_img_stack.shape
#
# prev_image, curr_image = None, full_img_stack[0]
# min_shift, max_shift, step_shift = -5, 5, 1 # must be square (i.e. -nXn, with m step size)
# no_displacement_index = (int((max_shift+abs(min_shift)) / 2), int((max_shift+abs(min_shift)) / 2))
# mx, my = np.meshgrid(np.arange(min_shift, max_shift, step_shift),
#                      np.arange(min_shift, max_shift, step_shift))
#
# nx = mx.ravel()
# ny = my.ravel()
# out_score_per_frame = np.zeros(tuple([n_frames] + list(nx.shape)), dtype=np.float32)
#
# for img_idx in range(len(full_img_stack) - 1):
#     if prev_image is None:
#         prev_image = curr_image.copy()
#         continue
#     # Get curr and prev images
#     prev_image, curr_image = curr_image.copy(), full_img_stack[img_idx].copy()
#     assert not np.array_equal(prev_image, curr_image)
#
#     # check current frame in relative to the previous one, and fix it according to the min error
#     min_shift_index = 0
#     fixed_frame2 = curr_image
#     curr_min_error = sys.maxsize
#     for shift_idx in range(len(nx)):
#         shifted_current = np.roll(np.roll(curr_image, nx[shift_idx], axis=1), ny[shift_idx], axis=0)
#         curr_shift_error = np.mean(np.square(prev_image - shifted_current))
#         if curr_shift_error < curr_min_error:
#             curr_min_error = curr_shift_error
#             fixed_frame2 = shifted_current.copy()
#             min_shift_index = shift_idx
#         out_score_per_frame[img_idx][shift_idx] = curr_shift_error
#
#     min_shift_error_index = np.argmin(out_score_per_frame[img_idx])
#     shift_x = nx[min_shift_error_index]
#     shift_y = ny[min_shift_error_index]
#
#     # fixed_frame = np.roll(np.roll(curr_image, shift_x, axis=1), shift_y, axis=0)
#     curr_image = fixed_frame2.copy()
#     full_img_stack[img_idx] = fixed_frame2.copy()
#
#     # print(f"Finished processing frame #{i+1}/{n_frames}")
#
# # Save full_img_stack
# # imsave('../Data/Pillars/FixedImages/fixed_01' + '.tif', full_img_stack)
#
#
# # This section is printing the displacement error -
# # notice that in this printing each frame is in relative to the previous already(!) displaced frame
# # use it just to help you in debugging, testing and validating
# fixed_image_stack = np.zeros_like(full_img_stack)
#
# for img_idx in range(1, len(out_score_per_frame)):
#     # get the involved frame and the displacements scores between them
#     prev_image, curr_image, displacement_scores = full_img_stack[img_idx - 1], full_img_stack[img_idx], out_score_per_frame[img_idx]
#     # normalize displacement scores
#     displacement_scores = np.divide(displacement_scores, displacement_scores.max(), out=np.zeros_like(displacement_scores), where=displacement_scores!=0)
#     displacement_scores_reshaped = displacement_scores.reshape(mx.shape)
#     min_shift_error_index = np.unravel_index(np.argmin(displacement_scores_reshaped), displacement_scores_reshaped.shape)
#     # flag for visualization only
#     displaced = False
#     if min_shift_error_index[0] != no_displacement_index[0] or min_shift_error_index[1] != no_displacement_index[1]:
#         displaced = True
#         movement_in_x, movement_in_y = mx[0, min_shift_error_index[0]], my[min_shift_error_index[1], 0]
#         print(f"displacement of x={movement_in_x}, y={movement_in_y} found between frame #{img_idx - 1} and frame #{img_idx}")
#         # i_min = np.argmin(displacement_scores)
#         # fixed_frame = np.roll(np.roll(curr_image, nx[i_min], axis=1), ny[i_min], axis=0)
#     # else:
#         # fixed_frame = curr_image.copy()
#     # fixed_image_stack[i, :, :] = fixed_frame
#
#     # visualizing using matplotlib and seaborn
#     # fig, axis = plt.subplots(1, 3, figsize=(21,7))
#     # axis[0].imshow(prev_image)
#     # axis[1].imshow(curr_image)
#     # sns.heatmap(displacement_scores_reshaped, ax=axis[2], cbar=False,
#     #             annot=True, fmt='1.3f', cmap='viridis')
#     #
#     # axis[2].set_xticklabels(mx[0, :])
#     # axis[2].set_yticklabels(my[:, 0])
#
#     # fig.suptitle(f'{i+1}/{n_frames}' + "\nDISPLACED" if displaced else '')
#     # plt.show()
