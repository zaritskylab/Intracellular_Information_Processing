import numpy as np
from tifffile import imsave
from skimage import io
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift


def reposition(vid_path, output_name):
    full_img_stack = io.imread(vid_path)
    n_frames_to_test = 1000
    # subpixel precision
    # Upsample factor 100 = images will be registered to within 1/100th of a pixel.
    # Default is 1 which means no upsampling.
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
        shifted, error_before, diffphase = phase_cross_correlation(prev_image, curr_image, upsample_factor=100,
                                                                   normalization=None)
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

        _, error_after, _ = phase_cross_correlation(prev_image, corrected_image, upsample_factor=100,
                                                    normalization=None)

        img_id2diffs[img_idx]['after_fix'] = error_after

    # x = 1
    # Save full_img_stack
    imsave('../Data/Pillars/FixedImages/' + output_name + '.tif', full_img_stack)
    return full_img_stack

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
