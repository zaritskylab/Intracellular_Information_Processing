from Miscellaneous.pillar_intensities import *
from Miscellaneous.pillar_neighbors import *
from Miscellaneous.analyzer import *
from Miscellaneous.pillars_mask import *
from Miscellaneous.consts import *
from Miscellaneous.visualization import *

if __name__ == '__main__':
    # images = get_images(get_images_path())
    # with open('../SavedPillarsData/SavedPillarsData_06/NewFixedImage/last_image_06.npy', 'wb') as f:
    #     np.save(f, images[-1])
    # masks_path = PATH_MASKS_VIDEO_01_15_35
    # build_pillars_mask(
    #     masks_path=masks_path,
    #     logic_centers=True
    # )
    # show_last_image_masked(masks_path)
    correlation_plot(only_alive=True,
                     neighbors_str='all',
                     alive_correlation_type='symmetric')
    correlation_histogram(get_all_pillars_correlation())
    mean_corr = correlation_histogram(alive_pillars_asymmetric_correlation())
    plot_pillar_time_series()
    means = []
    rand = []
    mean_original_nbrs = neighbors_correlation_histogram(alive_pillars_asymmetric_correlation(),
                                                         get_alive_pillars_to_alive_neighbors(),
                                                         symmetric_corr=False)
    for i in range(2):
        mean_random_nbrs = neighbors_correlation_histogram(alive_pillars_asymmetric_correlation(),
                                                           get_random_neighbors(),
                                                           symmetric_corr=False)
        means.append(mean_random_nbrs)
        rand.append('random' + str(i + 1))
    means.append(mean_original_nbrs)
    rand.append('original')
    fig, ax = plt.subplots()
    ax.scatter(rand, means)
    plt.ylabel('Average Correlation')
    plt.xticks(rotation=45)
    plt.show()
