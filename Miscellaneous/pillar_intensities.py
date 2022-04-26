import os
import pickle
import numpy as np



_pillar_to_intensities_path = '../SavedPillarsData/SavedPillarsData_06/NewFixedImage/pillar_to_intensities_cached.pickle'
_normalized = False


def get_pillar_to_intensities(path):
    """
    Mapping each pillar to its intensity in every frame
    :param path:
    :return:
    """
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
            # kernel = np.ones((10, 10), np.uint8) / 100
            # frame_filtered = rank.mean(frame, footprint=kernel)
            frame_masked = np.where(pillar_mask, frame, 0)
            curr_pillar_intensity.append(np.sum(frame_masked))
            # curr_pillar_masked_frames.append(frame_masked)
        # pillar2frames[pillar_id] = curr_pillar_masked_frames
        pillar2frame_intensity[pillar_id] = curr_pillar_intensity

    with open(_pillar_to_intensities_path, 'wb') as handle:
        pickle.dump(pillar2frame_intensity, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pillar2frame_intensity


def get_alive_pillars_to_intensities():
    """
    Mapping only alive pillars to theirs intensity in every frame
    :return:
    """
    if _normalized:
        pillar_intensity_dict = normalized_intensities_by_mean_background_intensity()
    else:
        pillar_intensity_dict = get_pillar_to_intensities(get_images_path())

    alive_pillars = get_alive_pillars_lst()

    alive_pillars_dict = {pillar: pillar_intensity_dict[pillar] for pillar in alive_pillars}

    return alive_pillars_dict


def normalized_intensities_by_max_background_intensity():
    """
    Normalization of pillars intensities by the maximum intensity of the background
    :return:
    """
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
    """
    Normalization of pillars intensities by the mean intensity of the background
    :return:
    """
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


def normalized_intensities_by_zscore():
    """
    Normalization of pillars intensities by z-score
    :return:
    """
    all_pillars = get_pillar_to_intensities(get_images_path())
    all_pillars_int_lst = list(all_pillars.values())
    all_pillars_zscore_int = stats.zscore(all_pillars_int_lst, axis=1)

    for i, pillar_item in enumerate(all_pillars.items()):
        for j, intensity in enumerate(pillar_item[1]):
            all_pillars[pillar_item[0]][j] = all_pillars_zscore_int[i][j]

    return all_pillars
