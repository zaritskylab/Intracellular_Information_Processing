from Pillars.pillars_utils import *
from Pillars.pillars_mask import *
import pickle
from scipy import stats
from Pillars.consts import *


def get_pillar_to_intensities(path):
    """
    Mapping each pillar to its intensity in every frame
    :param path:
    :return:
    """
    pillar_to_intensities_cache_path = Consts.pillar_to_intensities_cache_path
    if Consts.USE_CACHE and os.path.isfile(pillar_to_intensities_cache_path):
        with open(pillar_to_intensities_cache_path, 'rb') as handle:
            pillar2frame_intensity = pickle.load(handle)
            return pillar2frame_intensity

    pillar2last_mask = get_last_img_mask_for_each_pillar(get_all_center_generated_ids())
    pillar2frame_intensity = {}

    images = get_images(path)

    frame_to_alive_pillars_real_locations = get_alive_centers_real_location_by_frame_v2()

    for pillar_item in pillar2last_mask.items():
        pillar_id = pillar_item[0]
        mask = pillar_item[1]
        curr_pillar_intensity = []
        for frame_index, frame in enumerate(images):
            alive_pillars_real_locations_in_curr_frame = frame_to_alive_pillars_real_locations[frame_index]
            if len(alive_pillars_real_locations_in_curr_frame) > 0:
                alive_closest_real_location = min(alive_pillars_real_locations_in_curr_frame,
                                                  key=lambda point: math.hypot(pillar_item[0][1] - point[1],
                                                                            pillar_item[0][0] - point[0]))
                if np.linalg.norm(np.array(alive_closest_real_location) - np.array(pillar_item[0])) < Consts.MAX_DISTANCE_TO_CLOSEST:
                    # Use mask on the new, real actual center location
                    mask = get_mask_for_center(alive_closest_real_location)

            frame_masked = np.where(mask, frame, 0)

            curr_pillar_intensity.append(np.sum(frame_masked))
        pillar2frame_intensity[pillar_id] = curr_pillar_intensity

    if Consts.USE_CACHE:
        with open(pillar_to_intensities_cache_path, 'wb') as handle:
            pickle.dump(pillar2frame_intensity, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pillar2frame_intensity


def get_alive_pillars_to_intensities():
    """
    Mapping only alive pillars to theirs intensity in every frame
    :return:
    """
    if Consts.normalized:
        pillar_intensity_dict = normalized_intensities_by_mean_background_intensity()
    else:
        pillar_intensity_dict = get_pillar_to_intensities(get_images_path())

    alive_pillar_ids = get_alive_pillar_ids_overall_v3()

    alive_pillars_dict = {pillar: pillar_intensity_dict[pillar] for pillar in alive_pillar_ids}

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
    alive_pillars = get_alive_pillar_ids_in_last_frame_v3()
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


def get_cell_avg_intensity():
    p_to_intens = get_alive_pillars_to_intensities()
    frame_to_alive_pillars = get_alive_center_ids_by_frame_v3()
    alive_pillars_to_frame = {}
    for frame, alive_pillars_in_frame in frame_to_alive_pillars.items():
        for alive_pillar in alive_pillars_in_frame:
            if alive_pillar not in alive_pillars_to_frame:
                alive_pillars_to_frame[alive_pillar] = frame

    pillars_avg_intens = []
    for p, intens in p_to_intens.items():
        start_living_frame = alive_pillars_to_frame[p]
        avg_p_intens = np.mean(intens[start_living_frame:])
        pillars_avg_intens.append(avg_p_intens)

    total_avg_intensity = np.mean(pillars_avg_intens)
    return total_avg_intensity
