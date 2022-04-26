import os
import pickle


_correlation_alive_normalized_path = '../SavedPillarsData/SavedPillarsData_06/NewFixedImage/alive_pillar_correlation_normalized_cached.pickle'
_correlation_alive_not_normalized_path = '../SavedPillarsData/SavedPillarsData_06/NewFixedImage/alive_pillar_correlation_cached.pickle'
_all_pillars_correlation_normalized_path = '../SavedPillarsData/SavedPillarsData_06/NewFixedImage/all_pillar_correlation_normalized_cached.pickle'
_all_pillars_correlation_not_normalized_path = '../SavedPillarsData/SavedPillarsData_06/NewFixedImage/all_pillar_correlation_cached.pickle'
_normalized = False


def get_correlations_between_neighboring_pillars(pillar_to_pillars_dict):
    """
    Listing all correlations between pillar and its neighbors
    :param pillar_to_pillars_dict:
    :return:
    """
    all_corr = get_all_pillars_correlation()

    correlations = []
    for pillar, nbrs in pillar_to_pillars_dict.items():
        for n in nbrs:
            correlations.append(all_corr[str(pillar)][str(n)])

    return correlations


def get_alive_pillars_correlation():
    """
    Create dataframe of correlation between alive pillars only
    :return:
    """
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
    """
    Get the cached correlation of alive pillars
    :return:
    """
    if _normalized:
        path = _correlation_alive_normalized_path
    else:
        path = _correlation_alive_not_normalized_path

    return path


def get_all_pillars_correlation():
    """
    Create dataframe of correlation between all pillars
    :return:
    """
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
    """
    Get the cached correlation of all pillars
    :return:
    """
    if _normalized:
        path = _all_pillars_correlation_normalized_path
    else:
        path = _all_pillars_correlation_not_normalized_path

    return path


def alive_pillars_symmetric_correlation():
    """
    Create dataframe of alive pillars correlation as the correlation according to
    maximum_frame(pillar a start to live, pillar b start to live) -> correlation(a, b) == correlation(b, a)
    :return:
    """
    frame_to_pillars = get_frame_to_alive_pillars()
    pillar_to_frame = {}
    for k, v_lst in frame_to_pillars.items():
        for item in v_lst:
            if item not in pillar_to_frame:
                pillar_to_frame[item] = k

    alive_pillars_intens = get_alive_pillars_to_intensities()
    alive_pillars = list(alive_pillars_intens.keys())
    alive_pillars_str = [str(p) for p in alive_pillars]
    pillars_corr = pd.DataFrame(0, index=alive_pillars_str, columns=alive_pillars_str)

    # Symmetric correlation - calc correlation of 2 pillars start from the frame they are both alive: maxFrame(A, B)
    for p1 in alive_pillars:
        p1_living_frame = pillar_to_frame[p1]
        for p2 in alive_pillars:
            p2_living_frame = pillar_to_frame[p2]
            both_alive_frame = max(p1_living_frame, p2_living_frame)
            p1_relevant_intens = alive_pillars_intens[p1][both_alive_frame - 1:]
            p2_relevant_intens = alive_pillars_intens[p2][both_alive_frame - 1:]
            pillars_corr.loc[str(p2), str(p1)] = pearsonr(p1_relevant_intens, p2_relevant_intens)[0]

    return pillars_corr


def alive_pillars_asymmetric_correlation():
    """
    Create dataframe of alive pillars correlation. correlation of pillar first frame of living with all other pillars.
    -> correlation(a, b) != correlation(b, a)
    :return:
    """
    frame_to_pillars = get_frame_to_alive_pillars()
    pillar_to_frame = {}
    for k, v_lst in frame_to_pillars.items():
        for item in v_lst:
            if item not in pillar_to_frame:
                pillar_to_frame[item] = k

    alive_pillars_intens = get_alive_pillars_to_intensities()
    alive_pillars = list(alive_pillars_intens.keys())
    alive_pillars_str = [str(p) for p in alive_pillars]
    pillars_corr = pd.DataFrame(0, index=alive_pillars_str, columns=alive_pillars_str)

    # Asymmetric correlation - calc correlation with every pillar start from the frame p1 is alive
    for p1 in alive_pillars:
        alive_from_frame = pillar_to_frame[p1]
        p1_relevant_intens = alive_pillars_intens[p1][alive_from_frame - 1:]
        for p2 in alive_pillars:
            p2_relevant_intens = alive_pillars_intens[p2][alive_from_frame - 1:]
            pillars_corr.loc[str(p2), str(p1)] = pearsonr(p1_relevant_intens, p2_relevant_intens)[0]

    return pillars_corr
