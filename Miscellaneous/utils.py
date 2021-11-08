import json
import os
from typing import *
import numpy as np
import pandas as pd
from numpy import median
from scipy.spatial import Voronoi
from scipy.stats import linregress, zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics.pairwise import euclidean_distances as euc_dis
from sklearn.model_selection import train_test_split
from Miscellaneous.global_parameters import *
from Miscellaneous.consts import *
from Miscellaneous.global_parameters import *
from matplotlib.lines import Line2D
import random


def get_all_unique_treatments(meta_data_file_full_path: str = None):
    """

    :param meta_data_file_full_path:
    :return:
    """
    # todo: add documentation
    if meta_data_file_full_path is None:
        meta_data_file_full_path = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Data',
                                                                                 'Experiments_XYT_CSV',
                                                                                 'ExperimentsMetaData.csv'])
    meta_data_df = pd.read_csv(meta_data_file_full_path)
    all_treatments = meta_data_df['Treatment'].values
    unique_treatments = np.unique(all_treatments)
    return unique_treatments


def load_dict_from_json(path: str):
    """

    :param path:
    :return:
    """
    # todo: add documentation
    dict_to_load = None
    try:
        with open(path, 'r') as f:
            dict_to_load = json.load(f)
    except FileNotFoundError as e:
        print(f'problem loading the configuration file at:\n{e}')
    return dict_to_load


def write_dict_as_json(path: str, dict_to_write: dict):
    """

    :param path:
    :param dict_to_write:
    :return:
    """
    # todo: add documentation
    try:
        with open(path, 'w') as f:
            json.dump(dict_to_write, f)
    except OSError as e:
        print(f'problem saving the configuration file at:\n{e}')


def get_all_paths_csv_files_in_dir(dir_path: str) -> Tuple[List, List]:
    """
    returns full paths and file names with no file type
    :param dir_path:
    :return:
    """
    if dir_path is None:
        raise ValueError('dir path cant be none')
    full_paths = list(map(lambda x: os.sep.join([dir_path, x]), filter(lambda x: x.endswith('.csv') and
                                                                                 'ds_store' not in x.lower(),
                                                                       os.listdir(dir_path))))
    only_exp_names = list(map(lambda x: x.replace('.csv', ''), filter(lambda x: x.endswith('.csv'),
                                                                      os.listdir(dir_path))))
    return full_paths, only_exp_names


def get_exp_treatment_type_and_temporal_resolution(exp_file_name: str,
                                                   meta_data_file_full_path: str = None,
                                                   compressed_flag: bool = False) -> Tuple[str, int]:
    """
    returns an experiment treatment type and temporal resolution (i.e., interval between frames)
    :param exp_file_name:
    :param meta_data_file_full_path:
    :return:
    """
    if meta_data_file_full_path is None:
        if compressed_flag:
            meta_data_file_full_path = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Data',
                                                                                     'Experiments_XYT_CSV',
                                                                                     'Compressed_ExperimentsMetaData.csv'])
        else:
            meta_data_file_full_path = os.sep.join(os.getcwd().split(os.sep)[:-1] + ['Data',
                                                                                     'Experiments_XYT_CSV',
                                                                                     'ExperimentsMetaData.csv'])

    meta_data_file = pd.read_csv(meta_data_file_full_path)
    exp_meta_data = meta_data_file[meta_data_file['File Name'] == exp_file_name]
    exp_treatment, exp_time_res = exp_meta_data['Treatment'].values[0], \
                                  int(exp_meta_data['Time Interval (min)'].values[0])

    return exp_treatment, exp_time_res


def read_experiment_cell_xy_and_death_times(exp_full_path: str) -> Tuple[np.array, np.array]:
    """
    reads an experiment's csv file, returns the cell loci and times of deaths
    :param exp_full_path:
    :return:
    """
    full_df = pd.read_csv(exp_full_path)
    cells_loci = full_df.loc[:, ['cell_x', 'cell_y']].values
    cells_times_of_death = full_df.loc[:, ['death_time']].values
    return cells_loci, cells_times_of_death


def kl_divergence(p: np.array, q: np.asarray) -> np.ndarray:
    """
    returns the kl divergence score for p and q
    :param p: np array - signal from distribution a
    :param q: np array - signal from distribution b
    :return: float
    """
    if type(p) is not type(np.array) or type(q) is not type(np.array):
        p = np.array(p)
        q = np.array(q)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def get_euclidean_distance_between_cells_in_pixels(cell1_xy: Tuple, cell2_xy: Tuple) -> float:
    """
    returns the real distance
    :param cell1_xy:
    :param cell2_xy:
    :return:
    """
    cell1_x, cell1_y = cell1_xy
    cell2_x, cell2_y = cell2_xy
    return ((cell1_x - cell2_x) ** 2 + (cell1_y - cell2_y) ** 2) ** .5


def get_linear_regression_line_between_two_signals(x: np.array, y: np.array) -> Tuple[np.array, np.array]:
    """
    calculates the linear regression line between two signals and returns the x and y of the new regression line
    :param x:
    :param y:
    :return:
    """
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    x_new = np.linspace(0, x.max(), len(y))
    y_new = model.predict(x_new[:, np.newaxis])
    return x_new, y_new


def get_cells_neighbors(XY, threshold_dist: Union[int, float] = None) -> Tuple[List[int], List[int], List[int]]:
    """
    returns 3 levels of topological neighbors for each cell.
    if threshold_dist is not None, a distance constraint is employed on the neighbors to prevent neighbors that are
    very far away.
    :param XY:
    :param threshold_dist:
    :return:
    """
    vor = Voronoi(XY)
    neighbors = vor.ridge_points
    neighbors_list = []
    neighbors_list2 = []
    neighbors_list3 = []
    for i in range(len(XY)):
        neighbors_list.append([])
        neighbors_list2.append([])
        neighbors_list3.append([])
    for x in neighbors:
        if threshold_dist is None:
            neighbors_list[x[0]].append(x[1])
            neighbors_list[x[1]].append(x[0])
        else:
            if get_euclidean_distance_between_cells_in_pixels(XY[x[0]], XY[x[1]]) <= threshold_dist:
                neighbors_list[x[0]].append(x[1])
                neighbors_list[x[1]].append(x[0])

    for i in range(len(XY)):
        for j in neighbors_list[i]:
            # add the neighbors of a neighbor (j), exclude yourself (i) and
            # remove duplicates and cells that are in 1st neighbor level lists to i
            combined_neighbors = set(neighbors_list2[i] + neighbors_list[j])
            combined_neighbors.remove(i)
            clean_neighbors = combined_neighbors.copy()
            for neighbor_idx in combined_neighbors: clean_neighbors.remove(neighbor_idx) if neighbor_idx in \
                                                                                            neighbors_list[
                                                                                                i] else None
            neighbors_list2[i] = list(clean_neighbors)
    for i in range(len(XY)):
        for j in neighbors_list2[i]:
            # add the 2nd level neighbors of a neighbor (j), exclude yourself (i) and
            # remove duplicates and cells that are in 2nd and 1st neighbor level lists to i
            combined_neighbors = set(neighbors_list3[i] + neighbors_list2[j])
            combined_neighbors.remove(i)
            clean_neighbors = combined_neighbors.copy()
            for neighbor_idx in combined_neighbors: clean_neighbors.remove(neighbor_idx) if neighbor_idx in \
                                                                                            neighbors_list2[
                                                                                                i] else None
            for neighbor_idx in combined_neighbors: clean_neighbors.remove(neighbor_idx) if neighbor_idx in \
                                                                                            neighbors_list[
                                                                                                i] else None
            neighbors_list3[i] = list(clean_neighbors)

    return neighbors_list, neighbors_list2, neighbors_list3


def calc_fraction_from_candidates(dead_cells_at_time_indices: np.array, candidates_indices: np.array) -> float:
    """
    calculates the fraction of cells that died out of the candidates of the specific type of death.
    if there are no candidates, verifies that there are no dead cells and returns 0.
    :param dead_cells_at_time_indices:
    :param candidates_indices:
    :return:
    """
    if len(dead_cells_at_time_indices) > len(candidates_indices):
        raise ValueError('candidates number cant be less than dead cells')
    if len(candidates_indices) == 0:
        return 0
    return len(dead_cells_at_time_indices) / len(candidates_indices)


def calc_mask_from_indices(empty_mask: np.array, indices: Union[np.array, List], val_to_mask: bool = True) -> np.array:
    """
    for each idx in indices list, change the value @idx in the empty mask to val_to_mask
    :param empty_mask:
    :param indices:
    :param val_to_mask:
    :return:
    """
    for idx in indices:
        empty_mask[idx] = val_to_mask
    return empty_mask


def get_cells_not_neighboring_dead_cells(dead_cells_mask, neighbors, neighbors_list2, neighbors_list3, xy=None,
                                         threshold=200):
    """
    returns two groups of cells. 1st is all alive cells that are neighbors of dead cells' neighbors.
    2nd is the rest of the alive cells which are not direct topological neighbors of any dead cells.
    :param dead_cells_mask:
    :param neighbors:
    :param neighbors_list2:
    :param xy:
    :param threshold:
    :return:
    """
    all_alive_cells = np.array(dead_cells_mask - 1, dtype=bool)
    # get all cells neighboring dead cells (propagation candidates)
    around_dead_cells = np.zeros(dead_cells_mask.shape, dtype=bool)
    for cell_idx, is_dead in enumerate(dead_cells_mask):
        if is_dead:
            curr_neighbors = neighbors[cell_idx]
            for neighbor_idx in curr_neighbors:
                if xy is not None:
                    dist = get_euclidean_distance_between_cells_in_pixels(cell1_xy=xy[cell_idx],
                                                                          cell2_xy=xy[neighbor_idx])
                    around_dead_cells[neighbor_idx] = (True) * (dist < threshold)

    # get complementary & alive cells that are not near dead cells
    all_not_around_dead_cells_and_alive = np.array(around_dead_cells - 1, dtype=bool) * all_alive_cells
    # divide to two groups at different "neighboring" distances
    not_around_dead_cells_1 = np.zeros(dead_cells_mask.shape, dtype=bool)
    not_around_dead_cells_2 = np.zeros(dead_cells_mask.shape, dtype=bool)
    for cell_idx, is_cell_not_adjacent_to_death in enumerate(all_not_around_dead_cells_and_alive):
        if is_cell_not_adjacent_to_death:
            alive_cell_2nd_lvl_neighbors = neighbors_list2[cell_idx]
            for adjacent_neighbor_idx in alive_cell_2nd_lvl_neighbors:
                # if the cell(cell_idx) is a 2nd lvl neighbor to a dead cell
                if dead_cells_mask[adjacent_neighbor_idx]:
                    not_around_dead_cells_1[cell_idx] = True
                    break

            alive_cell_3rd_lvl_neighbors = neighbors_list3[cell_idx]
            for adjacent_neighbor_idx in alive_cell_3rd_lvl_neighbors:
                # if the cell(cell_idx) is a 3rd lvl neighbor to a dead cell
                # and not a 2nd lvl neighbor to a dead cell
                if dead_cells_mask[adjacent_neighbor_idx]:
                    not_around_dead_cells_2[cell_idx] = True * (not not_around_dead_cells_1[cell_idx])
                    break

    # not_around_dead_cells_2 = np.array(not_around_dead_cells_1-1, dtype=bool) * all_alive_cells
    return not_around_dead_cells_1, not_around_dead_cells_2


def calc_distance_metric_between_signals(y_true: np.array, y_pred: np.array, metric: str = 'rmse'):
    """
    calculates a distance metric between two np.array values, enforces equal lengths of arrays.
    supports lists, tuples and any iterables as well.
    9/08/2021 - supports the following metrics: rmse, mse, kl-divergence, euclidean distance (returns distances mean).
    :param y_true: np.array
    :param y_pred: np.array
    :param metric: str, metric to calculate
    :return: float, the metric calculation result.
    """
    assert len(y_true) == len(y_pred), f'y_true and y_pred must have equal lengths but y_true length = {len(y_true)}' \
                                       f' and y_pred length = {len(y_pred)}'
    if metric == 'rmse':
        return mse(y_true=y_true, y_pred=y_pred, squared=False)
    if metric == 'mse':
        return mse(y_true=y_true, y_pred=y_pred, squared=True)
    if metric == 'kl_divergence':
        return kl_divergence(y_true, y_pred)
    if metric == 'euclidean':
        return euc_dis(y_true, y_pred).mean()


def calc_signal_slope_and_intercept(x: np.array = None, y: np.array = None) -> Tuple[float, float]:
    """
    calculates a signal slope and intercept using scipy linegress model.
    if x is not given, this function generates an array of consequential indices with an interval of 1 and
    uses it as the signal x-axis.
    the function returns the slope and intercept attributes of the calculated linegress object.
    :param x: np.array - the signal x-axis
    :param y: np.array - the signal values (y-axis)
    :return: Tuple[float,float], slope and intercept accordingly
    """
    assert y is not None, 'Y cant be None!'
    if x is None:
        x = np.arange(0, len(y), 1)

    lr_object = linregress(x, y)
    return lr_object.slope, lr_object.intercept


def clean_string_from_bad_chars(treatment_name: str, replacement='_') -> str:
    return treatment_name.replace('\\', replacement).replace('/', replacement)


def normalize(values: np.array, normalization_method: str = 'z_score', axis: int = 0):
    if normalization_method == 'z_score':
        return zscore(values, axis=axis)
    Warning('the normalization method is unknown!')
    return values


def get_dead_cells_mask_in_window(window_start_time_min: int,
                                  window_end_time_min: int,
                                  cells_times_of_death: np.array,
                                  consider_death_in_window_only) -> np.array:
    """
    returns a boolean np.array the shape of cells_times_of_death where each value is True if the cell is dead
    and False otherwise.
    :param window_start_time_min: int, the starting time of the window in minutes
    :param window_end_time_min: int, the end time of the window in minutes
    :param cells_times_of_death: np.array, cells' times of death (in minutes) indexed according to cell indices.
    :param consider_death_in_window_only: bool, whether to consider death which occurred prior to window_start_time_min.
    :return:
    """
    if consider_death_in_window_only:
        return (cells_times_of_death >= window_start_time_min) * (cells_times_of_death < window_end_time_min)
    else:
        return cells_times_of_death < window_end_time_min


def create_trainable_dataset(file_path: str):
    """
    transform a csv into multiple lines of raw training data.
    for each alive cell in every frame, add new record for the cell with his new state
    :param file_path: the path where the csv data file is saved
    :return:
    """

    with open('config.json') as json_file:
        config = json.load(json_file)

    df = pd.read_csv(file_path)
    original_df = df.copy()

    df[CELL_X] = zscore(df[CELL_X])
    df[CELL_Y] = zscore(df[CELL_Y])
    df[DEATH_TIME] = zscore(df[DEATH_TIME])

    dataset = pd.DataFrame()
    dataset[CELL_IDX] = df[UNNAMED_COLUMN]
    dataset[CELL_X] = df[CELL_X]
    dataset[CELL_Y] = df[CELL_Y]

    cell_xy = dataset.loc[:, [CELL_X, CELL_Y]].values
    cells_neighbors_level_1 = get_cells_neighbors(XY=cell_xy, threshold_dist=DIST_THRESHOLD_IN_PIXELS)[0]

    # adding features to the dataframe
    dataset[FRAME_IN_MINUTES] = pd.Series(dtype=float)
    dataset[FRAME_IN_MINUTES] = -1
    dataset[AVG_NEIGHBORS_TIME_OF_DEATH] = pd.Series(dtype=float)
    dataset[NORMALIZED_DIST_TO_CELL_DEAD_NEIGHBORS] = pd.Series(dtype=float)
    dataset[NORMALIZED_DIST_TO_CELL_ALIVE_NEIGHBORS] = pd.Series(dtype=float)
    dataset[NUM_DEAD_NEIGHBORS_BY_ALL_NEIGHBORS] = pd.Series(dtype=float)
    dataset[NUM_ALIVE_NEIGHBORS_BY_ALL_ALIVE_CELLS_IN_TIME_POINT] = pd.Series(dtype=float)

    dead_cells_by_minute = original_df.groupby(DEATH_TIME, sort=True)[DEATH_TIME]


    for frame, deads in dead_cells_by_minute:
        all_alive_cells_at_this_point = len(original_df[original_df[DEATH_TIME] > frame])
        # for each alive cell
        for idx, cell in original_df[original_df[DEATH_TIME] > frame].iterrows():
            my_neighbors_original_tod = original_df.iloc[cells_neighbors_level_1[idx]]
            my_neighbors = df.iloc[cells_neighbors_level_1[idx]]
            dead_neighbors = my_neighbors[my_neighbors_original_tod[DEATH_TIME] <= frame]
            alive_neighbors = my_neighbors[my_neighbors_original_tod[DEATH_TIME] > frame]

            new_row = dataset.iloc[idx].copy()
            new_row[FRAME_IN_MINUTES] = frame

            # TODO: infinity?
            if len(dead_neighbors) == 0:
                new_row[AVG_NEIGHBORS_TIME_OF_DEATH] = -1
                new_row[NORMALIZED_DIST_TO_CELL_DEAD_NEIGHBORS] = -1
            else:
                # the average time of death of the cell's dead neighbors
                new_row[AVG_NEIGHBORS_TIME_OF_DEATH] = np.average(dead_neighbors[DEATH_TIME])

                # distance of the cell from each dead neighbor cell
                dist_from_each_dead_cell = [get_euclidean_distance_between_cells_in_pixels(cell_xy[idx], cell_xy[cell_idx])
                                            for cell_idx in dead_neighbors.index.values.tolist()]
                # normalize the distance
                normalized_dist = [pow(dist, 2) for dist in dist_from_each_dead_cell]
                new_row[NORMALIZED_DIST_TO_CELL_DEAD_NEIGHBORS] = np.mean(normalized_dist)

            if len(alive_neighbors) == 0:
                new_row[NORMALIZED_DIST_TO_CELL_ALIVE_NEIGHBORS] = -1

            else:
                # distance of the cell from each alive neighbor cell
                dist_from_each_alive_cell = [
                    get_euclidean_distance_between_cells_in_pixels(cell_xy[idx], cell_xy[cell_idx])
                    for cell_idx in alive_neighbors.index.values.tolist()]
                # normalize the distance
                normalized_dist = [pow(dist, 2) for dist in dist_from_each_alive_cell]
                new_row[NORMALIZED_DIST_TO_CELL_ALIVE_NEIGHBORS] = np.mean(normalized_dist)

            if len(my_neighbors) == 0:
                new_row[NUM_DEAD_NEIGHBORS_BY_ALL_NEIGHBORS] = 0
            else:
                new_row[NUM_DEAD_NEIGHBORS_BY_ALL_NEIGHBORS] = len(dead_neighbors)/len(my_neighbors)

            new_row[NUM_ALIVE_NEIGHBORS_BY_ALL_ALIVE_CELLS_IN_TIME_POINT] = len(alive_neighbors)/all_alive_cells_at_this_point

            dataset = dataset.append(new_row)

    # TODO: cut frame zero?
    # dataset = dataset.iloc[len(df) + len(dataset[dataset[FRAME_IN_MINUTES] == 0]):]
    dataset = dataset.iloc[len(df):]

    dataset[LABEL] = zscore(df[DEATH_TIME])
    dataset[FRAME_IN_MINUTES] = zscore(dataset[FRAME_IN_MINUTES])

    if config['filter_cells_without_dead_neighbors']:
        dataset = dataset[dataset[AVG_NEIGHBORS_TIME_OF_DEATH] != -1]

    dataset.reset_index(inplace=True)

    # randomize rows order
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # drop unnecessary columns
    prepared_dataset = dataset.drop(['index', CELL_IDX, CELL_X, CELL_Y, 'frame'], axis=1)

    prepared_dataset.to_pickle('dataset.pkl')


# create_trainable_dataset(NON_COMPRESSED_FILE_MAIN_DIR + '/20160820_10A_FB_xy11.csv')
