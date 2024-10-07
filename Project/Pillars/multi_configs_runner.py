import pandas as pd

from runner import *
from pathlib import Path
from scipy.stats import ttest_1samp
import plotly.express as px
from sklearn.manifold import TSNE
from Pillars.pillar_intensities import *



def shuffle_ts_between_all_cells(config_names):
    np.random.seed(42)
    cell2pillars_intens = {}
    series_length2cell = {}
    Consts.SHUFFLE_TS_BETWEEN_CELLS = True
    Consts.shuffled_ts = {}

    # Foreach cell
    # Load pillars to intens
    for config_name in config_names:
        f = open("../configs/" + config_name)
        config_data = json.load(f)
        update_const_by_config(config_data, config_name)

        pillar2ts = get_overall_alive_pillars_to_intensities()

        series_length = len(list(pillar2ts.values())[0])
        if series_length not in series_length2cell:
            series_length2cell[series_length] = []
        series_length2cell[series_length].append(config_name)

    for series, cells in series_length2cell.items():
        shuffle_ts_between_cells(cells)


def shuffle_ts_between_cells(config_names):
    np.random.seed(42)
    cell2pillars_intens = {}
    series_length2cell = {}

    # Foreach cell
    # Load pillars to intens
    for config_name in config_names:
        f = open("../configs/" + config_name)
        config_data = json.load(f)
        update_const_by_config(config_data, config_name)

        pillar2ts = get_overall_alive_pillars_to_intensities()
        cell2pillars_intens[config_name] = pillar2ts

    # Gather all elements into a single list
    all_elements = []
    structure = []

    shuffled_data = {}
    # Store the original structure and gather elements
    for key, subdict in cell2pillars_intens.items():
        for subkey, lst in subdict.items():

            all_elements.append(lst)
            structure.append((key, subkey))

    # Shuffle all elements
    random.shuffle(all_elements)

    # Redistribute elements back into the original structure
    index = 0
    for key, subkey in structure:
        if key not in shuffled_data:
            shuffled_data[key] = {}
        shuffled_data[key][subkey] = all_elements[index]
        index += 1

    Consts.shuffled_ts.update(shuffled_data)


if __name__ == '__main__':

    config_paths = [

        # '5.3/exp_08_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_09_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_12_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_27.1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_27.2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_30_type_5.3_mask_15_35_non-normalized_fixed.json',

        '5.3/exp_20230320-02-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-02-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-02-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-02-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-02-9_type_5.3_mask_15_35_non-normalized_fixed.json',

        '5.3/exp_20230320-03-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-03-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-03-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-03-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-03-9_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        '5.3/exp_20230320-04-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-04-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-04-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-04-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-04-9_type_5.3_mask_15_35_non-normalized_fixed.json',

        '5.3/exp_20230320-05-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-05-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-05-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-05-9_type_5.3_mask_15_35_non-normalized_fixed.json',

        '5.3/exp_20230320-06-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-06-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-06-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230320-06-9_type_5.3_mask_15_35_non-normalized_fixed.json',

        '5.3/exp_20230323-01-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-01-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-01-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-01-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-01-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-01-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-01-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-01-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        # #
        '5.3/exp_20230323-03-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-03-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-03-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-03-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-03-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-03-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-03-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-03-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        # #
        '5.3/exp_20230323-04-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-04-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-04-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-04-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-04-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-04-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-04-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-04-8_type_5.3_mask_15_35_non-normalized_fixed.json',

        '5.3/exp_20230323-05-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-05-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-05-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-05-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-05-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-05-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-05-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-05-8_type_5.3_mask_15_35_non-normalized_fixed.json',

        '5.3/exp_20230323-06-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-06-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-06-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-06-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-06-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-06-8_type_5.3_mask_15_35_non-normalized_fixed.json',

        '5.3/exp_20230323-07-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-07-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-07-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-07-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-07-8_type_5.3_mask_15_35_non-normalized_fixed.json',

        '5.3/exp_20230323-08-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-08-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-08-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-08-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-08-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-08-8_type_5.3_mask_15_35_non-normalized_fixed.json',

        '5.3/exp_20230323-09-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-09-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-09-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-09-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-09-8_type_5.3_mask_15_35_non-normalized_fixed.json',

        '5.3/exp_20230323-10-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-10-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-10-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-10-8_type_5.3_mask_15_35_non-normalized_fixed.json',

        ######### blebbistatin experiments #############
        # '5.3/exp_20230809-00-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-00-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-00-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-00-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-00-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230809-01-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-01-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-01-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-01-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-01-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230809-02-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-02-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-02-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-02-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-02-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # # #
        # '5.3/exp_20230809-03-1_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-03-2_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-03-3_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-03-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230809-04-1_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-04-2_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-04-3_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-04-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230809-05-1_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-05-2_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-05-3_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230809-05-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2023081501-01-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-01-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-01-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-01-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-01-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-01-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-01-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2023081501-02-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-02-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-02-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-02-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-02-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-02-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-02-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        # #
        # '5.3/exp_2023081501-03-1_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-03-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-03-5_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2023081501-04-1_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # #
        # '5.3/exp_2023081502-01-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-01-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-01-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-01-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2023081502-02-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-02-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-02-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-02-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # #
        # '5.3/exp_2023081502-03-1_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-03-2_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-03-3_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-03-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2023081502-04-2_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-04-3_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-04-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # #
        # '5.3/exp_20230818-00-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-00-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-00-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-00-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-00-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230818-01-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-01-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-01-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-01-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-01-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230818-02-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-02-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-02-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-02-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-02-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # #
        # '5.3/exp_20230818-03-2_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-03-3_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-03-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-03-5_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230818-04-3_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-04-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-04-5_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',


        ####################### formin exps ######################
        # '5.3/exp_2024091002-01-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091002-01-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091002-01-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091002-01-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2024091002-02-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091002-02-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091002-02-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091002-02-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # #
        # '5.3/exp_2024091002-04-1_type_5.3_formin_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091002-04-2_type_5.3_formin_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091002-04-3_type_5.3_formin_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091002-04-4_type_5.3_formin_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2024091002-05-1_type_5.3_formin_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091002-05-2_type_5.3_formin_mask_15_35_non-normalized_fixed.json',
        # # '5.3/exp_2024091002-05-3_type_5.3_formin_mask_15_35_non-normalized_fixed.json', # image is moving and cant be fixed
        # '5.3/exp_2024091002-05-4_type_5.3_formin_mask_15_35_non-normalized_fixed.json',
        # #
        # '5.3/exp_20240912-01-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20240912-01-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20240912-01-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # #
        # '5.3/exp_20240912-02-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20240912-02-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20240912-02-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20240912-04-1_type_5.3_formin_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20240912-04-2_type_5.3_formin_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20240912-04-3_type_5.3_formin_mask_15_35_non-normalized_fixed.json',
        # # #
        # '5.3/exp_20240912-05-1_type_5.3_formin_mask_15_35_non-normalized_fixed.json',
        # # '5.3/exp_20240912-05-2_type_5.3_formin_mask_15_35_non-normalized_fixed.json', # not able to detect pillars
        # '5.3/exp_20240912-05-3_type_5.3_formin_mask_15_35_non-normalized_fixed.json',

        #################### arp 2/3 exps ########################
        # '5.3/exp_2024091101-01-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-01-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-01-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-01-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-01-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2024091101-02-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-02-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-02-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-02-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-02-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2024091101-04-1_type_5.3_arp_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-04-2_type_5.3_arp_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-04-3_type_5.3_arp_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-04-4_type_5.3_arp_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-04-5_type_5.3_arp_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2024091101-05-1_type_5.3_arp_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-05-2_type_5.3_arp_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-05-3_type_5.3_arp_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-05-4_type_5.3_arp_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091101-05-5_type_5.3_arp_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2024091102-01-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091102-01-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091102-01-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2024091102-02-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091102-02-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091102-02-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # #
        # '5.3/exp_2024091102-04-1_type_5.3_arp_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091102-04-2_type_5.3_arp_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091102-04-3_type_5.3_arp_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2024091102-05-1_type_5.3_arp_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091102-05-2_type_5.3_arp_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2024091102-05-3_type_5.3_arp_mask_15_35_non-normalized_fixed.json',

        ###############################################################################################################

        # '13.2/exp_01_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_05_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_06_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20_type_13.2_mask_15_35_non-normalized_fixed.json',

        # '13.2/exp_20230319-1-149-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230319-1-149-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230319-1-149-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230319-1-149-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230319-1-149-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        #
        # '13.2/exp_20230319-210-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230319-210-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230319-210-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230319-210-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230319-210-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        #
        # '13.2/exp_20230327-01-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-01-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-01-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-01-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-01-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-01-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-01-7_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-01-8_type_13.2_mask_15_35_non-normalized_fixed.json',
        #
        # '13.2/exp_20230327-02-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-02-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-02-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-02-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-02-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-02-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-02-7_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-02-8_type_13.2_mask_15_35_non-normalized_fixed.json',
        #
        # '13.2/exp_20230327-03-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-03-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-03-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-03-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-03-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-03-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-03-7_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-03-8_type_13.2_mask_15_35_non-normalized_fixed.json',
        # #
        # '13.2/exp_20230327-04-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-04-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-04-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-04-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-04-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-04-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-04-7_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-04-8_type_13.2_mask_15_35_non-normalized_fixed.json',
        # #
        # '13.2/exp_20230327-05-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-05-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-05-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-05-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-05-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-05-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-05-7_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-05-8_type_13.2_mask_15_35_non-normalized_fixed.json',
        # #
        # '13.2/exp_20230327-06-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-06-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-06-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-06-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-06-8_type_13.2_mask_15_35_non-normalized_fixed.json',
        #
        # '13.2/exp_20230327-07-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-07-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-07-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-07-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-07-8_type_13.2_mask_15_35_non-normalized_fixed.json',
        #
        # '13.2/exp_20230327-08-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-08-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-08-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-08-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-08-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-08-8_type_13.2_mask_15_35_non-normalized_fixed.json',
        # #
        # '13.2/exp_20230327-09-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-09-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-09-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-09-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-09-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-09-8_type_13.2_mask_15_35_non-normalized_fixed.json',
        #
        #
        #### Before blebbing exps01 ####
        # '13.2/exp_2023071201-01-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-01-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-01-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-01-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-01-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-01-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        #
        # '13.2/exp_2023071201-02-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-02-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-02-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-02-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-02-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-02-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        # #
        # # ## After blebbing exps01 ####
        # '13.2/exp_2023071201-04-1_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-04-2_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-04-3_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-04-4_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-04-5_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-04-6_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        #
        # '13.2/exp_2023071201-05-1_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-05-2_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-05-3_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-05-4_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-05-5_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071201-05-6_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # #
        # ## Before blebbing exps02 ####
        # '13.2/exp_2023071202-01-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-01-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-01-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-01-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-01-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-01-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-01-7_type_13.2_mask_15_35_non-normalized_fixed.json',
        #
        # '13.2/exp_2023071202-02-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-02-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-02-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-02-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-02-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-02-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-02-7_type_13.2_mask_15_35_non-normalized_fixed.json',
        #
        # ### After blebbing exps02 ####
        # '13.2/exp_2023071202-04-1_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-04-2_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-04-3_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-04-6_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        #
        # '13.2/exp_2023071202-05-1_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-05-2_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-05-3_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_2023071202-05-6_type_13.2_bleb_mask_15_35_non-normalized_fixed.json',

    ]

    Consts.SHOW_GRAPH = True
    Consts.WRITE_OUTPUT = False

    # # TODO: delete
    nbrs_corrs = []
    non_nbrs_corrs = []
    list_53 = []
    list_132 = []
    # nbrs_corrs_new_rad = []
    # non_nbrs_corrs_new_rad = []
    nbrs_corrs_53 = []
    non_nbrs_corrs_53 = []
    nbrs_corrs_before_blebb = []
    non_nbrs_corrs_before_blebb = []
    nbrs_corrs_132 = []
    non_nbrs_corrs_132 = []
    nbrs_corrs_after_blebb = []
    non_nbrs_corrs_after_blebb = []
    delta_corrs_before = []
    delta_corrs_after = []
    # non_nbrs_corrs_bleb_132 = []
    avg_strenght_group_dist_from_center_20230320 = []
    avg_strenght_group_dist_from_center_20230323 = []
    avg_strenght_group_dist_from_center_20230319 = []
    avg_strenght_group_dist_from_center_20230327 = []

    avg_strenght_group_dist_from_center_20230320_rand = []
    avg_strenght_group_dist_from_center_20230323_rand = []
    avg_strenght_group_dist_from_center_20230319_rand = []
    avg_strenght_group_dist_from_center_20230327_rand = []
    #
    # avg_strenght_group_dist_from_center_2023071301 = []
    # avg_strenght_group_dist_from_center_2023071302 = []
    # avg_strenght_group_dist_from_center_2023071201 = []
    # avg_strenght_group_dist_from_center_2023071202 = []

    # avg_strenght_group_dist_from_center_20230809_before_blebb = []
    # avg_strenght_group_dist_from_center_20230809_after_blebb = []
    # avg_strenght_group_dist_from_center_2023081501_before_blebb = []
    # avg_strenght_group_dist_from_center_2023081501_after_blebb = []
    # avg_strenght_group_dist_from_center_2023081502_before_blebb = []
    # avg_strenght_group_dist_from_center_2023081502_after_blebb = []
    # avg_strenght_group_dist_from_center_20230818_before_blebb = []
    # avg_strenght_group_dist_from_center_20230818_after_blebb = []

    # # corrs = []
    # #####
    exps = []
    exps_type = []
    cells_ts = []
    map_exp_to_delta_corrs = {}
    map_exp_to_statistics_original_and_random_strong_nodes_distances = {}
    map_exp_to_statistics_original_and_random_strong_nodes_clusters = {}
    map_exp_to_statistics_original_and_random_strong_nodes_hops = {}
    map_exp_to_statistics_original_and_random_number_of_cc = {}
    map_exp_to_statistics_original_and_random_avg_similarity = {}
    exps_dict_distances_category_lst = []
    strength = []
    similarity = []
    avg_nbrs_sim_lst = []
    avg_non_nbrs_sim_lst = []
    avg_nbrs_sim_lst_53 = []
    avg_non_nbrs_sim_lst_53 = []
    avg_nbrs_sim_lst_132 = []
    avg_non_nbrs_sim_lst_132 = []
    all_sims = {}
    all_corrs = defaultdict(list)
    core_avgs = []
    periphery_avgs = []
    lag_avg_corrs = []
    similarity_nbrs_vs_nonbrs = []
    all_data_df = pd.DataFrame()
    all_nbrs_corrs = []
    all_non_nbrs_corrs = []
    exp_name_to_exp_type = {}
    all_lags = []
    lag_data = {'Lag': [], 'Correlation': []}
    corrs1 = {}
    corrs2 = {}
    corrs3 = {}
    corrs4 = {}
    corrs5 = {}
    corrs6 = {}
    map_cell_to_dist_corrs_lag = {}
    map_cell_to_dist_corrs = {}
    stat = 0
    non_stat = 0
    nbrs_corrs_shuffle_between_cells_ts = []
    total_mean = []
    total_std = []
    all_dist_to_gc = {}
    all_gc_dict_adjacent = {}
    all_gc_dict_non_adjacent = {}

    # radiuses_by_0_10 = ['(15, 35)', '(0, 10)', '(10, 30)', '(20, 40)', '(15, 40)', '(10, 40)']
    # my_dict_0_10 = {k: [] for k in radiuses_by_0_10}
    # radiuses_by_0_15 = ['(15, 35)', '(0, 15)', '(10, 30)', '(20, 40)', '(15, 40)', '(10, 40)']
    # my_dict_0_15 = {k: [] for k in radiuses_by_0_15}
    #
    ####### fig S5 #######
    # radiuses = ['(15, 35)', '(0, 10)', '(0, 15)', '(10, 30)', '(20, 40)', '(15, 40)', '(10, 40)']
    # my_dict = {k: [] for k in radiuses}

    ####### fig S3 #######
    radiuses = ['(15, 35)', '(0, 15)']
    my_dict = {k: [] for k in radiuses}
    my_dict_before_noise_norm = {k: [] for k in radiuses}
    #####

    for config_path in config_paths:
        print(config_path, "started")
        path_name_split = config_path.split('_')
        exp_type = path_name_split[0].split('/')[0]
        exp_name = path_name_split[1]
        Consts.RESULT_FOLDER_PATH = "../multi_config_runner_results/" + exp_type + '/' + exp_name
        Path(Consts.RESULT_FOLDER_PATH).mkdir(parents=True, exist_ok=True)

        # # TODO: delete
        ######
        # f = open(Consts.RESULT_FOLDER_PATH + "/avg_corr_peripheral_vs_central.json")
        # corrs_dict = json.load(f)
        # exps.append(exp_name)
        # exps_type.append(exp_type)
        # center_corrs.append(corrs_dict["centrals"])
        # periph_corrs.append(corrs_dict["peripherals"])
        ######
        # exp = str(exp_type) + " - " + str(exp_name)
        # exps.append(exp)
        # if "bleb" in path_name_split:
        #     exps_type.append("After blebb")
        #     blebb = "After blebb"
        # else:
        #     exps_type.append("Before blebb")
        #     blebb = "Before blebb"
        # if exp_name[9:12] == '210':
        #     exp_video_name = exp_name[9:12]
        # else:
        #     exp_video_name = exp_name[9:14]
        # exp_video_name = exp_name[9:11]
        # exp_video_name = 'Video' + exp_video_name
        # exps.append(exp_name)
        # f = open(Consts.RESULT_FOLDER_PATH + "/intens_movement_sync.txt", "r")
        # data = f.read()
        # corr = data.split(' ')[-1]
        # corrs.append(corr)
        # gc_edge_prob, avg_random_gc_edge_prob = get_experiment_results_data(Consts.RESULT_FOLDER_PATH + '/results.csv',
        #                                                                  ['gc_edge_prob',
        #                                                                   'avg_random_gc_edge_prob'])
        # avg_intens_corr, avg_movement_corr = get_experiment_results_data(Consts.RESULT_FOLDER_PATH + '/results.csv',
        #                                                                  ['nbrs_avg_correlation',
        #                                                                   'neighbors_avg_movement_correlation'])
        # nbrs_intsn_corrs.append(float(not_neighbors_avg_movement_correlation))
        # nbrs_movement_corrs.append(float(nbrs_avg_correlation))

        # nbrs_corrs_lst, non_nbrs_corrs_lst = run_config(config_path)
        # nbrs_corrs.extend(nbrs_corrs_lst)
        # non_nbrs_corrs.extend(non_nbrs_corrs_lst)
        # run_config(config_path)
        ## TODO:
        nbrs_avg_correlation, non_nbrs_avg_correlation = get_experiment_results_data(
            Consts.RESULT_FOLDER_PATH + '/results_6_nbrs.csv',
            ['nbrs_avg_correlation', 'non_nbrs_avg_correlation'])
        nbrs_corrs.append(float(nbrs_avg_correlation))
        non_nbrs_corrs.append(float(non_nbrs_avg_correlation))
        # exps_type.append(exp_type)
        # delta_corr = float(nbrs_avg_correlation) - float(non_nbrs_avg_correlation)

        ###### fig S9 ######
        # if "bleb" in path_name_split or "formin" in path_name_split or "arp" in path_name_split:
        #     exps_type.append('after inhibitor')
        #     nbrs_corrs_after_blebb.append(float(nbrs_avg_correlation))
        #     non_nbrs_corrs_after_blebb.append(float(non_nbrs_avg_correlation))
        # else:
        #     exps_type.append('before inhibitor')
        #     nbrs_corrs_before_blebb.append(float(nbrs_avg_correlation))
        #     non_nbrs_corrs_before_blebb.append(float(non_nbrs_avg_correlation))

        # map_radius_corrs_0_10 = get_experiment_results_data(Consts.RESULT_FOLDER_PATH + '/results.csv',
        #                                                ['correlations_norm_by_noise_(0,10)'])
        # map_radius_corrs_0_10 = eval(map_radius_corrs_0_10[0])
        # for k, v in map_radius_corrs_0_10.items():
        #     my_dict_0_10[str(k)].append(float(v['nbrs_corrs']))

        ######## fig S5 ##########
        # map_radius_corrs = get_experiment_results_data(Consts.RESULT_FOLDER_PATH + '/results_6_nbrs.csv',
        #                                                ['change_mask_radius'])
        # map_radius_corrs = eval(map_radius_corrs[0])
        # for k, v in map_radius_corrs.items():
        #     my_dict[str(k)].append(float(v['nbrs_corrs']))
        # exps_type.append('(15,35)')
        # my_dict['(15, 35)'].append(float(nbrs_avg_correlation))

        ###### Before noise norm for fig S3 ######
        # run_config(config_path)
        # temp_small_mask_radius = Consts.SMALL_MASK_RADIUS
        # temp_large_mask_radius = Consts.LARGE_MASK_RADIUS
        # mask_radiuses_tuples = [(15, 35), (0, 15)]
        # map_radius_corrs_not_norm = {}
        # for mask_radius in mask_radiuses_tuples:
        #     ratio_radiuses = get_mask_radiuses({'small_radius': mask_radius[0], 'large_radius': mask_radius[1]})
        #     Consts.SMALL_MASK_RADIUS = ratio_radiuses['small']
        #     Consts.LARGE_MASK_RADIUS = ratio_radiuses['large']
        #     pillar_intensity_dict = get_pillar_to_intensities(get_images_path(), use_cache=False)
        #     alive_pillar_ids = get_alive_pillar_ids_overall_v3()
        #     alive_pillars_dict = {pillar: pillar_intensity_dict[pillar] for pillar in alive_pillar_ids}
        #     avg_corrs_not_norm = get_neighbors_avg_correlation(
        #                 get_alive_pillars_symmetric_correlation(use_cache=False, norm_by_noise=False,
        #                                                         pillar_to_intensities_dict=alive_pillars_dict),
        #                                                         get_alive_pillars_to_alive_neighbors())
        #     map_radius_corrs_not_norm[mask_radius] = {}
        #     map_radius_corrs_not_norm[mask_radius]['nbrs_corrs'] = avg_corrs_not_norm[0]
        # Consts.SMALL_MASK_RADIUS = temp_small_mask_radius
        # Consts.LARGE_MASK_RADIUS = temp_large_mask_radius
        # # After noise norm
        # map_radius_corrs = get_experiment_results_data(Consts.RESULT_FOLDER_PATH + '/results_6_nbrs.csv',
        #                                                ['change_mask_radius'])
        # map_radius_corrs = eval(map_radius_corrs[0])
        # for k, v in map_radius_corrs.items():
        #     if str(k) in my_dict:
        #         my_dict[str(k)].append(float(v['nbrs_corrs']))
        # for k, v in map_radius_corrs_not_norm.items():
        #     my_dict_before_noise_norm[str(k)].append(float(map_radius_corrs_not_norm[k]['nbrs_corrs']))
        # exps_type.append('(15,35)')
        # my_dict['(15, 35)'].append(float(nbrs_avg_correlation))
        # # x=1

        ##### fig 2A inset #####
        # if exp_type == '5.3':
        #     nbrs_corrs_53.append(float(nbrs_avg_correlation))
        #     non_nbrs_corrs_53.append(float(non_nbrs_avg_correlation))
        # else:
        #     nbrs_corrs_132.append(float(nbrs_avg_correlation))
        #     non_nbrs_corrs_132.append(float(non_nbrs_avg_correlation))

        # rad_0_10 = map_radius_corrs[(0, 10)]
        # nbrs_corrs_new_rad.append(float(rad_0_10['nbrs_corrs']))
        # non_nbrs_corrs_new_rad.append(float(rad_0_10['non_nbrs_corrs']))
        # exps_type2.append('(0,10)')

        # avg_dist, nodes_strengths, strong_nodes = run_config(config_path)
        # strong_nodes_mean_strength = np.mean([nodes_strengths[node_idx] for node_idx in strong_nodes])
        # tup = (avg_dist, strong_nodes_mean_strength)
        # if '20230809' in exp_name.split('-'):
        #     if 'bleb' in path_name_split:
        #         avg_strenght_group_dist_from_center_20230809_after_blebb.append(tup)
        #     else:
        #         avg_strenght_group_dist_from_center_20230809_before_blebb.append(tup)
        # elif '2023081501' in exp_name.split('-'):
        #     if 'bleb' in path_name_split:
        #         avg_strenght_group_dist_from_center_2023081501_after_blebb.append(tup)
        #     else:
        #         avg_strenght_group_dist_from_center_2023081501_before_blebb.append(tup)
        # elif '2023081502' in exp_name.split('-'):
        #     if 'bleb' in path_name_split:
        #         avg_strenght_group_dist_from_center_2023081502_after_blebb.append(tup)
        #     else:
        #         avg_strenght_group_dist_from_center_2023081502_before_blebb.append(tup)
        # elif '20230818' in exp_name.split('-'):
        #     if 'bleb' in path_name_split:
        #         avg_strenght_group_dist_from_center_20230818_after_blebb.append(tup)
        #     else:
        #         avg_strenght_group_dist_from_center_20230818_before_blebb.append(tup)
        # try:
        #     avg = run_config(config_path)
        #     if exp_type == '5.3':
        #         if '20230320' in exp_name.split('-'):
        #             avg_strenght_group_dist_from_center_20230320.append(avg)
        #         elif '20230323' in exp_name.split('-'):
        #             avg_strenght_group_dist_from_center_20230323.append(avg)
        #         # nbrs_corrs_53.append(nbrs_avg_correlation)
        #         # non_nbrs_corrs_53.append(non_nbrs_avg_correlation)
        #     else:
        #         if '20230319' in exp_name.split('-'):
        #             avg_strenght_group_dist_from_center_20230319.append(avg)
        #         elif '20230327' in exp_name.split('-'):
        #             avg_strenght_group_dist_from_center_20230327.append(avg)
        #         # nbrs_corrs_132.append(nbrs_avg_correlation)
        #         # non_nbrs_corrs_132.append(non_nbrs_avg_correlation)
        # except Exception as error:
        #     print("there was an error in config path " + str(config_path) + str(error))
        #     print(config_path, "completed")

        # corrs = run_config(config_path)
        # sum_corrs = np.nansum(corrs.values)
        # sum_diagonal = np.trace(corrs)
        # n = corrs.shape[0]
        # avg_all_corrs = (sum_corrs - sum_diagonal) / (n * (n - 1))
        # if exp_type == '5.3':
        #     list_53.append(avg_all_corrs)
        # if exp_type == '13.2':
        #     list_132.append(avg_all_corrs)

        # strong_nodes_avg_distance, strong_nodes_avg_distance_rand = run_config(config_path)
        # if exp_type == '5.3':
        #     if '20230320' in exp_name.split('-'):
        #         avg_strenght_group_dist_from_center_20230320.append(strong_nodes_avg_distance)
        #         avg_strenght_group_dist_from_center_20230320_rand.append(strong_nodes_avg_distance_rand)
        #     elif '20230323' in exp_name.split('-'):
        #         avg_strenght_group_dist_from_center_20230323.append(strong_nodes_avg_distance)
        #         avg_strenght_group_dist_from_center_20230323_rand.append(strong_nodes_avg_distance_rand)
        #     # nbrs_corrs_53.append(nbrs_avg_correlation)
        #     # non_nbrs_corrs_53.append(non_nbrs_avg_correlation)
        # else:
        #     if '20230319' in exp_name.split('-'):
        #         avg_strenght_group_dist_from_center_20230319.append(strong_nodes_avg_distance)
        #         avg_strenght_group_dist_from_center_20230319_rand.append(strong_nodes_avg_distance_rand)
        #     elif '20230327' in exp_name.split('-'):
        #         avg_strenght_group_dist_from_center_20230327.append(strong_nodes_avg_distance)
        #         avg_strenght_group_dist_from_center_20230327_rand.append(strong_nodes_avg_distance_rand)

        # features = [float(v) for v in features]
        # features.append(corrs_dict["peripherals"])
        # features.append(corrs_dict["centrals"])
        # all_exps_features.append(features)
        # list_dicts.append(run_config(config_path))
        # cc_cp_corrs_dict = run_config(config_path)
        # cc_corr = cc_cp_corrs_dict['cc_corr']
        # cp_corr = cc_cp_corrs_dict['cp_corr']
        # first_corrs.append(gc_edge_prob)
        # second_corrs.append(avg_random_gc_edge_prob)

        # _, corrs = run_config(config_path)
        #
        # if "bleb" in path_name_split:
        # nbrs_corrs_after_bleb.extend(corrs)
        # else:
        # nbrs_corrs_before_blebb.extend(corrs)

        ##### fig 2B #####
        #### correlations delta before and after blebb #####
        # delta_corr = float(nbrs_avg_correlation) - float(non_nbrs_avg_correlation)
        # exp_name_lst = exp_name.split('-')
        # exp_name_to_exp_type[exp_name_lst[0]] = exp_type
        # if exp_name_lst[0] == '20230809':
        #     if '20230809' not in map_exp_to_delta_corrs:
        #         map_exp_to_delta_corrs['20230809'] = {}
        #     if exp_name_lst[-1] not in map_exp_to_delta_corrs['20230809']:
        #         map_exp_to_delta_corrs['20230809'][exp_name_lst[-1]] = {}
        #     if 'bleb' in path_name_split:
        #         if 'after' not in map_exp_to_delta_corrs['20230809'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['20230809'][exp_name_lst[-1]]['after'] = []
        #         map_exp_to_delta_corrs['20230809'][exp_name_lst[-1]]['after'].append(delta_corr)
        #     if 'bleb' not in path_name_split:
        #         if 'before' not in map_exp_to_delta_corrs['20230809'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['20230809'][exp_name_lst[-1]]['before'] = []
        #         map_exp_to_delta_corrs['20230809'][exp_name_lst[-1]]['before'].append(delta_corr)
        #
        # if exp_name_lst[0] == '2023081501':
        #     if '2023081501' not in map_exp_to_delta_corrs:
        #         map_exp_to_delta_corrs['2023081501'] = {}
        #     if exp_name_lst[-1] not in map_exp_to_delta_corrs['2023081501']:
        #         map_exp_to_delta_corrs['2023081501'][exp_name_lst[-1]] = {}
        #     if 'bleb' in path_name_split:
        #         if 'after' not in map_exp_to_delta_corrs['2023081501'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['2023081501'][exp_name_lst[-1]]['after'] = []
        #         map_exp_to_delta_corrs['2023081501'][exp_name_lst[-1]]['after'].append(delta_corr)
        #     if 'bleb' not in path_name_split:
        #         if 'before' not in map_exp_to_delta_corrs['2023081501'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['2023081501'][exp_name_lst[-1]]['before'] = []
        #         map_exp_to_delta_corrs['2023081501'][exp_name_lst[-1]]['before'].append(delta_corr)
        #
        # if exp_name_lst[0] == '2023081502':
        #     if '2023081502' not in map_exp_to_delta_corrs:
        #         map_exp_to_delta_corrs['2023081502'] = {}
        #     if exp_name_lst[-1] not in map_exp_to_delta_corrs['2023081502']:
        #         map_exp_to_delta_corrs['2023081502'][exp_name_lst[-1]] = {}
        #     if 'bleb' in path_name_split:
        #         if 'after' not in map_exp_to_delta_corrs['2023081502'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['2023081502'][exp_name_lst[-1]]['after'] = []
        #         map_exp_to_delta_corrs['2023081502'][exp_name_lst[-1]]['after'].append(delta_corr)
        #     if 'bleb' not in path_name_split:
        #         if 'before' not in map_exp_to_delta_corrs['2023081502'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['2023081502'][exp_name_lst[-1]]['before'] = []
        #         map_exp_to_delta_corrs['2023081502'][exp_name_lst[-1]]['before'].append(delta_corr)
        #
        # if exp_name_lst[0] == '20230818':
        #     if '20230818' not in map_exp_to_delta_corrs:
        #         map_exp_to_delta_corrs['20230818'] = {}
        #     if exp_name_lst[-1] not in map_exp_to_delta_corrs['20230818']:
        #         map_exp_to_delta_corrs['20230818'][exp_name_lst[-1]] = {}
        #     if 'bleb' in path_name_split:
        #         if 'after' not in map_exp_to_delta_corrs['20230818'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['20230818'][exp_name_lst[-1]]['after'] = []
        #         map_exp_to_delta_corrs['20230818'][exp_name_lst[-1]]['after'].append(delta_corr)
        #     if 'bleb' not in path_name_split:
        #         if 'before' not in map_exp_to_delta_corrs['20230818'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['20230818'][exp_name_lst[-1]]['before'] = []
        #         map_exp_to_delta_corrs['20230818'][exp_name_lst[-1]]['before'].append(delta_corr)
        #
        # if exp_name_lst[0] == '2023071201':
        #     if '2023071201' not in map_exp_to_delta_corrs:
        #         map_exp_to_delta_corrs['2023071201'] = {}
        #     if exp_name_lst[-1] not in map_exp_to_delta_corrs['2023071201']:
        #         map_exp_to_delta_corrs['2023071201'][exp_name_lst[-1]] = {}
        #     if 'bleb' in path_name_split:
        #         if 'after' not in map_exp_to_delta_corrs['2023071201'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['2023071201'][exp_name_lst[-1]]['after'] = []
        #         map_exp_to_delta_corrs['2023071201'][exp_name_lst[-1]]['after'].append(delta_corr)
        #     if 'bleb' not in path_name_split:
        #         if 'before' not in map_exp_to_delta_corrs['2023071201'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['2023071201'][exp_name_lst[-1]]['before'] = []
        #         map_exp_to_delta_corrs['2023071201'][exp_name_lst[-1]]['before'].append(delta_corr)
        #
        # if exp_name_lst[0] == '2023071202':
        #     if '2023071202' not in map_exp_to_delta_corrs:
        #         map_exp_to_delta_corrs['2023071202'] = {}
        #     if exp_name_lst[-1] not in map_exp_to_delta_corrs['2023071202']:
        #         map_exp_to_delta_corrs['2023071202'][exp_name_lst[-1]] = {}
        #     if 'bleb' in path_name_split:
        #         if 'after' not in map_exp_to_delta_corrs['2023071202'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['2023071202'][exp_name_lst[-1]]['after'] = []
        #         map_exp_to_delta_corrs['2023071202'][exp_name_lst[-1]]['after'].append(delta_corr)
        #     if 'bleb' not in path_name_split:
        #         if 'before' not in map_exp_to_delta_corrs['2023071202'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['2023071202'][exp_name_lst[-1]]['before'] = []
        #         map_exp_to_delta_corrs['2023071202'][exp_name_lst[-1]]['before'].append(delta_corr)
        #
        # # for revision #
        # if exp_name_lst[0] == '2024091002':
        #     if '2024091002' not in map_exp_to_delta_corrs:
        #         map_exp_to_delta_corrs['2024091002'] = {}
        #     if exp_name_lst[-1] not in map_exp_to_delta_corrs['2024091002']:
        #         map_exp_to_delta_corrs['2024091002'][exp_name_lst[-1]] = {}
        #     if 'formin' in path_name_split:
        #         if 'after' not in map_exp_to_delta_corrs['2024091002'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['2024091002'][exp_name_lst[-1]]['after'] = []
        #         map_exp_to_delta_corrs['2024091002'][exp_name_lst[-1]]['after'].append(delta_corr)
        #     if 'formin' not in path_name_split:
        #         if 'before' not in map_exp_to_delta_corrs['2024091002'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['2024091002'][exp_name_lst[-1]]['before'] = []
        #         map_exp_to_delta_corrs['2024091002'][exp_name_lst[-1]]['before'].append(delta_corr)
        #
        # if exp_name_lst[0] == '20240912':
        #     if '20240912' not in map_exp_to_delta_corrs:
        #         map_exp_to_delta_corrs['20240912'] = {}
        #     if exp_name_lst[-1] not in map_exp_to_delta_corrs['20240912']:
        #         map_exp_to_delta_corrs['20240912'][exp_name_lst[-1]] = {}
        #     if 'formin' in path_name_split:
        #         if 'after' not in map_exp_to_delta_corrs['20240912'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['20240912'][exp_name_lst[-1]]['after'] = []
        #         map_exp_to_delta_corrs['20240912'][exp_name_lst[-1]]['after'].append(delta_corr)
        #     if 'formin' not in path_name_split:
        #         if 'before' not in map_exp_to_delta_corrs['20240912'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['20240912'][exp_name_lst[-1]]['before'] = []
        #         map_exp_to_delta_corrs['20240912'][exp_name_lst[-1]]['before'].append(delta_corr)
        #
        # if exp_name_lst[0] == '2024091101':
        #     if '2024091101' not in map_exp_to_delta_corrs:
        #         map_exp_to_delta_corrs['2024091101'] = {}
        #     if exp_name_lst[-1] not in map_exp_to_delta_corrs['2024091101']:
        #         map_exp_to_delta_corrs['2024091101'][exp_name_lst[-1]] = {}
        #     if 'arp' in path_name_split:
        #         if 'after' not in map_exp_to_delta_corrs['2024091101'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['2024091101'][exp_name_lst[-1]]['after'] = []
        #         map_exp_to_delta_corrs['2024091101'][exp_name_lst[-1]]['after'].append(delta_corr)
        #     if 'arp' not in path_name_split:
        #         if 'before' not in map_exp_to_delta_corrs['2024091101'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['2024091101'][exp_name_lst[-1]]['before'] = []
        #         map_exp_to_delta_corrs['2024091101'][exp_name_lst[-1]]['before'].append(delta_corr)
        #
        # if exp_name_lst[0] == '2024091102':
        #     if '2024091102' not in map_exp_to_delta_corrs:
        #         map_exp_to_delta_corrs['2024091102'] = {}
        #     if exp_name_lst[-1] not in map_exp_to_delta_corrs['2024091102']:
        #         map_exp_to_delta_corrs['2024091102'][exp_name_lst[-1]] = {}
        #     if 'arp' in path_name_split:
        #         if 'after' not in map_exp_to_delta_corrs['2024091102'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['2024091102'][exp_name_lst[-1]]['after'] = []
        #         map_exp_to_delta_corrs['2024091102'][exp_name_lst[-1]]['after'].append(delta_corr)
        #     if 'arp' not in path_name_split:
        #         if 'before' not in map_exp_to_delta_corrs['2024091102'][exp_name_lst[-1]]:
        #             map_exp_to_delta_corrs['2024091102'][exp_name_lst[-1]]['before'] = []
        #         map_exp_to_delta_corrs['2024091102'][exp_name_lst[-1]]['before'].append(delta_corr)

        # sim = run_config(config_path)
        # sns.histplot(sim, label="similarity", kde=True, alpha=0.3)
        # plt.xlabel('Similarity')
        # plt.title("Strength Similarity Distribution of Neighboring Pillars")
        # plt.legend()
        # plt.show()

        # avg_strength, avg_similarity = run_config(config_path)
        # strength.append(avg_strength)
        # similarity.append(avg_similarity)

        # total_avg_intensity, pillars_avg_intens, pillars_strength = run_config(config_path)
        # p_to_intens, avg_ts, total_avg_intensity, pillars_avg_raw_intens, pillars_strength = run_config(config_path)

        # nbrs_sim_lst, non_nbrs_sim_lst = run_config(config_path)
        # p_val = plot_distribution_similarity_of_exp_nbrs_vs_non_nbrs(nbrs_sim_lst, non_nbrs_sim_lst)
        # similarity_nbrs_vs_nonbrs.append(p_val)
        # avg_nbrs, avg_nonbrs = run_config(config_path)
        # if avg_nbrs > avg_nonbrs:
        #     similarity_nbrs_vs_nonbrs.append(1)
        # else:
        #     similarity_nbrs_vs_nonbrs.append(0)

        # avg_nbrs_sim, avg_non_nbrs_sim = run_config(config_path)
        # avg_nbrs_sim_lst.append(avg_nbrs_sim)
        # avg_non_nbrs_sim_lst.append(avg_non_nbrs_sim)
        # exps_type.append(exp_type)

        ##### fig 1F #####
        # level_to_corrs = run_config(config_path)
        # # print(similarity_to_nbrhood_level_correlation(level_to_similarities))
        # for k, v in level_to_corrs.items():
        #     if k not in all_corrs.keys():
        #         all_corrs[k] = []
        #     all_corrs[k].append(np.mean(v))
        # ###### for significant in each distace before and after shuffle - 1f vs S6 ######
        # exp_name_lst = exp_name.split('-')
        # exp = exp_name_lst[0]
        # cell = exp_name_lst[-1]
        # key = exp + '-' + cell
        # dist_to_mean_corrs = {dist: [np.mean(corrs)] for dist, corrs in level_to_corrs.items()}
        # if key not in map_cell_to_dist_corrs.keys():
        #     map_cell_to_dist_corrs[key] = dist_to_mean_corrs
        # else:
        #     for dist, corr in dist_to_mean_corrs.items():
        #         if dist in map_cell_to_dist_corrs[key]:
        #             map_cell_to_dist_corrs[key][dist].extend(corr)
        #         else:
        #             map_cell_to_dist_corrs[key][dist] = corr


        # core_strength, periphery_strength = run_config(config_path)
        # print("Average core:", "%.3f" % np.mean(core_strength))
        # print("Average periphery:", "%.3f" % np.mean(periphery_strength))
        # core_corrs, periphery_corrs, border_corrs = run_config(config_path)
        # print("Average core:", "%.3f" % np.mean(core_corrs))
        # print("Average periphery:", "%.3f" % np.mean(periphery_corrs))
        # print("Average border:", "%.3f" % np.mean(border_corrs))
        # nbrs_sims, core_sims, periphery_sims, border_sims = run_config(config_path)
        # print("Average nbrs similarity:", "%.3f" % np.mean(nbrs_sims))
        # print("Average core:", "%.3f" % np.mean(core_sims))
        # print("Average periphery:", "%.3f" % np.mean(periphery_sims))
        # print("Average border:", "%.3f" % np.mean(border_sims))
        # plt.bar(['total avg', 'core avg', 'periphery avg', 'border avg'],
        #         [np.mean(nbrs_sims), np.mean(core_sims), np.mean(periphery_sims), np.mean(border_sims)])
        # plt.show()
        # avg_core_sim, avg_periphery_sim = run_config(config_path)
        # core_avgs.append(avg_core_sim)
        # periphery_avgs.append(avg_periphery_sim)

        # p_2_sim_dict = run_config(config_path)
        # lag_to_avg_corr_dict = run_config(config_path)
        # lag_avg_corrs.append(lag_to_avg_corr_dict)

        ### t-SNE ###
        # p_to_intns, core, periph = run_config(config_path)
        # ts_len = len(list(p_to_intns.values())[0])
        # blebb = None
        # # if "bleb" in path_name_split:
        # #     exps_type.append("After blebb")
        # #     blebb = "After blebb"
        # # else:
        # #     exps_type.append("Before blebb")
        # #     blebb = "Before blebb"
        # if ts_len > 120:
        #     p_to_intns = {k: v[:120] for k, v in p_to_intns.items()}
        # if ts_len < 120:
        #     continue
        # data_df = df_for_tsne(p_to_intns, exp_name, exp_type, core=core, periph=periph, blebb=blebb, ts_features=False)
        # all_data_df = pd.concat([all_data_df, data_df], axis=0)
        # all_data_df.reset_index(drop=True, inplace=True)

        # id_to_intns = {exp_name+str(k): v for k, v in p_to_intns.items()}
        # df = pd.DataFrame(id_to_intns.items(), columns=['id', 'time_series'])
        # type_series = pd.Series([exp_type] * len(df))
        # df['type'] = type_series
        # for i, p in enumerate(list(p_to_intns.keys())):
        #     if p in core:
        #         df.loc[i, 'loc'] = 'core'
        #     if p in periph:
        #         df.loc[i, 'loc'] = 'periphery'
        # time_series_expanded = df['time_series'].apply(pd.Series)
        # time_series_expanded.columns = [str(i) for i in time_series_expanded.columns]
        # result_df = pd.concat([df.drop('time_series', axis=1), time_series_expanded], axis=1)
        # # result_df = df.drop('time_series', axis=1)
        # X = result_df.drop(['id', 'type', 'loc'], axis=1)
        # # feature_list = X.apply(lambda row: extract_ts_features(row), axis=1)
        # # X_features = pd.DataFrame(feature_list.tolist())
        # df = pd.concat([df.drop(['time_series'], axis=1), X], axis=1)
        # # df['id'] = exp_name
        # # df = df.groupby(['id', 'type'], as_index=False).mean()
        # all_data_df = pd.concat([all_data_df, df], axis=0)
        # all_data_df.reset_index(drop=True, inplace=True)

        # p_to_intns = run_config(config_path)
        # row = {'experiment name': exp_name, 'type': exp_type, 'number of pillars': len(p_to_intns.keys()), 'average intensity': np.mean([np.mean(i) for i in list(p_to_intns.values())])}
        # all_data_df = all_data_df.append(row, ignore_index=True)

        # med = np.median(avg_ts)
        # peaks = []
        # for i, int in enumerate(avg_ts):
        #     if int > med:
        #         peaks.append(i)
        # np.diff(peaks)
        # p_to_intns, _, _ = run_config(config_path)
        # for p, intens in p_to_intns.items():
        #     med = np.median(intens)
        #     peaks = []
        #     for i, int in enumerate(intens):
        #         if int > med:
        #             peaks.append(i)
        #     np.diff(peaks)

        # nbrs_corrs_list, non_nbrs_corrs_list = run_config(config_path)
        # all_nbrs_corrs.extend(nbrs_corrs_list)
        # all_non_nbrs_corrs.extend(non_nbrs_corrs_list)

        # peak_lags = run_config(config_path)
        # for pair, values in peak_lags.items():
        #     lag_data['Lag'].append(values[0])
        #     lag_data['Correlation'].append(values[1])
        # lags = [lag[0] for _, lag in peak_lags.items()]
        # all_lags.extend(lags)

        # stationary_p_to_intens, non_stationary_pillars = run_config(config_path)
        # stat += len(stationary_p_to_intens)
        # non_stat += len(non_stationary_pillars)

        ######### Fig 1H ##########
        # correlations, correlations2, correlations3, correlations4, correlations5, correlations6 = run_config(config_path)
        # all_corrs_dicts = [correlations, correlations2, correlations3, correlations4, correlations5, correlations6]
        # exp_name_lst = exp_name.split('-')
        # exp = exp_name_lst[0]
        # cell = exp_name_lst[-1]
        # key = exp + '-' + cell
        # if key not in map_cell_to_dist_corrs_lag.keys():
        #     map_cell_to_dist_corrs_lag[key] = {dist: {} for dist, dict_corr in enumerate(all_corrs_dicts, start=1)}
        # for dist, corrs_dict in map_cell_to_dist_corrs_lag[key].items():
        #     corrs_dict.update(all_corrs_dicts[dist-1])

        ##### get the mean and std distance between neighbors ######
        # mean_distance, std_distance = run_config(config_path)
        # total_mean.append(mean_distance)
        # total_std.append(std_distance)

        # vid_dist_to_gc = run_config(config_path)
        # for k, v in vid_dist_to_gc.items():
        #     if k in all_dist_to_gc:
        #         all_dist_to_gc[k].update(v)
        #     else:
        #         all_dist_to_gc[k] = v
        # gc_dict_adjacent, gc_dict_non_adjacent = run_config(config_path)
        # all_gc_dict_adjacent.update(gc_dict_adjacent)
        # all_gc_dict_non_adjacent.update(gc_dict_non_adjacent)

        ##################################### RUN  ############################################
        run_config(config_path)

        # try:
        #     run_config(config_path)
        # #     # observed_diff, permuted_diffs, p_value = run_config(config_path)
        # #     # map_exp_to_statistics_original_and_random_avg_similarity[exp_name] = (
        # #     #     observed_diff, permuted_diffs, p_value)
        # except Exception as error:
        #     print("there was an error in config path " + str(config_path) + str(error))
        # print(config_path, "completed")
        ##################################### RUN  ############################################
    # data = []
    # for dist, pairs in all_dist_to_gc.items():
    #     if dist < 6:
    #         for pair, p_value in pairs.items():
    #             data.append((dist, p_value))
    # df = pd.DataFrame(data, columns=['Distance', 'P-Value'])
    # plt.figure(figsize=(10, 6))
    # bplot = sns.boxplot(x='Distance', y='P-Value', data=df, color="gray")
    # medians = df.groupby(['Distance'])['P-Value'].median()
    # vertical_offset = df['P-Value'].median() * 0.05
    # for xtick in bplot.get_xticks():
    #     bplot.text(xtick, medians[xtick + 1] + vertical_offset, f'{medians[xtick + 1]:.2f}',
    #                horizontalalignment='center', size='small', color='black', weight='semibold')
    # plt.title('Granger Causality P-Values by Topological Distance')
    # plt.xlabel('Topological Distance')
    # plt.ylabel('P-Value')
    # plt.show()
    # distances = sorted(df['Distance'].unique())
    # for i in range(len(distances) - 1):
    #     dist1 = distances[i]
    #     dist2 = distances[i + 1]
    #     p_values_dist1 = df[df['Distance'] == dist1]['P-Value']
    #     p_values_dist2 = df[df['Distance'] == dist2]['P-Value']
    #     stat, p_value = stats.mannwhitneyu(p_values_dist1, p_values_dist2, alternative='two-sided')
    #     print(f"Mann-Whitney U Test between distances {dist1} and {dist2}: U-statistic = {stat}, p-value = {p_value}")

    # data_adjacent = pd.DataFrame(list(all_gc_dict_adjacent.items()), columns=['Pair', 'P-Value'])
    # data_non_adjacent = pd.DataFrame(list(all_gc_dict_non_adjacent.items()), columns=['Pair', 'P-Value'])
    # data_adjacent['Type'] = 'Adjacent'
    # data_non_adjacent['Type'] = 'Non-Adjacent'
    # combined_data = pd.concat([data_adjacent, data_non_adjacent])
    # stat, p_value = stats.mannwhitneyu(data_adjacent['P-Value'], data_non_adjacent['P-Value'], alternative='two-sided')
    # print(f"Mann-Whitney U Test results: U-statistic = {stat}, p-value = {p_value}")
    # plt.figure(figsize=(8, 6))
    # bplot = sns.boxplot(x='Type', y='P-Value', data=combined_data, color="gray")
    # medians = combined_data.groupby(['Type'])['P-Value'].median()
    # vertical_offset = combined_data['P-Value'].median() * 0.05
    # for xtick in bplot.get_xticks():
    #     bplot.text(xtick, medians[xtick] + vertical_offset, f'{medians[xtick]:.2f}',
    #                horizontalalignment='center', color='black', fontsize=10, weight='semibold')
    # plt.title('Comparison of Granger Causality P-Values')
    # plt.xlabel('Type of Pair')
    # plt.ylabel('Granger Causality P-Value')
    # plt.show()

    # print("mean:", np.mean(total_mean))
    # print("std:", np.std(total_std))
    ###### revision for supplementary - shuffle pillars between cells - 1F + 1H + S3 ######
    # for config_path in config_paths:
    #     print("config:", config_path)
    #     temp_corrs = defaultdict(list)
    #     config_nbrs_corrs_temp = []
    #     accumulated_corrs = defaultdict(lambda: [None, []])
    #     accumulated_corrs2 = defaultdict(lambda: [None, []])
    #     accumulated_corrs3 = defaultdict(lambda: [None, []])
    #     accumulated_corrs4 = defaultdict(lambda: [None, []])
    #     accumulated_corrs5 = defaultdict(lambda: [None, []])
    #     accumulated_corrs6 = defaultdict(lambda: [None, []])
    #     for i in range(10):
    #         print("iteration:", i)
    #         random.seed(i)
    #         shuffle_ts_between_all_cells(config_paths)
    #     ### 1F ####
    #         level_to_corrs = run_config(config_path)
    #         for k, v in level_to_corrs.items():
    #             temp_corrs[k].append(np.mean(v))
    #     for k, v in temp_corrs.items():
    #         all_corrs[k].append(np.mean(v))
        #### 1H ####
        #     correlations, correlations2, correlations3, correlations4, correlations5, correlations6 = run_config(
        #         config_path)
        #     for pillars, (lags, corr_values) in correlations.items():
        #         if accumulated_corrs[pillars][0] is None:
        #             accumulated_corrs[pillars][0] = lags  # Store lags once
        #         accumulated_corrs[pillars][1].append(corr_values)
        #     for pillars, (lags, corr_values) in correlations2.items():
        #         if accumulated_corrs2[pillars][0] is None:
        #             accumulated_corrs2[pillars][0] = lags
        #         accumulated_corrs2[pillars][1].append(corr_values)
        #     for pillars, (lags, corr_values) in correlations3.items():
        #         if accumulated_corrs3[pillars][0] is None:
        #             accumulated_corrs3[pillars][0] = lags
        #         accumulated_corrs3[pillars][1].append(corr_values)
        #     for pillars, (lags, corr_values) in correlations4.items():
        #         if accumulated_corrs4[pillars][0] is None:
        #             accumulated_corrs4[pillars][0] = lags
        #         accumulated_corrs4[pillars][1].append(corr_values)
        #
        #     for pillars, (lags, corr_values) in correlations5.items():
        #         if accumulated_corrs5[pillars][0] is None:
        #             accumulated_corrs5[pillars][0] = lags
        #         accumulated_corrs5[pillars][1].append(corr_values)
        #     for pillars, (lags, corr_values) in correlations6.items():
        #         if accumulated_corrs6[pillars][0] is None:
        #             accumulated_corrs6[pillars][0] = lags
        #         accumulated_corrs6[pillars][1].append(corr_values)
        # def compute_mean_correlation(accumulated_corrs):
        #     mean_correlations = {}
        #     for pillars, (lags, corr_values_list) in accumulated_corrs.items():
        #         # Compute mean correlation across all iterations for each time lag
        #         mean_corr_values = np.mean(corr_values_list, axis=0)
        #         mean_correlations[pillars] = (lags, mean_corr_values)
        #     return mean_correlations
        # all_corrs_dicts = [compute_mean_correlation(accumulated_corrs), compute_mean_correlation(accumulated_corrs2), compute_mean_correlation(accumulated_corrs3), compute_mean_correlation(accumulated_corrs4),
        #     compute_mean_correlation(accumulated_corrs5), compute_mean_correlation(accumulated_corrs6)]
        # path_name_split = config_path.split('_')
        # exp_type = path_name_split[0].split('/')[0]
        # exp_name = path_name_split[1]
        # exp_name_lst = exp_name.split('-')
        # exp = exp_name_lst[0]
        # cell = exp_name_lst[-1]
        # key = exp + '-' + cell
        # if key not in map_cell_to_dist_corrs_lag.keys():
        #     map_cell_to_dist_corrs_lag[key] = {dist: {} for dist, dict_corr in enumerate(all_corrs_dicts, start=1)}
        # for dist, corrs_dict in map_cell_to_dist_corrs_lag[key].items():
        #     corrs_dict.update(all_corrs_dicts[dist-1])
        #### fig S3 ####
    #         run_config(config_path)
    #         nbrs_avg_correlation, _ = get_neighbors_avg_correlation(get_alive_pillars_symmetric_correlation(),
    #                                                                 get_alive_pillars_to_alive_neighbors())
    #         config_nbrs_corrs_temp.append(float(nbrs_avg_correlation))
    #     nbrs_corrs_shuffle_between_cells_ts.append(np.mean(config_nbrs_corrs_temp))
    # data = [nbrs_corrs, nbrs_corrs_shuffle_between_cells_ts]
    # print("Data 0 Length:", len(data[0]), "Data:", data[0])
    # print("Data 1 Length:", len(data[1]), "Data:", data[1])
    # fig, ax = plt.subplots()
    # bplot = ax.boxplot(data, patch_artist=True, notch=False)
    # # Apply colors and linestyles
    # for patch, color in zip(bplot['boxes'], ['black', 'black']):
    #     patch.set_facecolor('white')  # Set the box's face color to white
    #     patch.set_edgecolor(color)  # Set the edge color
    # for i, mean in enumerate([np.mean(d) for d in data]):
    #     ax.text(i + 1, mean, f'{mean:.2f}', ha='center', va='bottom', fontsize=8)
    # plt.xticks([1, 2], ['cell original pillars', 'shuffled pillars between cells'])
    # y_min, y_max = ax.get_ylim()
    # ax.set_ylim(y_min, y_max + 0.1)
    # ax.set_ylabel('Neighbors Correlations')
    # plt.tight_layout()
    # plt.savefig('../control_fig_S3.svg', format="svg")
    # plt.show()
    # t_stat, p_value = ttest_ind(nbrs_corrs, nbrs_corrs_shuffle_between_cells_ts)
    # print("P-Value: ", p_value)




    ######### fig 1G ###########
    # np.random.seed(100)
    # time = np.arange(20)  # Time from 0 to 19 for Signal 1
    # extended_time = np.arange(3, 23)  # Corrected extended time
    # base_signal = np.random.rand(20)  # Regenerate base signal for consistency
    # signal1 = base_signal.copy()  # Signal 1 is the base signal
    # extended_signal2 = np.concatenate(
    #     [base_signal[-3:], base_signal, base_signal[:3]])
    # plt.figure(figsize=(12, 6))
    # plt.plot(time, signal1, label='Signal 1', color='blue')
    # plt.plot(extended_time, extended_signal2[3:23], label='Signal 2', color='orange')
    # plt.title('Two Almost Identical Time Series with Corrected Shift and Extension')
    # plt.xlabel('Time')
    # plt.ylabel('Signal')
    # plt.legend()
    # plt.xticks([])  # Remove x-axis numbers
    # plt.yticks([])  # Remove y-axis numbers
    # plt.savefig('../time_shift_illustration_1g.svg', format="svg")
    # plt.show()

    # print('non stat', non_stat)
    # print('stat', stat)
    # lag_distribution_plot(all_lags)
    # print(peak_lags)
    # sns.distplot(all_lags, kde=True)
    # plt.hist(all_lags, bins=range(min(all_lags), max(all_lags) + 2), alpha=0.7, align='left')
    # plt.xlabel('Lags')
    # plt.ylabel('Frequency')
    # plt.xticks(range(min(all_lags), max(all_lags) + 1))
    # plt.show()
    #
    # df = pd.DataFrame(lag_data)
    # sns.boxplot(x='Lag', y='Correlation', data=df)
    # plt.title('Peak Correlation Lags')
    # plt.xlabel('Lags')
    # plt.ylabel('Correlation Values')
    # plt.show()

    ######### Fig 1H ###########
    # map_cell_to_dist_mean_corrs = {}
    # diffs_by_dist = {}
    # stats_by_dist = {}
    # start_idx = 3
    # for cell, dist_corrs_dict in map_cell_to_dist_corrs_lag.items():
    #     map_cell_to_dist_mean_corrs[cell] = {}
    #     for dist, dict_corr in dist_corrs_dict.items():
    #         diff_3_0 = []
    #         for val in dict_corr.values():
    #             lag = val[0]
    #             corr = val[1]
    #             # corr3 = corr[start_idx:start_idx + 4]
    #             # diff_3_0.append(max(corr3) - corr3[0])
    #             diff_3_0.append(max(corr) - corr[3])
    #         map_cell_to_dist_mean_corrs[cell][dist] = np.mean(diff_3_0)
    # rows = []
    # for cell, distances in map_cell_to_dist_mean_corrs.items():
    #     for distance, correlation in distances.items():
    #         rows.append({'Cell': cell, 'Topological Distance': distance, 'Mean Delta Correlation': correlation})
    # df = pd.DataFrame(rows)
    # # from scipy.stats import spearmanr
    # # correlation, p_value = spearmanr(df['Topological Distance'], df['Mean Delta Correlation'])
    # # print(f"Spearman correlation: {correlation}, P-value: {p_value}")
    # correlation, p_value = pearsonr(df['Topological Distance'], df['Mean Delta Correlation'])
    # print(f"Pearson correlation: {correlation}, P-value: {p_value}")
    # cells = df['Cell'].unique()
    # palette = sns.color_palette("Spectral", len(cells))
    # plt.figure(figsize=(12, 6))
    # ax = sns.swarmplot(data=df, x='Topological Distance', y='Mean Delta Correlation', hue='Cell', palette=sns.color_palette("Spectral", len(cells)), dodge=True, size=8)
    # sns.boxplot(data=df, x='Topological Distance', y='Mean Delta Correlation', color='lightgrey', showcaps=True,
    #             boxprops={'facecolor': 'None'})
    # # ax.set_ylim(None, 0.11)
    # for i in range(1, 7):
    #     subset = df[df['Topological Distance'] == i]
    #     mean_val = subset['Mean Delta Correlation'].mean()
    #     median_val = subset['Mean Delta Correlation'].median()
    #     print("mean for boxplot distance",i, ":", mean_val)
    #     print("median for boxplot distance", i, ":", median_val)
    # # plt.title("Cell's Mean Delta Correlation of Max Lag 3 by Topological Distance")
    # plt.legend(title='Cell ID', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.savefig('../cells_mean_delta_corr_by_dist_fig_1h.svg', format="svg",  bbox_inches='tight')
    # plt.show()
    # results = []
    # # Group the DataFrame by 'Cell' and loop through each unique distance pair
    # max_distance = df['Topological Distance'].max()
    # for dist1 in range(1, max_distance + 1):
    #     for dist2 in range(dist1 + 1,  max_distance + 1):  # Ensure dist2 > dist1 to avoid duplicate pairs and self-comparison
    #         diffs = []
    #         for cell in df['Cell'].unique():
    #             subset = df[df['Cell'] == cell]
    #             if dist1 in subset['Topological Distance'].values and dist2 in subset['Topological Distance'].values:
    #                 val1 = subset[subset['Topological Distance'] == dist1]['Mean Delta Correlation'].iloc[0]
    #                 val2 = subset[subset['Topological Distance'] == dist2]['Mean Delta Correlation'].iloc[0]
    #                 diff = val2 - val1
    #                 diffs.append(diff)
    #         # Perform a statistical test on the diffs if there are enough pairs to compare
    #         if len(diffs) > 1:  # Wilcoxon requires at least two pairs
    #             stat, p_value = stats.wilcoxon(diffs)
    #             results.append({
    #                 'Distance Pair': f'{dist1} to {dist2}',
    #                 'Statistic': stat,
    #                 'P-value': p_value
    #             })
    #         else:
    #             results.append({
    #                 'Distance Pair': f'{dist1} to {dist2}',
    #                 'Statistic': None,
    #                 'P-value': None
    #             })
    #
    # results_df = pd.DataFrame(results)
    # print(results_df)

    ##### T-test to show the diff between distance 1 to 2 with original pillars is greater than with shuffle pillars ####
    # pivot_table = df[df['Topological Distance'] <= 2].groupby(['Cell', 'Topological Distance']).mean().unstack(level=-1)
    # pivot_table['difference'] = pivot_table[('Mean Delta Correlation', 2)] - pivot_table[('Mean Delta Correlation', 1)]
    # difference_list = pivot_table['difference'].tolist()
    # print(difference_list)

    # original_diff = [0.01347506043284638, 0.01981329596221728, 0.013110951307862065, 0.010850689564463507, 0.014578043104805094, 0.011923835944163408, 0.016093697121362327, 0.01113576723367013, 0.013056583636326696, 0.015471192713592097, 0.019914096379239146, 0.020421025255710573, 0.012253706376115912]
    # shuffle_diff = [0.00315207781713632, 0.003470229497558888, 0.0021011370483291075, -0.0016783489019132833, 0.0013467403682689907, -0.0021742507006386608, -0.001418196035618352, -0.00155777575554511, -0.001270533614032135, -0.0013536421766883744, 0.006282928887969899, 0.003836860741354947, -0.0024718812650539046]
    # t_stat, p_value = ttest_ind(original_diff, shuffle_diff)
    # print("P-Value: ", p_value)
    ########

    # dicts = [corrs1, corrs2, corrs3, corrs4, corrs5, corrs6]
    # diffs_by_dist = {}
    # stats_by_dist = {}
    # start_idx = 3
    # for dist, dict_corr in enumerate(dicts, start=1):
    #     diff_1_0 = []
    #     diff_2_0 = []
    #     diff_3_0 = []
    #     for val in dict_corr.values():
    #         lag = val[0]
    #         corr = val[1]
    #         corr1 = corr[start_idx:start_idx + 2]
    #         diff_1_0.append(max(corr1) - corr1[0])
    #         corr2 = corr[start_idx:start_idx + 3]
    #         diff_2_0.append(max(corr2) - corr2[0])
    #         corr3 = corr[start_idx:start_idx + 4]
    #         diff_3_0.append(max(corr3) - corr3[0])
    #     mean_std_1_0 = (np.mean(diff_1_0), np.std(diff_1_0))
    #     mean_std_2_0 = (np.mean(diff_2_0), np.std(diff_2_0))
    #     mean_std_3_0 = (np.mean(diff_3_0), np.std(diff_3_0))
    #     diffs_by_dist[dist] = (np.mean(diff_1_0), np.mean(diff_2_0), np.mean(diff_3_0))
    #     stats_by_dist[dist] = (mean_std_1_0, mean_std_2_0, mean_std_3_0)
    # distances = list(diffs_by_dist.keys())
    # differences = list(diffs_by_dist.values())
    # differences_array = np.array(differences)
    # num_lags = differences_array.shape[1]
    # plt.figure(figsize=(10, 6))
    # means = differences_array[:, 2]
    # stds = [stats_by_dist[dist][2][1] for dist in distances]
    # plt.errorbar(distances, means, yerr=stds, fmt='o', capsize=5, alpha=0.7)
    # y_min = np.min(means - stds)
    # y_max = np.max(means + stds)
    # y_ticks = np.arange(np.floor(y_min * 100) / 100, np.ceil(y_max * 100) / 100 + 0.01, 0.01)
    # plt.yticks(y_ticks)
    # # for lag in range(num_lags):
    # #     means = differences_array[:, lag]
    # #     stds = [stats_by_dist[dist][lag][1] for dist in distances]
    # #     # plt.plot(distances, differences_array[:, lag], marker='o', label=f'Diff(max_corr_lag{lag + 1}-corr_lag0)')
    # #     plt.errorbar(distances, means, yerr=stds, fmt='o', capsize=5, label=f'Diff(max_corr_lag{lag + 1}-corr_lag0)',
    # #                  alpha=0.7)
    # #     # for i, (mean, std) in enumerate(zip(means, stds)):
    # #     #     plt.annotate(f'{mean:.3f}±{std:.3f}', (distances[i], means[i]), textcoords="offset points",
    # #     #                  xytext=(0, 10), ha='center')
    # plt.xlabel('Topological Distance')
    # plt.ylabel('Differences in Correlation (max_corr_in_lag_range - corr_lag_0)')
    # # plt.title('Differences in Correlation for Different Lags vs. Topological Distance')
    # plt.title('Differences in Correlation for Max Lag 3 vs. Topological Distance')
    # # plt.legend()
    # plt.grid(True)
    # plt.show()


    # lag_correlations = defaultdict(lambda: defaultdict(list))
    # for dist, d in enumerate(dicts, start=1):
    #     for key, (lags, correlations) in d.items():
    #         for lag, correlation in zip(lags, correlations):
    #             lag_correlations[lag][dist].append(correlation)
    # mean_correlations = {lag: {dist: np.mean(values) for dist, values in dists.items()} for lag, dists in
    #                      lag_correlations.items()}
    # plt.figure(figsize=(10, 6))
    # for lag, dists in mean_correlations.items():
    #     # Sorting the distances to ensure the line plot is ordered
    #     sorted_distances = sorted(dists.items())  # Sort by topological distance
    #     distances = [x[0] for x in sorted_distances]
    #     means = [x[1] for x in sorted_distances]
    #     plt.plot(distances, means, marker='o', label=f'Lag {lag}')
    # plt.title('Mean Correlation by Topological Distance for Each Lag')
    # plt.xlabel('Topological Distance')
    # plt.ylabel('Mean Correlation')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # histogram_for_53_exps_time_distribution()
    # sns.histplot(data=all_nbrs_corrs, label="nbrs corrs", kde=True, alpha=0.3, stat='density')
    # sns.histplot(data=all_non_nbrs_corrs, label="non-neighbors corrs", kde=True, alpha=0.3, stat='density')
    # plt.xlabel("Correlations")
    # plt.title("Neighbors & Non-Neighbors Correlation Histogram")
    # print("avg nbrs corrs:", np.mean(all_nbrs_corrs))
    # print("avg non-nbrs corrs:", np.mean(all_non_nbrs_corrs))
    # t_stat, p_value = ttest_ind(all_nbrs_corrs, all_non_nbrs_corrs)
    # print("P-Value: ", p_value)
    # plt.legend()
    # plt.show()
    # correlation_plot()
    # df_long = pd.melt(all_data_df, id_vars=['id', 'type'], value_vars=np.arange(120).astype(str),
    #                                     var_name='Time', value_name='Intensity')
    # df_long['Time'] = df_long['Time'].astype(int)
    # # specific_ids = ['20230323-08-2(124, 203)', '20230327-07-2(148, 220)']
    # # df_specific_ids = df_long[df_long['id'].isin(specific_ids)]
    # # sns.lineplot(data=df_specific_ids, x='Time', y='Intensity', hue='type', ci=None)
    # # plt.title('Norm Intensity Over Time for Specific IDs')
    # # plt.show()
    # sns.lineplot(data=df_long, x='Time', y='Intensity', hue='type', ci=None)
    # plt.title('Norm Intensity Over Time by Type - all pillars of 2 cells')
    # plt.show()

    ###### fig S9 #####
    # avg_nbrs_corr_after = np.mean(nbrs_corrs_after_blebb)
    # avg_nbrs_corr_before = np.mean(nbrs_corrs_before_blebb)
    # print("Average neighbors correlations before blebbistatin:", avg_nbrs_corr_before)
    # print("Average neighbors correlations after blebbistatin:", avg_nbrs_corr_after)
    # diff_nbrs_nonbrs_before = np.array(nbrs_corrs_before_blebb) - np.array(non_nbrs_corrs_before_blebb)
    # print("Number of cells over the diagonal before blebb:", np.sum(diff_nbrs_nonbrs_before > 0), "/", len(diff_nbrs_nonbrs_before))
    # diff_nbrs_nonbrs_after = np.array(nbrs_corrs_after_blebb) - np.array(non_nbrs_corrs_after_blebb)
    # print("Number of cells over the diagonal after blebb:", np.sum(diff_nbrs_nonbrs_after > 0), "/", len(diff_nbrs_nonbrs_after))
    # t_stat, p_value = ttest_ind(nbrs_corrs_before_blebb, nbrs_corrs_after_blebb)
    # print("P-Value t-test: ", p_value)
    # # before_mean = [0.5955, 0.6795, 0.495, 0.3765, 0.339, 0.5, 0.311, 0.33]
    # # after_mean = [0.4395, 0.496, 0.252, 0.1905, 0.22, 0.3605, 0.2965, 0.3995]
    # # diff_corrs = np.array(after_mean) - np.array(before_mean)
    # # w_stat, w_p_value = stats.wilcoxon(diff_corrs)
    # # print("p-value for wilcoxon rank test:", w_p_value)
    # plot_average_correlation_neighbors_vs_non_neighbors(non_nbrs_corrs, nbrs_corrs, labels=exps_type, colors=['tab:blue', 'tab:orange'], title="Average Correlations Experiments Before vs. After Inhibitor", save_fig=True, fig_name='nbrs_nonbrs_before_after_inhibitor_53_fig_s9')

    # X_type = all_data_df.drop(['id'], axis=1)
    # X = all_data_df.drop(['id', 'type'], axis=1)
    # for i, row in X_type.iterrows():
    #     if row.loc['type'] == '5.3':
    #         plt.plot(X.iloc[i], c='tab:blue', alpha=0.1)
    #     if row.loc['type'] == '13.2':
    #         plt.plot(X.iloc[i], c='tab:orange', alpha=0.1)
    # plt.show()

    # for i, ts in enumerate(cells_ts):
    #     if exps_type[i] == '5.3':
    #         plt.plot(ts, c='tab:blue', alpha=0.3)
    #     if exps_type[i] == '13.2':
    #         plt.plot(ts, c='tab:orange', alpha=0.3)
    # plt.show()
    # df_long = pd.melt(all_data_df, id_vars=['id', 'type'], value_vars=np.arange(120).astype(str),
    #                   var_name='Time', value_name='Intensity')
    # df_long['Time'] = df_long['Time'].astype(int)
    # stats_5_3 = df_long[df_long['type'] == '5.3']['Intensity'].describe()
    # stats_13_2 = df_long[df_long['type'] == '13.2']['Intensity'].describe()
    # print("Summary statistics for type 5.3:\n", stats_5_3)
    # print("\nSummary statistics for type 13.2:\n", stats_13_2)
    # sns.lineplot(data=df_long, x='Time', y='Intensity', hue='type', ci=None)
    # plt.title('Intensity Over Time by Type')
    # plt.show()
    # sns.boxplot(x='type', y='Intensity', data=df_long)
    # plt.title('Distribution of Intensities by Type')
    # plt.show()
    # x=1

    ################# t-SNE ####################
    # time_series_tsne_for_type(all_data_df)
    # time_series_tsne_for_location(all_data_df)
    # time_series_tsne_for_blebb(all_data_df)


    # significants = [1 if p < 0.1 else 0 for p in similarity_nbrs_vs_nonbrs]
    # print("number of significant results:", np.sum(significants), "/", len(significants))
    # plot_average_correlation_neighbors_vs_non_neighbors(non_nbrs_corrs, nbrs_corrs, labels=exps_type)
    # nbrs_corrs_53 = [float(c) for i, c in enumerate(nbrs_corrs) if exps_type[i] == '5.3']
    # non_nbrs_corrs_53 = [float(c) for i, c in enumerate(non_nbrs_corrs) if exps_type[i] == '5.3']
    # nbrs_corrs_132 = [float(c) for i, c in enumerate(nbrs_corrs) if exps_type[i] == '13.2']
    # non_nbrs_corrs_132 = [float(c) for i, c in enumerate(non_nbrs_corrs) if exps_type[i] == '13.2']
    # # diff_53 = np.array(nbrs_corrs_53) - np.array(non_nbrs_corrs_53)
    # # diff_132 = np.array(nbrs_corrs_132) - np.array(non_nbrs_corrs_132)
    # t_stat_53, p_value_53 = ttest_ind(nbrs_corrs_53, non_nbrs_corrs_53)
    # print("p-value 5.3 nbrs and non-nbrs:", p_value_53)
    # t_stat_132, p_value_132 = ttest_ind(nbrs_corrs_132, non_nbrs_corrs_132)
    # print("p-value 13.2 nbrs and non-nbrs:", p_value_132)

    ##### fig 2A #####
    # plot_average_correlation_neighbors_vs_non_neighbors(non_nbrs_corrs, nbrs_corrs, labels=exps_type, title='Average Correlation - 5.3 vs 13.2', save_fig=True, fig_name='53_vs_132_corrs_fig_2a')

    # nbrs53 = [a for i, a in enumerate(avg_nbrs_sim_lst) if exps_type[i] == '5.3']
    # nbrs132 = [a for i, a in enumerate(avg_nbrs_sim_lst) if exps_type[i] == '13.2']
    # print("Average 5.3 nbrs similarity:", "%.3f" % np.mean(nbrs53))
    # print("Average 13.2 nbrs similarity:", "%.3f" % np.mean(nbrs132))
    # t_stat, p_value = ttest_ind(nbrs53, nbrs132)
    # print("P-Value: ", p_value)

    # for i, dict in enumerate(lag_avg_corrs):
    #     lags = dict.keys()
    #     avg_corrs = dict.values()
    #     plt.plot(lags, avg_corrs, label=exps[i])
    # plt.axvline(x=0, color="gainsboro", linestyle="--")
    # plt.axhline(y=0, color="gainsboro", linestyle="--")
    # plt.xlabel('lag')
    # plt.ylabel('avg correlation')
    # plt.title('Avg Correlation in Lag')
    # plt.legend(title='Experiment')
    # plt.show()
    # plot_average_correlation_neighbors_vs_non_neighbors(periphery_avgs, core_avgs, labels=exps,
    #                                                     title='Experiment Avg Similarity Core vs. Periphery',
    #                                                     xlabel='Periphery', ylabel='Core')
    # print("Average core similarity:", "%.3f" % np.mean(core_avgs))
    # print("Average periphery similarity:", "%.3f" % np.mean(periphery_avgs))
    # t_stat, p_value = ttest_ind(core_avgs, periphery_avgs)
    # print("P-Value: ", p_value)

    # sns.histplot(core_corrs, label="core", kde=True, alpha=0.3)
    # sns.histplot(periphery_corrs, label="core", kde=True, alpha=0.3)
    # plt.show()
    # print("Average core:", "%.3f" % np.mean(core_corrs))
    # print("Average periphery:", "%.3f" % np.mean(periphery_corrs))
    # print("Average border:", "%.3f" % np.mean(border_corrs))
    # t_stat, p_value = ttest_ind(core_corrs, periphery_corrs)
    # print("P-Value: ", p_value)

    ##### fig 1F #####
    # max_list_values = max([len(v) for v in all_corrs.values()])
    # cut_dict = {key: value for key, value in all_corrs.items() if
    #             len(value) >= max_list_values * 0.5 or key == 1 or key <= 6}
    # correlation_between_nbrhood_level_to_avg_correlation(cut_dict)
    # plot_avg_correlation_by_nbrhood_degree(cut_dict, save_fig=True)
    ##### fig 1F new #####
    # cut_dict = {key: value for key, value in all_corrs.items() if key <= 6}
    # medians = [np.median(v) for v in cut_dict.values()]
    # means = [np.mean(v) for v in cut_dict.values()]
    # print("Number of cells at each distance:", [len(v) for k, v in cut_dict.items()])
    # print("Means of each distance:", means)
    # all_corrs_six_levels_df = pd.DataFrame(list(cut_dict.items()), columns=['topological_distance', 'correlation'])
    # df_expanded = all_corrs_six_levels_df.explode('correlation')
    # box_plot = sns.boxplot(data=df_expanded, y='correlation', x='topological_distance', color=".8")
    # for i in range(len(medians)):
    #     box_plot.annotate(str("%.3f" % medians[i]), xy=(i+0.2, medians[i]), ha='right', color='black', fontsize="8")
    # for key1 in cut_dict:
    #     for key2 in cut_dict:
    #         if key1 < key2:
    #             # Perform the t-test for independent samples
    #             stat_ttest, p_value_ttest = stats.ttest_ind(cut_dict[key1], cut_dict[key2])
    #             print(f"T-test between {key1} and {key2}: p-value = {p_value_ttest}")
    # topological_distances = []
    # correlations = []
    # for distance, corr_list in cut_dict.items():
    #     topological_distances.extend([distance] * len(corr_list))
    #     correlations.extend(corr_list)
    # pearson_corr, p_value = pearsonr(topological_distances, correlations)
    # print(f"Pearson correlation: {pearson_corr}, P-value: {p_value}")
    # plt.savefig('../top_distance_boxplot_1f_new.svg', format="svg")
    # plt.show()
    ###### for significant in each distace before and after shuffle - 1f vs S6 ######
    # cell_to_distance_to_mean_corr = {}
    # for cell, distances in map_cell_to_dist_corrs.items():
    #     cell_to_distance_to_mean_corr[cell] = {distance: sum(correlations) / len(correlations)
    #                     for distance, correlations in distances.items()}
    # print(cell_to_distance_to_mean_corr)
    #### wilcoxon test on the diffs of original pillars vs shuffle in cell pillars
    # original_dict = {'20230320-5': {1: 0.22503654730639147, 2: 0.10723211573577573, 3: 0.10032120151109378, 4: 0.09114147021533622, 5: 0.09597892531265195, 6: 0.08236430222875896, 7: 0.07895370523793194, 8: 0.05705169534234702, 9: 0.043436538214417494}, '20230320-6': {1: 0.25984114060068536, 2: 0.15202299392848773, 3: 0.12472479057865636, 4: 0.10341219608559205, 5: 0.0918991334483285, 6: 0.0845953322084087, 7: 0.06057759722135544, 8: 0.04435127849778822, 9: 0.010816911905839907, 10: 0.07310865255955226}, '20230320-7': {1: 0.4043782239559096, 2: 0.33811778455484254, 3: 0.28852333540338065, 4: 0.24608609119249877, 5: 0.21280099202929081, 6: 0.17928304704116063, 7: 0.0649970069699358, 8: 0.05264456895621299}, '20230320-8': {1: 0.2712914575594924, 2: 0.19066477450562563, 3: 0.16965120221949936, 4: 0.15027295192653492, 5: 0.14547091923117664, 6: 0.1413290820685724}, '20230320-9': {1: 0.1910102175771357, 2: 0.1309737792936859, 3: 0.1209920882913678, 4: 0.08259173502233398, 5: 0.07473589890587791}, '20230323-1': {1: 0.29553940462694395, 2: 0.21495715477731842, 3: 0.1830149041042908, 4: 0.19412273592169269, 5: 0.16954354842471772, 6: 0.07355366375657338}, '20230323-2': {1: 0.2982276514406073, 2: 0.18772584586399121, 3: 0.15055964711361003, 4: 0.13240236070221456, 5: 0.1222002940502894, 6: 0.11412464704737829, 7: 0.09534873726455159, 8: 0.004641932021561539}, '20230323-3': {1: 0.16994007108549247, 2: 0.0654795548846897, 3: 0.0592675745805424, 4: 0.06438698357012893, 5: 0.05560658984803742, 6: 0.012922562724342945, 7: -0.08589752674283102}, '20230323-4': {1: 0.2751978651441353, 2: 0.18767596444007972, 3: 0.17123980901618413, 4: 0.18083086081484712, 5: 0.17400849796989062, 6: 0.17827070751180915}, '20230323-5': {1: 0.29565531596886935, 2: 0.22417423419916427, 3: 0.2169929603899525, 4: 0.2113102148907313, 5: 0.21714722318126334, 6: 0.2252539208112331, 7: 0.13378761028256014, 8: 0.14744372027656785, 9: 0.11317361766134451}, '20230323-6': {1: 0.3029028527492706, 2: 0.21249952409552522, 3: 0.19243769948875888, 4: 0.17402642970311683, 5: 0.15392774068469386, 6: 0.13236218673198868, 7: 0.07637732746444077, 8: -0.013367321881219151}, '20230323-7': {1: 0.3992898924721755, 2: 0.32337096844913404, 3: 0.31207013950839496, 4: 0.3133261145126943, 5: 0.26363851063498656, 6: 0.19339365590724567}, '20230323-8': {1: 0.3763000434657713, 2: 0.25548551424804405, 3: 0.20483914725333557, 4: 0.19165410308410846, 5: 0.17360014434703502, 6: 0.17377592031328126, 7: 0.18397622686601797, 8: 0.16843757602948908, 9: -0.036769783945195395, 10: -0.047804019737084695}}
    # shuffle_dict = {'20230320-5': {1: 0.08665009164370154, 2: 0.0904065795742828, 3: 0.085234155215305, 4: 0.086034858037695, 5: 0.08624637948266407, 6: 0.08754636376195261, 7: 0.09259279689296765, 8: 0.0910806399985463, 9: 0.08537682707289432}, '20230320-6': {1: 0.07526411191274923, 2: 0.07840577667129427, 3: 0.07525942353597015, 4: 0.07801383801437199, 5: 0.07803747186541393, 6: 0.07727127919495036, 7: 0.07466439142054289, 8: 0.07359457645050707, 9: 0.04264352077706614, 10: 0.023979772860058715}, '20230320-7': {1: 0.21550818552425097, 2: 0.22484723269581658, 3: 0.21924320080558446, 4: 0.21857516050482392, 5: 0.21687636214594072, 6: 0.21931547519433364, 7: 0.13512004276341408, 8: 0.062478601718201}, '20230320-8': {1: 0.14980200824119647, 2: 0.14848070386418505, 3: 0.14467148207931635, 4: 0.14462162837020381, 5: 0.14643420948965907, 6: 0.14215474018240254}, '20230320-9': {1: 0.09987689660850296, 2: 0.09996079053228013, 3: 0.10281915930618094, 4: 0.10238708002648596, 5: 0.10233634254083797}, '20230323-1': {1: 0.19276348848289512, 2: 0.19154087198259254, 3: 0.19082568199073102, 4: 0.18746306721059222, 5: 0.1832209801265099, 6: 0.15060726945123754}, '20230323-2': {1: 0.13839470799379433, 2: 0.1408165135465747, 3: 0.14367955046880274, 4: 0.13650641132446975, 5: 0.13874101375111253, 6: 0.13582184704464662, 7: 0.1349302219208141, 8: 0.1257973990411691}, '20230323-3': {1: 0.04309910820076598, 2: 0.04410572210887193, 3: 0.04140859853670647, 4: 0.04707167547845309, 5: 0.04541587313780152, 6: 0.008551081553666993, 7: -0.001480388113205671}, '20230323-4': {1: 0.18317791450549292, 2: 0.18671950957847339, 3: 0.1855662782012636, 4: 0.18655443446959724, 5: 0.18621480682505764, 6: 0.18514452572890852}, '20230323-5': {1: 0.16738694880625304, 2: 0.16802315568423504, 3: 0.17293126172890427, 4: 0.16777374843281873, 5: 0.1723432578473528, 6: 0.1738961425329822, 7: 0.10835702258616378, 8: 0.12184422649012572, 9: 0.17827943700825283}, '20230323-6': {1: 0.11500915628580703, 2: 0.11869778082316132, 3: 0.11800059313063897, 4: 0.11759977875078786, 5: 0.11515927474209355, 6: 0.11360623611208427, 7: 0.04842851671399429, 8: 0.03197877438596029}, '20230323-7': {1: 0.24904987974762885, 2: 0.2356928807903797, 3: 0.2420946223921914, 4: 0.24985071821624494, 5: 0.24236801306703915, 6: 0.19328302183100873}, '20230323-8': {1: 0.1647212811454815, 2: 0.16293896930619012, 3: 0.16223071287001656, 4: 0.15890546006194115, 5: 0.15971920187127908, 6: 0.15831862188483567, 7: 0.1598169214328397, 8: 0.11308912647616134, 9: 0.022682118959445507, 10: 0.030516414074833954}}
    # diffs_dist = []
    # dist = 5
    # for cell in original_dict:
    #     if dist in original_dict[cell] and cell in shuffle_dict and dist in shuffle_dict[cell]:
    #         original_value = original_dict[cell][dist]
    #         shuffle_value = shuffle_dict[cell][dist]
    #         difference = original_value - shuffle_value
    #         diffs_dist.append(difference)
    # if len(diffs_dist) > 1:
    #     stat, p_value = stats.wilcoxon(diffs_dist)
    #     print("p-value:", p_value)
    # def prepare_data_for_plot(data_dict, label):
    #     rows = []
    #     for cell, distances in data_dict.items():
    #         if dist in distances:  # Only take data from distance 2
    #             rows.append({'Cell': cell, 'Correlation': distances[dist], 'Type': label})
    #     return rows
    # original_data = prepare_data_for_plot(original_dict, 'Observed')
    # shuffle_data = prepare_data_for_plot(shuffle_dict, 'Shuffle')
    # df = pd.DataFrame(original_data + shuffle_data)
    # cells = df['Cell'].unique()
    # num_cells = len(cells)
    # mean_original = df[df['Type'] == 'Observed']['Correlation'].mean()
    # median_original = df[df['Type'] == 'Observed']['Correlation'].median()
    # mean_shuffle = df[df['Type'] == 'Shuffle']['Correlation'].mean()
    # median_shuffle = df[df['Type'] == 'Shuffle']['Correlation'].median()
    # print(f"Observed - Mean: {mean_original}, Median: {median_original}")
    # print(f"Shuffle - Mean: {mean_shuffle}, Median: {median_shuffle}")
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(x='Type', y='Correlation', data=df, palette=['white', 'white'], showmeans=False, color='lightgrey', boxprops={'facecolor': 'None'})
    # sns.stripplot(x='Type', y='Correlation', data=df, hue='Cell', jitter=True, dodge=True, palette=sns.color_palette("Spectral", num_cells), marker="o", size=8)
    # # plt.title('Correlations at Distance', dist, 'Original vs. Shuffle', fontsize=14)
    # plt.ylabel('Correlation', fontsize=12)
    # plt.legend(title='Cell', bbox_to_anchor=(1.05, 1), loc='upper left')
    # ymin, ymax = plt.ylim()
    # plt.ylim(ymin, ymax + 0.05)
    # plt.tight_layout()
    # plt.savefig('../observed_vs_shuffle_dist5_fig_S8.jpg', format="jpg")
    # plt.show()

    # nbrs_sim = [tup[0] for tup in list(p_2_sim_dict.values())]
    # non_nbrs_sim = [tup[1] for tup in list(p_2_sim_dict.values())]
    # plot_average_correlation_neighbors_vs_non_neighbors(avg_non_nbrs_sim_lst, avg_nbrs_sim_lst, labels=exps_type,
    #                                                     title='Experiment Avg Similarity Neighbors vs. Non-Neighbors',
    #                                                     xlabel='Non-Neighbors avg similarity', ylabel='Neighbors avg similarity')
    # print("Average neighbors similarity:", "%.3f" % np.mean(avg_nbrs_sim_lst))
    # print("Average non-neighbors similarity:", "%.3f" % np.mean(avg_non_nbrs_sim_lst))
    # t_stat, p_value = ttest_ind(avg_nbrs_sim_lst, avg_non_nbrs_sim_lst)
    # print("###### t-test 13.2 originals vs permutations clusters #####")
    # print("T-statistic value: ", "%.2f" % t_stat)
    # print("P-Value: ", p_value)

    ######################## Individuall cell strength vs intensity and super strong pillars TS ########################
    # min_value = min(pillars_avg_raw_intens.values())
    # max_value = max(pillars_avg_raw_intens.values())
    # pillars_avg_intens = {}
    # for k, v in pillars_avg_raw_intens.items():
    #     norm_value = (v - min_value) / (max_value - min_value)
    #     pillars_avg_intens[k] = norm_value
    # pillars_lst = list(pillars_avg_intens.keys())
    # avg_strength = np.mean(list(pillars_strength.values()))
    # avg_norm_intens = np.mean(list(pillars_avg_intens.values()))
    # labels = []
    # pillars_above_avgs = []
    # for p in pillars_lst:
    #     if pillars_strength[p] > avg_strength and pillars_avg_intens[p] > avg_norm_intens:
    #         pillars_above_avgs.append(p)
    #         labels.append('higher')
    #     else:
    #         labels.append('lower')
    # plot_average_correlation_neighbors_vs_non_neighbors(list(pillars_strength.values()), list(pillars_avg_intens.values()), labels=labels,
    #                                                     title='Pillars Avg Strength vs. Avg Intensity - 20230327-05-4',
    #                                                     xlabel='Avg strength', ylabel='Avg Intensity', arg1=avg_strength, arg2=avg_norm_intens)
    # G = build_pillars_graph(random_neighbors=False, shuffle_ts=False, draw=False)
    # ns, strong_nodes, _ = nodes_strengths(G, draw=True, color_map_nodes=False, group_nodes_colored=pillars_above_avgs)
    #
    # avg_strength = np.mean(list(pillars_strength.values()))
    # for p, ts in p_to_intens.items():
    #     if pillars_avg_raw_intens[p] > total_avg_intensity and pillars_strength[p] > avg_strength:
    #         plt.plot(ts, color='grey')
    # plt.plot(avg_ts, linestyle='--', color='black', label="avg TS")
    # plt.title("Super strong pillars time series")
    # plt.xlabel("Frame")
    # plt.ylabel("Raw Intensity")
    # plt.legend()
    # plt.show()

    # with open('../map_vid_to_original_and_shuffle_ts_500_avg_core_periphery_sim_diff_with_pval_5.3_20230323_cell_2_exps', 'wb') as f:
    #     pickle.dump(map_exp_to_statistics_original_and_random_avg_similarity, f)
    # with open('../map_vid_to_original_and_shuffle_ts_500_avg_core_periphery_sim_diff_with_pval_5.3_20230323_cell_2_exps', 'rb') as file:
    #     loaded_dict_53 = pickle.load(file)
    # with open('../map_vid_to_original_and_shuffle_ts_500_avg_sim_with_pval_5.3_exps', 'rb') as file:
    #     loaded_dict_132 = pickle.load(file)
    #     x=1
    #
    # significant = 0
    # no_significant = 0
    # for v in loaded_dict_132.values():
    #     if v[2] < 0.05:
    #         significant += 1
    #     else:
    #         no_significant += 1
    #
    # cat = ['significant', 'not significant']
    # vals = [significant, no_significant]
    # plt.bar(cat, vals)
    # for i, v in enumerate(vals):
    #     plt.text(i, v, str(v), ha='center', va='bottom')
    # plt.ylabel("Amount")
    # plt.title("Number of significant results permutations of average strength similarity - 5.3 exps")
    # plt.show()
    # originals = []
    # permuted = []
    # for v in loaded_dict_132.values():
    #     avg_permuted = np.mean(v[1])
    #     permuted.append(avg_permuted)
    #     originals.append(v[0])
    # sns.histplot(originals, label="originals", kde=True, alpha=0.3)
    # sns.histplot(permuted, label="permutations", kde=True, alpha=0.3)
    # plt.title("Distribution of Average Strength Similarity Original vs. Permutations (shuffle ts) - 5.3 Exps")
    # plt.xlabel('Average Strength Similarity')
    # plt.legend()
    # plt.show()
    # print("Average pillars strength similarity in 13.2 exps - original ts:", "%.3f" % np.mean(originals))
    # print("Average pillars strength similarity in 13.2 exps - permuted ts:", "%.3f" % np.mean(permuted))
    # t_stat, p_value = ttest_ind(originals, permuted)
    # print("###### t-test 13.2 originals vs permutations clusters #####")
    # print("T-statistic value: ", "%.2f" % t_stat)
    # print("P-Value: ", p_value)

    # mask_radiuses_tuples = [(0,10), (0,15), (15,35)]
    # for mask_radius in mask_radiuses_tuples:
    #     ratio_radiuses = get_mask_radiuses({'small_radius': mask_radius[0], 'large_radius': mask_radius[1]})
    #     Consts.SMALL_MASK_RADIUS = ratio_radiuses['small']
    #     Consts.LARGE_MASK_RADIUS = ratio_radiuses['large']
    #
    #     p_to_intens = get_pillar_to_intensities(get_images_path(), use_cache=False)
    #     time = [i * 30.02 for i in range(len(p_to_intens[(28,73)]))]
    #     plt.plot(time, p_to_intens[(28,73)])
    # plt.show()

    ##### fig 2B + inset #####
    ######## Plot correlations delta before and after blebb ########
    # lbls = []
    # type = []
    # delta_before = []
    # delta_after = []
    # diff_delta_after_before_53 = []
    # diff_delta_after_before_132 = []
    # for exp_k, exp_v in map_exp_to_delta_corrs.items():
    #     for cell_k, cell_v in exp_v.items():
    #         if 'after' not in cell_v:
    #             continue
    #         d_before = np.mean(cell_v['before'])
    #         d_after = np.mean(cell_v['after'])
    #         delta_before.append(d_before)
    #         delta_after.append(d_after)
    #         if exp_name_to_exp_type[exp_k] == '5.3':
    #             diff_delta_after_before_53.append(np.array(d_after) - np.array(d_before))
    #         else:
    #             diff_delta_after_before_132.append(np.array(d_after) - np.array(d_before))
    #         lbls.append(exp_k + '-' + cell_k)
    #         type.append(exp_name_to_exp_type[exp_k])
    # t_stat, p_value = ttest_ind(delta_before, delta_after)
    # print("P-Value: ", p_value)
    # # plot_average_correlation_neighbors_vs_non_neighbors(delta_before, delta_after, labels=lbls,
    # #                                                     xlabel='delta correlations before', ylabel='delta correlations after',
    # #                                                     title='Neighbors & Non-Neighbors Delta Correlations Before & After Blebbistatin (5.3)')
    # print("Number of cells over the diagonal for 5.3:", np.sum(np.array(diff_delta_after_before_53) > 0), "/", len(diff_delta_after_before_53))
    # print("Number of cells over the diagonal for 13.2:", np.sum(np.array(diff_delta_after_before_132) > 0), "/",
    #       len(diff_delta_after_before_132))
    # plot_average_correlation_neighbors_vs_non_neighbors(delta_before, delta_after, labels=type,
    #                                                     xlabel='Correlation difference before', ylabel='Correlation difference after', save_fig=True, fig_name='delta_corrs_fig_2b_formin.svg')
    # # t_stat, p_value = ttest_ind(diff_delta_after_before_53, diff_delta_after_before_132)
    # # print("P-Value for inset: ", p_value)
    # plt.rcParams.update({'font.size': 12})
    # plt.rcParams['font.family'] = 'Arial'
    # plt.figure(figsize=(8, 4))
    # sns.distplot(diff_delta_after_before_53, kde=True, label='5.3', color='tab:green')
    # sns.distplot(diff_delta_after_before_132, kde=True, label='13.2', color='tab:red')
    # plt.axvline(x=0, color="gainsboro", linestyle="--")
    # plt.xlabel("Difference of (Δcorrelations_before - Δcorrelations_after)")
    # plt.yticks(rotation=90)
    # plt.legend()
    # # plt.savefig('../diff_delta_corrs_fig_2B_inset.svg', format="svg")
    # plt.show()

    # t_stat, p_value_nbrs_vs_non = ttest_ind(nbrs_corrs, non_nbrs_corrs)
    # print("p-value for nbrs vs non-nbrs before after blebbb:", p_value_nbrs_vs_non)
    # diff = np.array(nbrs_corrs) - np.array(non_nbrs_corrs)
    # print("Number of cells over the diagonal:", np.sum(diff > 0), "/", len(diff))
    # t_stat, p_value = ttest_ind(nbrs_corrs_before_blebb, nbrs_corrs_after_blebb)
    # print("P-Value: ", p_value)
    # plot_average_correlation_neighbors_vs_non_neighbors(non_nbrs_corrs, nbrs_corrs, labels=exps_type, colors=['tab:blue', 'tab:orange'], title="Average Correlations 5.3 Experiments Before vs. After Blebbistatin", save_fig=True, fig_name='nbrs_vs_nonbrs_before_after_53.svg')



    #### cell strong nodes group avg distance from cell center through time ####
    # cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8]
    # labels = ['cell 1', 'cell 2', 'cell 3', 'cell 4', 'cell 5', 'cell 6', 'cell 7', 'cell 8']
    # fig, ax = plt.subplots()
    # for i in range(len(cells)):
    #     ax.plot(cells[i], label=labels[i], linestyle='dashed', marker='o')
    #
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Strong pillars avg distance from cell center')
    # ax.set_title("Exp 13.2 20230327 cells' strong pillars avg distance from cell center in time")
    # ax.legend()
    # plt.show()

    #### node strength in a pillars graph ####
    # print("mean strength nodes group distance from center in 20230809 exp before blebb", np.mean(avg_strenght_group_dist_from_center_20230809_before_blebb))
    # print("mean strength nodes group distance from center in 20230809 exp after blebb", np.mean(avg_strenght_group_dist_from_center_20230809_after_blebb))
    # print("mean strength nodes group distance from center in 2023081501 exp before blebb", np.mean(avg_strenght_group_dist_from_center_2023081501_before_blebb))
    # print("mean strength nodes group distance from center in 2023081501 exp after blebb", np.mean(avg_strenght_group_dist_from_center_2023081501_after_blebb))
    # print("mean strength nodes group distance from center in 2023081502 exp before blebb", np.mean(avg_strenght_group_dist_from_center_2023081502_before_blebb))
    # print("mean strength nodes group distance from center in 2023081502 exp after blebb", np.mean(avg_strenght_group_dist_from_center_2023081502_after_blebb))
    # print("mean strength nodes group distance from center in 20230818 exp before blebb", np.mean(avg_strenght_group_dist_from_center_20230818_before_blebb))
    # print("mean strength nodes group distance from center in 20230818 exp after blebb", np.mean(avg_strenght_group_dist_from_center_20230818_after_blebb))
    # t_stat, p_value = ttest_ind(avg_strenght_group_dist_from_center_20230818_before_blebb, avg_strenght_group_dist_from_center_20230818_after_blebb)
    # print("###### t-test 20230818 before and after blebb - distances from center #####")
    # print("T-statistic value: ", "%.2f" % t_stat)
    # print("P-Value: ", "%.2f" % p_value)
    # sns.histplot(avg_strenght_group_dist_from_center_20230818_before_blebb, label="20230818 - before blebb", kde=True, alpha=0.3)
    # sns.histplot(avg_strenght_group_dist_from_center_20230818_after_blebb, label="20230818 - after blebb", kde=True, alpha=0.3)
    # plt.title("Distribution of Strong Pillars Distance From Cell Center - 20230818 before & after blebb")
    # plt.xlabel('Distance from Cell Center')
    # plt.legend()
    # plt.show()

    # exps_before_blebb = avg_strenght_group_dist_from_center_20230809_before_blebb + avg_strenght_group_dist_from_center_2023081501_before_blebb + \
    #                     avg_strenght_group_dist_from_center_2023081502_before_blebb + avg_strenght_group_dist_from_center_20230818_before_blebb
    # exps_after_blebb = avg_strenght_group_dist_from_center_20230809_after_blebb + avg_strenght_group_dist_from_center_2023081501_after_blebb + \
    #                    avg_strenght_group_dist_from_center_2023081502_after_blebb + avg_strenght_group_dist_from_center_20230818_after_blebb
    # print("mean strength nodes group distance from center before blebb", np.mean(exps_before_blebb))
    # print("mean strength nodes group distance from center after blebb", np.mean(exps_after_blebb))
    # t_stat, p_value = ttest_ind(exps_before_blebb, exps_after_blebb)
    # print("###### t-test before and after blebb - distances from center #####")
    # print("T-statistic value: ", "%.2f" % t_stat)
    # print("P-Value: ", "%.2f" % p_value)
    # sns.histplot(exps_before_blebb, label="before blebb", kde=True, alpha=0.3)
    # sns.histplot(exps_after_blebb, label="after blebb", kde=True, alpha=0.3)
    # plt.title("Distribution of Strong Pillars Distance From Cell Center - before & after blebb")
    # plt.xlabel('Distance from Cell Center')
    # plt.legend()
    # plt.show()

    # dist_values_before = [item[0] for item in exps_before_blebb]
    # strength_values_before = [item[1] for item in exps_before_blebb]
    # lbls_before = ['before blebb' for i in range(len(dist_values_before))]
    # dist_values_after = [item[0] for item in exps_after_blebb]
    # strength_values_after = [item[1] for item in exps_after_blebb]
    # lbls_after = ['after blebb' for i in range(len(dist_values_after))]
    #
    # dist_vals = dist_values_before + dist_values_after
    # strength_vals = strength_values_before + strength_values_after
    # lbls = lbls_before + lbls_after
    # plot_average_correlation_neighbors_vs_non_neighbors(dist_vals, strength_vals, labels=lbls,
    #                                                     xlabel='Strong nodes avg distance from center', ylabel='Avg strength of strong nodes',
    #                                                     title='Avg Strength by Distance of Strong Nodes - Before & After Blebb')

    # print("mean strength nodes group distance from center in 5.3_20230320 exp", np.mean(avg_strenght_group_dist_from_center_20230320))
    # print("mean strength nodes group distance from center in 5.3_20230323 exp", np.mean(avg_strenght_group_dist_from_center_20230323))
    # print("mean strength nodes group distance from center in 13.2_20230319 exp", np.mean(avg_strenght_group_dist_from_center_20230319))
    # print("mean strength nodes group distance from center in 13.2_20230327 exp", np.mean(avg_strenght_group_dist_from_center_20230327))
    # exp_53 = avg_strenght_group_dist_from_center_20230320 + avg_strenght_group_dist_from_center_20230323
    # exp_132 = avg_strenght_group_dist_from_center_20230319 + avg_strenght_group_dist_from_center_20230327
    # print("Average distance from cell center of the strong pillars in 5.3 exps:", "%.2f" % np.mean(exp_53))
    # print("Average distance from cell center of the strong pillars in 13.2 exps:", "%.2f" % np.mean(exp_132))
    # t_stat, p_value = ttest_ind(exp_53, exp_132)
    # print("###### t-test 5.3 and 13.2 distances from center #####")
    # print("T-statistic value: ", "%.2f" % t_stat)
    # print("P-Value: ", p_value)
    # # sns.histplot(avg_strenght_group_dist_from_center_20230320, label="5.3_20230320", kde=True, alpha=0.3, color='red')
    # # sns.histplot(avg_strenght_group_dist_from_center_20230323, label="5.3_20230323", kde=True, alpha=0.3, color='orange')
    # # sns.histplot(avg_strenght_group_dist_from_center_20230319, label="13.2_20230319", kde=True, alpha=0.3, color='green')
    # # sns.histplot(avg_strenght_group_dist_from_center_20230327, label="13.2_20230327", kde=True, alpha=0.3, color='blue')
    # sns.histplot(exp_53, label="5.3", kde=True, alpha=0.3)
    # sns.histplot(exp_132, label="13.2", kde=True, alpha=0.3)
    # plt.title("Distribution of Strong Pillars Distance From Cell Center")
    # plt.xlabel('Distance from cell center')
    # plt.legend()
    # plt.show()

    # exp_true = avg_strenght_group_dist_from_center_20230319 + avg_strenght_group_dist_from_center_20230327
    # exp_rand = avg_strenght_group_dist_from_center_20230319_rand + avg_strenght_group_dist_from_center_20230327_rand
    # exp_132 = avg_strenght_group_dist_from_center_20230319 + avg_strenght_group_dist_from_center_20230327
    # exp_132_rand = avg_strenght_group_dist_from_center_20230319_rand + avg_strenght_group_dist_from_center_20230327_rand
    # print("Average distance from cell center of the strong pillars in", exp_type, "exps - true nbrs:", "%.2f" % np.mean(exp_true))
    # print("Average distance from cell center of the strong pillars in", exp_type, "exps - random nbrs:", "%.2f" % np.mean(exp_rand))
    # t_stat, p_value = ttest_ind(exp_true, exp_rand)
    # print("###### t-test", exp_type,"true vs random nbrs distances from center #####")
    # print("T-statistic value: ", "%.2f" % t_stat)
    # print("P-Value: ", "%.3f" % p_value)
    # sns.histplot(exp_true, label="true nbrs", kde=True, alpha=0.3)
    # sns.histplot(exp_rand, label="random nbrs", kde=True, alpha=0.3)
    # plt.title("Distribution of Strong Pillars Distance From Cell Center - " + exp_type + " exps - True vs Random Neighbors")
    # plt.xlabel('Distance from Cell Center')
    # plt.legend()
    # plt.show()

    #### change in correlation by radius - control and sensitivity #### fig S5 #########
    # df = pd.DataFrame.from_dict(my_dict)
    # means = [(df[col].mean()) for col in list(df.columns)]
    # median = [(df[col].median()) for col in list(df.columns)]
    # print("means", [float("%.3f" % m) for m in means])
    # print("median", median)
    # box_plot = sns.boxplot(data=df, color=".8")
    # for i in range(len(median)):
    #     box_plot.annotate(str("%.3f" % median[i]), xy=(i+0.2, median[i]), ha='right', color='black', fontsize="8")
    # plt.ylabel("Neighbors Correlations", size=10)
    # plt.xlabel("Quantification Ring", size=10)
    # plt.title("Neighbors Correlations of Different Radius - 5.3")
    # plt.savefig('../nbrs_corrs_different_radii_fig_S5.svg', format="svg")
    # plt.show()
    # ##t-test for control##
    # rad_0_15 = df['(0, 15)']
    # rad_15_35 = df['(15, 35)']
    # t_stat, p_value = ttest_ind(rad_0_15, rad_15_35)
    # print("###### t-test (0,15) and (15,35) corrs #####")
    # print("T-statistic value: ", t_stat)
    # print("P-Value: ", p_value)
    # rad_0_10 = df['(0, 10)']
    # rad_15_35 = df['(15, 35)']
    # t_stat, p_value = ttest_ind(rad_0_10, rad_15_35)
    # print("###### t-test (0,10) and (15,35) corrs #####")
    # print("T-statistic value: ", t_stat)
    # print("P-Value: ", p_value)

    ########## control before and after norm by noise fig S3 - for experiments 20230809 before blebb ###########
    # df_after_norm = pd.DataFrame.from_dict(my_dict)
    # df_before_norm = pd.DataFrame.from_dict(my_dict_before_noise_norm)
    # means_after = [(df_after_norm[col].mean()) for col in list(df_after_norm.columns)]
    # median_after = [(df_after_norm[col].median()) for col in list(df_after_norm.columns)]
    # print("means after norm", [float("%.3f" % m) for m in means_after])
    # print("median after norm", median_after)
    # means_before = [(df_before_norm[col].mean()) for col in list(df_before_norm.columns)]
    # median_before = [(df_before_norm[col].median()) for col in list(df_before_norm.columns)]
    # print("means before norm", [float("%.3f" % m) for m in means_after])
    # print("median before norm", median_after)
    # data_to_plot = []
    # ticks = []
    # colors = []
    # linestyles = []
    # for col in df_before_norm.columns:
    #     data_to_plot.append(df_before_norm[col])
    #     data_to_plot.append(df_after_norm[col])
    #     # # Add tick labels
    #     # ticks.append(col + " Before")
    #     # ticks.append(col + " After")
    #     # Set colors and linestyles
    #     colors += ['black', 'black']  # both boxplots are black
    #     linestyles += ['-', '--']  # solid for before, dashed for after
    # # Create the boxplot
    # fig, ax = plt.subplots()
    # bplot = ax.boxplot(data_to_plot, patch_artist=True, notch=False, positions=range(1, 2 * len(df_before_norm.columns) + 1))
    # # Apply colors and linestyles
    # for patch, color, linestyle in zip(bplot['boxes'], colors, linestyles):
    #     patch.set_facecolor('white')  # Set the box's face color to white
    #     patch.set_edgecolor(color)  # Set the edge color
    #     patch.set_linestyle(linestyle)  # Set the linestyle
    # for i, mean in enumerate([np.mean(data) for data in data_to_plot]):
    #     ax.text(i + 1, mean, f'{mean:.2f}', ha='center', va='bottom', fontsize=8)
    # # Formatting the plot
    # # ax.set_xticks(range(1, 2 * len(df_before_norm.columns) + 1))
    # # ax.set_xticklabels(ticks, rotation=45, ha='right')
    # ax.set_ylabel('Neighbors Correlations')
    # y_min, y_max = ax.get_ylim()
    # ax.set_ylim(y_min, y_max + 0.1)
    # # ax.set_title('Comparison Before and After Background Normalization')
    # plt.tight_layout()
    # plt.savefig('../control_fig_S3.svg', format="svg")
    # plt.show()
    # t_stat, p_value = ttest_ind(df_before_norm['(0, 15)'], df_after_norm['(0, 15)'])
    # print("P-Value: ", p_value)
    # x=1

    #### plot nbrs correlations vs non-neighbors correlations ####
    # exps_type.extend(exps_type2)
    # nbrs_corrs.extend(nbrs_corrs_new_rad)
    # non_nbrs_corrs.extend(non_nbrs_corrs_new_rad)
    # plot_average_correlation_neighbors_vs_non_neighbors(non_nbrs_corrs, nbrs_corrs, labels=exps,
    #                                                     title='Average Correlations 5.3 20230815_02 - after blebbistatin')
    # plot_average_correlation_neighbors_vs_non_neighbors(non_nbrs_corrs, nbrs_corrs, labels=exps_type, title='Average Correlations 5.3-20230815_02- Before Blebb vs. After Blebb')
    # print("Avg neighbors correlations:", np.mean(nbrs_corrs))
    # print("Avg non-neighbors correlations:", np.mean(non_nbrs_corrs))
    # print("Avg neighbors correlations before blebb:", np.mean(nbrs_corrs_before_blebb))
    # print("Avg non-neighbors correlations before blebb:", np.mean(non_nbrs_corrs_before_blebb))
    # print("Avg neighbors correlations after blebb:", np.mean(nbrs_corrs_after_blebb))
    # print("Avg non-neighbors correlations after blebb:", np.mean(non_nbrs_corrs_after_blebb))
    # print("Avg neighbors correlations 5.3:", np.mean(nbrs_corrs_53))
    # print("Avg non-neighbors correlations 5.3:", np.mean(non_nbrs_corrs_53))
    # print("Avg neighbors correlations 13.2:", np.mean(nbrs_corrs_132))
    # print("Avg non-neighbors correlations 13.2:", np.mean(non_nbrs_corrs_132))

    #### interactive plot ####
    # nbrs_corrs = [float(c) for c in nbrs_corrs]
    # non_nbrs_corrs = [float(c) for c in non_nbrs_corrs]
    # exps_shorten = [e[11:15] for e in exps]
    # df = pd.DataFrame({'experiment': exps, 'exp_shorten': exps_shorten, 'type': exps_type, 'nbrs_corrs': nbrs_corrs, 'non_nbrs_corrs': non_nbrs_corrs})
    # fig = px.scatter(df, x='non_nbrs_corrs', y='nbrs_corrs', color='type', text='exp_shorten', hover_name='experiment',
    #                  title='Average Correlations Type 5.3 Exp 20230815_02 - Cells Before Blebb vs. After Blebb',
    #                  labels=dict(exp_shorten="Experiment Video-Cell", type="Type", nbrs_corrs="Neighbors Correlations", non_nbrs_corrs="Non-Neighbors Correlations"))
    # fig.update_layout(xaxis=dict(range=[-0.5, 1]), yaxis=dict(range=[-0.5, 1]),
    #                   shapes=[{'type': 'line', 'yref': 'paper', 'xref': 'paper', 'y0': 0, 'y1': 1, 'x0': 0, 'x1': 1}])
    # fig.update_shapes(line_color='grey', line_dash='dash', line_width=1)
    # fig.update_traces(marker={'size': 15, 'opacity': 0.5})
    # fig.write_html("../avg_corrs_5.3_exp2023081502.html")
    # print("Avg neighbors correlations before blebb:", np.mean(nbrs_corrs_before_blebb))
    # print("Avg non-neighbors correlations before blebb:", np.mean(non_nbrs_corrs_before_blebb))
    # print("Avg neighbors correlations after blebb:", np.mean(nbrs_corrs_after_blebb))
    # print("Avg non-neighbors correlations after blebb:", np.mean(non_nbrs_corrs_after_blebb))

    #### plot indivitual cells w & w/o bleb ####
    # corrs_53 = [float(c) for c in nbrs_corrs_before_blebb]
    # print("nbrs avg corrs before blebb", np.mean(corrs_53))
    # cors_bleb_53 = [float(c) for c in nbrs_corrs_after_bleb]
    # print("nbrs avg corrs after blebb", np.mean(cors_bleb_53))
    # sns.histplot(corrs_53, label="before blebb", kde=True)
    # sns.histplot(cors_bleb_53, label="after blebb", kde=True)
    # plt.title("exp 20230713_02 cell 4 nbrs corrs histogram")
    # plt.legend()
    # plt.show()

    ###### plot correlation of cells in time ########
    # cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8]
    # labels = ['cell 1', 'cell 2', 'cell 3', 'cell 4', 'cell 5', 'cell 6', 'cell 7', 'cell 8']
    # cells = [cell5, cell6, cell7, cell8, cell9]
    # labels = ['cell 5', 'cell 6', 'cell 7', 'cell 8', 'cell 9']
    # fig, ax = plt.subplots()
    # for i in range(len(cells)):
    #     ax.plot(cells[i], label=labels[i], linestyle='dashed', marker='o')
    #
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Nbrs Correlations')
    # ax.set_title("Exp 5.3 20230323 cells' correlations in time")
    # ax.legend()
    # plt.show()

    # cell1_before_mean = np.mean(cell1_before)
    # cell2_before_mean = np.mean(cell2_before)
    # cell3_before_mean = np.mean(cell3_before)
    # cell6_before_mean = np.mean(cell6_before)
    #
    # cell1_after_mean = np.mean(cell1_after)
    # cell2_after_mean = np.mean(cell2_after)
    # cell3_after_mean = np.mean(cell3_after)
    # cell6_after_mean = np.mean(cell6_after)
    #
    # x = ['before', 'after', 'before', 'after', 'before', 'after', 'before', 'after']
    # y = [cell1_before_mean, cell1_after_mean, cell2_before_mean, cell2_after_mean, cell3_before_mean, cell3_after_mean, cell6_before_mean, cell6_after_mean]

    # with open('csv_res.csv', 'w', encoding='UTF8', newline='') as f:
    #     writer = csv.writer(f)
    #     for i in range(len(exps)):
    #         row = [exps[i], nbrs_corrs[i], non_nbrs_corrs[i]]
    #         # print(exps[i], nbrs_corrs[i], non_nbrs_corrs[i])
    #         writer.writerow(row)
    # t_test(second_corrs, first_corrs)
    # diff = np.array(second_corrs) - np.array(first_corrs)
    # tset, pval = ttest_1samp(diff, 0)
    # print(pval)
    # plot_correlation_by_distance_from_center_cell(list_dicts, exps)
    # special_legend = ['special', 'non special', 'non special', 'non special', 'special', 'special',
    #                   'special', 'special', 'special', 'special']
    # title = "Probability for a GC Edge 13.2"
    # xlabel = "Random Neighbors"
    # ylabel = "Original Neighbors"
    # cells_lst = [exp.split('-')[-1] for exp in exps]
    # plot_average_correlation_neighbors_vs_non_neighbors(non_nbrs_corrs, nbrs_corrs, labels=exps, title='Average Correlation - 5.3 vs 13.2', cells_lst=exps)
    # print(np.mean([float(cor) for cor in nbrs_corrs]))

    ##### fig 2A inset #####
    # nbrs_corrs_53 = [float(c) for c in nbrs_corrs_53]
    # non_nbrs_corrs_53 = [float(c) for c in non_nbrs_corrs_53]
    # t_stat, p_value_nbrs_vs_non_53 = ttest_ind(nbrs_corrs_53, non_nbrs_corrs_53)
    # print("p-value for nbrs vs non-nbrs corrs 5.3:", p_value_nbrs_vs_non_53)
    # nbrs_corrs_132 = [float(c) for c in nbrs_corrs_132]
    # non_nbrs_corrs_132 = [float(c) for c in non_nbrs_corrs_132]
    # t_stat, p_value_nbrs_vs_non_132 = ttest_ind(nbrs_corrs_132, non_nbrs_corrs_132)
    # print("p-value for nbrs vs non-nbrs corrs 13.2:", p_value_nbrs_vs_non_132)
    # diff_53 = np.array(nbrs_corrs_53) - np.array(non_nbrs_corrs_53)
    # print("Number of cells over the diagonal for 5.3:", np.sum(diff_53 > 0), "/", len(diff_53))
    # diff_132 = np.array(nbrs_corrs_132) - np.array(non_nbrs_corrs_132)
    # print("Number of cells over the diagonal for 13.2:", np.sum(diff_132 > 0), "/", len(diff_132))
    # t_stat, p_value = ttest_ind(diff_53, diff_132)
    # print("T-statistic value: ", t_stat)
    # print("P-Value: ", p_value)
    # plt.rcParams.update({'font.size': 12})
    # plt.rcParams['font.family'] = 'Arial'
    # plt.figure(figsize=(6, 4))
    # sns.distplot(diff_53, kde=True, label='5.3', color='green')
    # sns.distplot(diff_132, kde=True, label='13.2', color='red')
    # plt.axvline(x=0, color="gainsboro", linestyle="--")
    # plt.xlabel("Difference of adjacent and non-adjacent correlation")
    # # plt.title("Delta correlations of nbrs and non-nbrs histogram")
    # plt.yticks(rotation=90)
    # plt.legend()
    # plt.savefig('../diff_corrs_fig_2A_inset.svg', format="svg")
    # plt.show()

    # features_lst = ['nbrs_avg_signal_corr', 'non_nbrs_avg_signal_corr', 'neighbors_avg_move_corr',
    #                 'not_neighbors_avg_move_corr', 'out/in_factor', 'gc_edge_prob', 'avg_random_gc_edge_prob',
    #                 'reciprocity', 'heterogeneity', 'periph_avg_move_signal_corr', 'central_avg_move_signal_corr']
    # plot_experiment_features_heatmap(exps, features_lst, all_exps_features)
    # spreading_level = ['high', 'low', 'low', 'low', 'high', 'high', 'high', 'high', 'high', 'high']
    # plot_avg_correlation_spreading_level(exps, corrs, spreading_level)
    # plot_average_movement_signal_sync_peripheral_vs_central(center_corrs, periph_corrs, labels=exps, title=exp_type)
    # neighbors_correlation_histogram(nbrs_corrs, get_alive_pillars_to_alive_neighbors())
    # non_neighbors_correlation_histogram(non_nbrs_corrs, get_alive_pillars_to_alive_neighbors())
    # plot_average_correlation_neighbors_vs_non_neighbors(nbrs_movement_corrs, nbrs_intsn_corrs, labels=exps, title=exp_type)
    # corrs = [float(corr) for corr in corrs]
    # plt.scatter(exps, corrs)
    # plt.xticks(rotation=45, fontsize=8)
    # plt.ylim(-1, 1)
    # plt.axhline(y=0, color='r', linestyle='dashed')
    # plt.xlabel("Experiment")
    # plt.ylabel("Average Correlation")
    # plt.title("Experiments average correlation of intensity-movement")
    # plt.show()

    # TODO: delete

    # total_movements_percentage = {}
    # fig, ax = plt.subplots()

    # move_percentage = run_config(config_path)
    # total_movements_percentage[str(exp_type) + " - " + str(exp_name)] = move_percentage

    # r132=[]
    # r94=[]
    # kd132 = []
    # kd94 = []
    # mef53 = []
    # vr132=[]
    # vr94=[]
    # vkd132 = []
    # vkd94 = []
    # vmef53 = []
    # for k,v in total_movements_percentage.items():
    #     if "13.2" in k:
    #         r132.append(k)
    #         vr132.append(v)
    #     if "KD13.2" in k:
    #         kd132.append(k)
    #         vkd132.append(v)
    #     if "9.4" in k:
    #         r94.append(k)
    #         vr94.append(v)
    #     if "KD9.4" in k:
    #         kd94.append(k)
    #         vkd94.append(v)
    #     if "5.3" in k:
    #         mef53.append(k)
    #         vmef53.append(v)
    # x = [r132, kd132, r94, kd94, mef53]
    # y = [vr132, vkd132, vr94, vkd94, vmef53]
    # labels = ["13.2", "KD13.2", "9.4", "KD9.4", "5.3"]
    # for i, label in enumerate(labels):
    #     ax.scatter(x[i], y[i], label=label)
    # plt.xticks(rotation=45)
    # plt.xlabel("Experiment")
    # plt.ylabel("Movement percentage")
    # # plt.title("Percentage Chance of Movement - Exp Type: " + str(exp_type))
    # plt.title("Percentage Chance of Movement")
    # plt.legend()
    # plt.show()




