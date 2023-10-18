import pandas as pd

from runner import *
from pathlib import Path
from scipy.stats import ttest_1samp
import plotly.express as px

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

        '5.3/exp_20230323-03-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-03-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-03-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-03-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-03-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-03-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-03-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_20230323-03-8_type_5.3_mask_15_35_non-normalized_fixed.json',

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

        ### Before blebbing exp 01 ####
        # '5.3/exp_2023071301-01-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-01-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-01-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-01-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-01-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-01-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2023071301-02-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-02-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-02-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-02-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-02-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-02-7_type_5.3_mask_15_35_non-normalized_fixed.json',

        ### After blebbing exp 01####
        # '5.3/exp_2023071301-04-1_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-04-2_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-04-3_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-04-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # #
        # '5.3/exp_2023071301-05-1_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-05-2_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-05-3_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071301-05-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        #
        # # #### Before blebbing exp 02 ####
        # '5.3/exp_2023071302-01-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071302-01-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071302-01-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071302-01-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # #
        # '5.3/exp_2023071302-02-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071302-02-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071302-02-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071302-02-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # #
        # # #### After blebbing exp 02 ####
        # '5.3/exp_2023071302-04-1_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071302-04-2_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071302-04-3_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071302-04-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2023071302-05-1_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071302-05-2_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023071302-05-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',

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
        #
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
        #
        # '5.3/exp_2023081501-03-1_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-03-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081501-03-5_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2023081501-04-1_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2023081502-01-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-01-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-01-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-01-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # #
        # '5.3/exp_2023081502-02-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-02-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-02-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-02-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2023081502-03-1_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-03-2_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-03-3_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-03-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_2023081502-04-2_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-04-3_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_2023081502-04-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        #
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
        #
        # '5.3/exp_20230818-03-2_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-03-3_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-03-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-03-5_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230818-04-3_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-04-4_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230818-04-5_type_5.3_bleb_mask_15_35_non-normalized_fixed.json',

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
        # #
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
        # #
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
        #
        # ### After blebbing exps01 ####
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
        #
        # ### Before blebbing exps02 ####
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
    Consts.WRITE_OUTPUT = True

    # # TODO: delete
    nbrs_corrs = []
    non_nbrs_corrs = []
    list_53 = []
    list_132 = []
    # nbrs_corrs_new_rad = []
    # non_nbrs_corrs_new_rad = []
    # nbrs_corrs_53 = []
    # non_nbrs_corrs_53 = []
    nbrs_corrs_before_blebb = []
    non_nbrs_corrs_before_blebb = []
    # nbrs_corrs_132 = []
    # non_nbrs_corrs_132 = []
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
    map_exp_to_delta_corrs = {}
    map_exp_to_statistics_original_and_random_strong_nodes_distances = {}
    map_exp_to_statistics_original_and_random_strong_nodes_clusters = {}
    map_exp_to_statistics_original_and_random_strong_nodes_hops = {}
    map_exp_to_statistics_original_and_random_number_of_cc = {}
    map_exp_to_statistics_original_and_random_avg_similarity = {}
    exps_dict_distances_category_lst = []
    strength = []
    similarity = []

    # radiuses_by_0_10 = ['(15, 35)', '(0, 10)', '(10, 30)', '(20, 40)', '(15, 40)', '(10, 40)']
    # my_dict_0_10 = {k: [] for k in radiuses_by_0_10}
    # radiuses_by_0_15 = ['(15, 35)', '(0, 15)', '(10, 30)', '(20, 40)', '(15, 40)', '(10, 40)']
    # my_dict_0_15 = {k: [] for k in radiuses_by_0_15}
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
        exps.append(exp_name)
        # center_corrs.append(corrs_dict["centrals"])
        # periph_corrs.append(corrs_dict["peripherals"])
        ######
        # exp = str(exp_type) + " - " + str(exp_name)
        # exps.append(exp)
        # if "bleb" in path_name_split:
        #     exps_type.append("After blebb")
        # else:
        #     exps_type.append("Before blebb")
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

        ## TODO:
        # nbrs_avg_correlation, non_nbrs_avg_correlation = get_experiment_results_data(
        #     Consts.RESULT_FOLDER_PATH + '/results.csv',
        #     ['nbrs_avg_correlation', 'non_nbrs_avg_correlation'])
        # nbrs_corrs.append(float(nbrs_avg_correlation))
        # non_nbrs_corrs.append(float(non_nbrs_avg_correlation))
        # delta_corr = float(nbrs_avg_correlation) - float(non_nbrs_avg_correlation)
        # if "bleb" in path_name_split:
        #     exps_type.append('after blebb')
        #     nbrs_corrs_after_blebb.append(float(nbrs_avg_correlation))
        #     non_nbrs_corrs_after_blebb.append(float(non_nbrs_avg_correlation))
        # else:
        #     exps_type.append('before blebb')
        #     nbrs_corrs_before_blebb.append(float(nbrs_avg_correlation))
        #     non_nbrs_corrs_before_blebb.append(float(non_nbrs_avg_correlation))

        # map_radius_corrs_0_10 = get_experiment_results_data(Consts.RESULT_FOLDER_PATH + '/results.csv',
        #                                                ['correlations_norm_by_noise_(0,10)'])
        # map_radius_corrs_0_10 = eval(map_radius_corrs_0_10[0])
        # for k, v in map_radius_corrs_0_10.items():
        #     my_dict_0_10[str(k)].append(float(v['nbrs_corrs']))
        # map_radius_corrs_0_15 = get_experiment_results_data(Consts.RESULT_FOLDER_PATH + '/results.csv',
        #                                                ['correlations_norm_by_noise_(0,15)'])
        # map_radius_corrs_0_15 = eval(map_radius_corrs_0_15[0])
        # for k, v in map_radius_corrs_0_15.items():
        #     my_dict_0_15[str(k)].append(float(v['nbrs_corrs']))
        # exps_type.append('(15,35)')
        # my_dict['(15, 35)'].append(float(nbrs_avg_correlation))

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

        #### correlations delta before and after blebb #####
        # exp_name_lst = exp_name.split('-')
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

        # sim = run_config(config_path)
        # sns.histplot(sim, label="similarity", kde=True, alpha=0.3)
        # plt.xlabel('Similarity')
        # plt.title("Strength Similarity Distribution of Neighboring Pillars")
        # plt.legend()
        # plt.show()

        # avg_strength, avg_similarity = run_config(config_path)
        # strength.append(avg_strength)
        # similarity.append(avg_similarity)

        ##################################### RUN  ############################################
        # run_config(config_path)

        # try:
        #     # run_config(config_path)
        #     avg_sim, permuted_test_statistics, p_value = run_config(config_path)
        #     map_exp_to_statistics_original_and_random_avg_similarity[exp_name] = (
        #         avg_sim, permuted_test_statistics, p_value)
        # except Exception as error:
        #     print("there was an error in config path " + str(config_path) + str(error))
        # print(config_path, "completed")
        ##################################### RUN  ############################################

    # plot_average_correlation_neighbors_vs_non_neighbors(strength, similarity, labels=exps,
    #                                                     title='Pillars Avg Strength vs. Similarity - 13.2 exps',
    #                                                     xlabel='Avg strength', ylabel='Avg similarity')

    # sns.histplot(list_53, label="5.3", kde=True, alpha=0.3)
    # sns.histplot(list_132, label="13.2", kde=True, alpha=0.3)
    # plt.title("Distribution of Average Correlations Between All Pillars")
    # plt.xlabel('Avg Correlations')
    # plt.legend()
    # plt.show()
    # print("Average correlation in 5.3 exps:", "%.3f" % np.mean(list_53))
    # print("Average correlation in 13.2 exps:", "%.3f" % np.mean(list_132))
    # t_stat, p_value = ttest_ind(list_53, list_132)
    # print("###### t-test 13.2 originals vs permutations clusters #####")
    # print("T-statistic value: ", "%.2f" % t_stat)
    # print("P-Value: ", p_value)

    # with open('../map_vid_to_original_and_shuffle_ts_500_avg_sim_with_pval_5.3_exps', 'wb') as f:
    #     pickle.dump(map_exp_to_statistics_original_and_random_avg_similarity, f)
    # with open('../map_vid_to_strong_nodes_original_and_shuffle_ts_largest_cc_with_pval_5.3_exps', 'rb') as file:
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

    ######## Plot correlations delta before and after blebb ########
    # lbls = []
    # delta_before = []
    # delta_after = []
    # for exp_k, exp_v in map_exp_to_delta_corrs.items():
    #     for cell_k, cell_v in exp_v.items():
    #         d_before = np.mean(cell_v['before'])
    #         d_after = np.mean(cell_v['after'])
    #         delta_before.append(d_before)
    #         delta_after.append(d_after)
    #         lbls.append(exp_k + '-' + cell_k)
    #
    # plot_average_correlation_neighbors_vs_non_neighbors(delta_before, delta_after, labels=lbls,
    #                                                     xlabel='delta before', ylabel='delta after',
    #                                                     title='Correlations Delta Before and After Blebb')

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

    #### change in correlation by radius - control and sensitivity ####
    # df = pd.DataFrame.from_dict(my_dict_0_15)
    # means = [np.mean(df[col]) for col in list(df.columns)]
    # median = [np.median(df[col]) for col in list(df.columns)]
    # print("means", [float("%.3f" % m) for m in means])
    # print("median", median)
    # box_plot = sns.boxplot(data=df)
    # for i in range(len(median)):
    #     box_plot.annotate(str("%.3f" % median[i]), xy=(i+0.2, median[i]), ha='right', color='white', fontsize="8")
    # plt.ylabel("Neighbors Correlations", size=10)
    # plt.xlabel("Radius", size=10)
    # plt.title("Neighbors Correlations of Different Radius - 5.3 exp202308018 After Blebb")
    # plt.show()
    ##t-test for control##
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
    #
    # nbrs_corrs_53 = [float(c) for c in nbrs_corrs_53]
    # non_nbrs_corrs_53 = [float(c) for c in non_nbrs_corrs_53]
    # nbrs_corrs_132 = [float(c) for c in nbrs_corrs_132]
    # non_nbrs_corrs_132 = [float(c) for c in non_nbrs_corrs_132]
    # diff_53 = np.array(nbrs_corrs_53) - np.array(non_nbrs_corrs_53)
    # diff_132 = np.array(nbrs_corrs_132) - np.array(non_nbrs_corrs_132)
    # t_stat, p_value = ttest_ind(diff_53, diff_132)
    # print("T-statistic value: ", t_stat)
    # print("P-Value: ", p_value)
    # sns.distplot(diff_53, kde=True, label='5.3', color='green')
    # sns.distplot(diff_132, kde=True, label='13.2', color='red')
    # plt.axvline(x=0, color="gainsboro", linestyle="--")
    # plt.xlabel("sub(avg_nbrs_corrs, avg_non_nbrs_corrs)")
    # plt.legend()
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
