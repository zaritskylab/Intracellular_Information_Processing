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

        # '5.3/exp_20230320-02-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-02-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-02-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-02-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-02-9_type_5.3_mask_15_35_non-normalized_fixed.json',
        # #
        # '5.3/exp_20230320-03-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-03-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-03-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-03-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-03-9_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230320-04-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-04-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-04-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-04-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-04-9_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230320-05-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # # '5.3/exp_20230320-05-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-05-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-05-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-05-9_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230320-06-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-06-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-06-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230320-06-9_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230323-01-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-01-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-01-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-01-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-01-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-01-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-01-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-01-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230323-03-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-03-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-03-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-03-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-03-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-03-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-03-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-03-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230323-04-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-04-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-04-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-04-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-04-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-04-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-04-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-04-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230323-05-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-05-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-05-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-05-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-05-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-05-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-05-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-05-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230323-06-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-06-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-06-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-06-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-06-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-06-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230323-07-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-07-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-07-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-07-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-07-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        # #
        # '5.3/exp_20230323-08-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-08-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-08-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-08-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-08-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-08-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230323-09-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-09-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-09-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-09-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-09-8_type_5.3_mask_15_35_non-normalized_fixed.json',
        #
        # '5.3/exp_20230323-10-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # # '5.3/exp_20230323-10-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-10-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-10-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_20230323-10-8_type_5.3_mask_15_35_non-normalized_fixed.json',


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
        #
        # '13.2/exp_20230327-04-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-04-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-04-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-04-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-04-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-04-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-04-7_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-04-8_type_13.2_mask_15_35_non-normalized_fixed.json',
        #
        # '13.2/exp_20230327-05-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-05-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-05-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-05-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-05-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-05-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-05-7_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-05-8_type_13.2_mask_15_35_non-normalized_fixed.json',
        #
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
        #
        # '13.2/exp_20230327-09-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-09-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-09-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-09-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-09-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230327-09-8_type_13.2_mask_15_35_non-normalized_fixed.json',

        # '13.2/exp_20230712_01-01-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230712_01-01-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20230712_01-01-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-01-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-01-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-01-6_type_13.2_mask_15_35_non-normalized_fixed.json',

        '13.2/exp_20230712_01-02-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-02-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-02-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-02-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-02-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-02-6_type_13.2_mask_15_35_non-normalized_fixed.json',

        '13.2/exp_20230712_01-04-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-04-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-04-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-04-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-04-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-04-6_type_13.2_mask_15_35_non-normalized_fixed.json',

        '13.2/exp_20230712_01-05-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-05-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-05-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-05-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-05-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_01-05-6_type_13.2_mask_15_35_non-normalized_fixed.json',

        '13.2/exp_20230712_02-01-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-01-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-01-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-01-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-01-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-01-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-01-7_type_13.2_mask_15_35_non-normalized_fixed.json',

        '13.2/exp_20230712_02-02-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-02-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-02-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-02-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-02-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-02-6_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-02-7_type_13.2_mask_15_35_non-normalized_fixed.json',

        '13.2/exp_20230712_02-04-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-04-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-04-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-04-6_type_13.2_mask_15_35_non-normalized_fixed.json',

        '13.2/exp_20230712_02-05-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-05-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-05-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20230712_02-05-6_type_13.2_mask_15_35_non-normalized_fixed.json',



        # 'KD13.2/exp_49.1_type_KD13.2_mask_15_35_non-normalized_fixed.json',
        # 'KD13.2/exp_46_type_KD13.2_mask_15_35_non-normalized_fixed.json',
        # 'KD13.2/exp_01_type_KD13.2_mask_15_35_non-normalized_fixed.json',
        # 'KD13.2/exp_02_type_KD13.2_mask_15_35_non-normalized_fixed.json',
        # 'KD13.2/exp_17_type_KD13.2_mask_15_35_non-normalized_fixed.json',

        # '9.4/exp_04.1_type_9.4_mask_15_35_non-normalized_fixed.json',
        # '9.4/exp_04.2_type_9.4_mask_15_35_non-normalized_fixed.json',
        # '9.4/exp_06.1_type_9.4_mask_15_35_non-normalized_fixed.json',
        # '9.4/exp_06.2_type_9.4_mask_15_35_non-normalized_fixed.json',
        # '9.4/exp_11.1_type_9.4_mask_15_35_non-normalized_fixed.json',
        # '9.4/exp_11.2_type_9.4_mask_15_35_non-normalized_fixed.json',
        # '9.4/exp_12_type_9.4_mask_15_35_non-normalized_fixed.json',
        #
        # 'KD9.4/exp_02_type_KD9.4_mask_15_35_non-normalized_fixed.json',
        # 'KD9.4/exp_08_type_KD9.4_mask_15_35_non-normalized_fixed.json',
        # 'KD9.4/exp_11_type_KD9.4_mask_15_35_non-normalized_fixed.json',
        # 'KD9.4/exp_17_type_KD9.4_mask_15_35_non-normalized_fixed.json',
        # 'KD9.4/exp_53_type_KD9.4_mask_15_35_non-normalized_fixed.json',

        # 'REF5.3/exp_37.41.1_type_REF5.3_mask_15_35_non-normalized_fixed.json',
        # 'REF5.3/exp_37.41.2_type_REF5.3_mask_15_35_non-normalized_fixed.json',
        # 'REF5.3/exp_37.41.3_type_REF5.3_mask_15_35_non-normalized_fixed.json',
        # 'REF5.3/exp_37.41.4_type_REF5.3_mask_15_35_non-normalized_fixed.json',
        # 'REF5.3/exp_69.1_type_REF5.3_mask_15_35_non-normalized_fixed.json',
        # 'REF5.3/exp_69.2_type_REF5.3_mask_15_35_non-normalized_fixed.json',
        # 'REF5.3/exp_69.3_type_REF5.3_mask_15_35_non-normalized_fixed.json',
    ]

    Consts.SHOW_GRAPH = False
    Consts.WRITE_OUTPUT = True

    # # TODO: delete
    nbrs_corrs = []
    non_nbrs_corrs = []
    nbrs_corrs_new_rad = []
    non_nbrs_corrs_new_rad = []
    # nbrs_corrs_53 = []
    # non_nbrs_corrs_53 = []
    # nbrs_corrs_132 = []
    # non_nbrs_corrs_132 = []
    avg_strenght_group_dist_from_center_53 = []
    avg_strenght_group_dist_from_center_132 = []
    # # corrs = []
    # #####
    # exps = []
    exps_type = []
    exps_type2 = []
    # list_dicts = []
    # first_corrs = []
    # second_corrs = []
    # center_corrs = []
    # periph_corrs = []
    # all_exps_features = []
    # radius = ['(15, 35)', '(0, 10)', '(0, 15)', '(10, 30)', '(20, 40)', '(15, 40)', '(10, 40)']
    # my_dict = {k: [] for k in radius}
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
        # center_corrs.append(corrs_dict["centrals"])
        # periph_corrs.append(corrs_dict["peripherals"])
        ######
        # exp = str(exp_type) + " - " + str(exp_name)
        # exps.append(exp)
        # exps_type.append(exp_type)
        # if exp_name[9:12] == '210':
        #     exp_video_name = exp_name[9:12]
        # else:
        #     exp_video_name = exp_name[9:14]
        # exp_video_name = exp_name[9:11]
        # exp_video_name = 'Video' + exp_video_name
        # exps.append(exp_type)
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
        # map_radius_corrs = get_experiment_results_data(Consts.RESULT_FOLDER_PATH + '/results.csv',
        #                                                ['change_mask_radius'])
        # exps_type.append('(15,35)')
        # map_radius_corrs = eval(map_radius_corrs[0])
        # rad_0_10 = map_radius_corrs[(0, 10)]
        # nbrs_corrs_new_rad.append(float(rad_0_10['nbrs_corrs']))
        # non_nbrs_corrs_new_rad.append(float(rad_0_10['non_nbrs_corrs']))
        # exps_type2.append('(0,10)')
        # for k, v in map_radius_corrs.items():
        #     my_dict[str(k)].append(float(v))
        # my_dict['(15, 35)'].append(float(nbrs_avg_correlation))

        # if exp_type == '5.3':
        #     avg = run_config(config_path)
        #     avg_strenght_group_dist_from_center_53.append(avg)
        #     # nbrs_corrs_53.append(nbrs_avg_correlation)
        #     # non_nbrs_corrs_53.append(non_nbrs_avg_correlation)
        # else:
        #     avg = run_config(config_path)
        #     avg_strenght_group_dist_from_center_132.append(avg)
        #     # nbrs_corrs_132.append(nbrs_avg_correlation)
        #     # non_nbrs_corrs_132.append(non_nbrs_avg_correlation)

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

        ##################################### RUN  ############################################
        run_config(config_path)

        # try:
        #     run_config(config_path)
        # except Exception as error:
        #     print("there was an error in config path " + str(config_path) + str(error))
        # print(config_path, "completed")
        ##################################### RUN  ############################################
    # print("mean strength nodes group distance from center in 5.3 exp", np.mean(avg_strenght_group_dist_from_center_53))
    # print("mean strength nodes group distance from center in 13.2 exp", np.mean(avg_strenght_group_dist_from_center_132))
    # sns.histplot(avg_strenght_group_dist_from_center_53, label="5.3", kde=True)
    # sns.histplot(avg_strenght_group_dist_from_center_132, label="13.2", kde=True)
    # plt.legend()
    # plt.show()
    # df = pd.DataFrame.from_dict(my_dict)
    # df['(0, 10)'].fillna(df['(0, 10)'].mean(), inplace=True)
    # means = [np.mean(df[col]) for col in list(df.columns)]
    # median = [np.median(df[col]) for col in list(df.columns)]
    # print("means", means)
    # print("median", median)
    # box_plot = sns.boxplot(data=df)
    # for i in range(len(median)):
    #     box_plot.annotate(str(median[i]), xy=(i+0.2, median[i]-0.025), ha='right', color='white', fontsize="8")
    # plt.ylabel("Neighbors Correlations", size=10)
    # plt.xlabel("Radius", size=10)
    # plt.title("Neighbors Correlations of Different Radius")
    # plt.show()

    # exps_type.extend(exps_type2)
    # nbrs_corrs.extend(nbrs_corrs_new_rad)
    # non_nbrs_corrs.extend(non_nbrs_corrs_new_rad)
    # plot_average_correlation_neighbors_vs_non_neighbors(non_nbrs_corrs, nbrs_corrs, labels=exps_type, title='Average Correlation 5.3 - (0,10) vs (15,35)', cells_lst=exps)

    # rad_0_15 = df['(0, 15)']
    # rad_15_35 = df['(15, 35)']
    # t_stat, p_value = ttest_ind(rad_0_15, rad_15_35)
    # print("T-statistic value: ", t_stat)
    # print("P-Value: ", p_value)

    # nbrs_corrs = [float(c) for c in nbrs_corrs]
    # non_nbrs_corrs = [float(c) for c in non_nbrs_corrs]
    # df = pd.DataFrame({'experiment': exps, 'type': exps_type, 'nbrs_corrs': nbrs_corrs, 'non_nbrs_corrs': non_nbrs_corrs})
    # fig = px.scatter(df, x='non_nbrs_corrs', y='nbrs_corrs', color='experiment',
    #                  labels=dict(nbrs_corrs="Neighbors Correlations", non_nbrs_corrs="Non-Neighbors Correlations"))
    # # fig.update_layout(xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1]))
    # fig.write_html("../5.3_vs_13.2.html")

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
