from Pillars.runner import *
from pathlib import Path

if __name__ == '__main__':

    config_paths = [
        # 'KD13.2/exp_49.1_type_KD13.2_mask_15_35_non-normalized_fixed.json',
        # 'KD13.2/exp_46_type_KD13.2_mask_15_35_non-normalized_fixed.json',
        # 'KD13.2/exp_01_type_KD13.2_mask_15_35_non-normalized_fixed.json',
        # 'KD13.2/exp_02_type_KD13.2_mask_15_35_non-normalized_fixed.json',
        # 'KD13.2/exp_17_type_KD13.2_mask_15_35_non-normalized_fixed.json',

        '13.2/exp_01_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_05_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_06_type_13.2_mask_15_35_non-normalized_fixed.json',
        '13.2/exp_20_type_13.2_mask_15_35_non-normalized_fixed.json',

        '9.4/exp_04.1_type_9.4_mask_15_35_non-normalized_fixed.json',
        '9.4/exp_04.2_type_9.4_mask_15_35_non-normalized_fixed.json',
        '9.4/exp_06.1_type_9.4_mask_15_35_non-normalized_fixed.json',
        '9.4/exp_06.2_type_9.4_mask_15_35_non-normalized_fixed.json',
        '9.4/exp_11.1_type_9.4_mask_15_35_non-normalized_fixed.json',
        '9.4/exp_11.2_type_9.4_mask_15_35_non-normalized_fixed.json',
        '9.4/exp_12_type_9.4_mask_15_35_non-normalized_fixed.json',

        'KD9.4/exp_02_type_KD9.4_mask_15_35_non-normalized_fixed.json',
        'KD9.4/exp_08_type_KD9.4_mask_15_35_non-normalized_fixed.json',
        'KD9.4/exp_11_type_KD9.4_mask_15_35_non-normalized_fixed.json',
        'KD9.4/exp_17_type_KD9.4_mask_15_35_non-normalized_fixed.json',
        'KD9.4/exp_53_type_KD9.4_mask_15_35_non-normalized_fixed.json',

        '5.3/exp_08_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_09_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_12_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_27.1_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_27.2_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_30_type_5.3_mask_15_35_non-normalized_fixed.json',

        'REF5.3/exp_37.41.1_type_REF5.3_mask_15_35_non-normalized_fixed.json',
        'REF5.3/exp_37.41.2_type_REF5.3_mask_15_35_non-normalized_fixed.json',
        'REF5.3/exp_37.41.3_type_REF5.3_mask_15_35_non-normalized_fixed.json',
        'REF5.3/exp_37.41.4_type_REF5.3_mask_15_35_non-normalized_fixed.json',
        'REF5.3/exp_69.1_type_REF5.3_mask_15_35_non-normalized_fixed.json',
        'REF5.3/exp_69.2_type_REF5.3_mask_15_35_non-normalized_fixed.json',
        'REF5.3/exp_69.3_type_REF5.3_mask_15_35_non-normalized_fixed.json',
    ]

    Consts.BLOCK_TO_SHOW_GRAPH = False
    Consts.WRITE_OUTPUT = True

    for config_path in config_paths:
        path_name_split = config_path.split('_')
        exp_type = path_name_split[0].split('/')[0]
        exp_name = path_name_split[1]
        Consts.RESULT_FOLDER_PATH = "../multi_config_runner_results/" + exp_type + '/' + exp_name
        Path(Consts.RESULT_FOLDER_PATH).mkdir(parents=True, exist_ok=True)
        run_config(config_path)
