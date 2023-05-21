from Pillars.runner import *
from pathlib import Path
from scipy.stats import ttest_1samp

if __name__ == '__main__':

    config_paths = [

        # '5.3/exp_08_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_09_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_12_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_27.1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_27.2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_30_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_07-1_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_07-2_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_07-3_type_5.3_mask_15_35_non-normalized_fixed.json',
        '5.3/exp_07-4_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_07-5_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_07-6_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_07-7_type_5.3_mask_15_35_non-normalized_fixed.json',
        # '5.3/exp_07-8_type_5.3_mask_15_35_non-normalized_fixed.json',

        # '13.2/exp_01_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_05_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_06_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_20_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_1-149-1_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_1-149-2_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_1-149-3_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_1-149-4_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_1-149-5_type_13.2_mask_15_35_non-normalized_fixed.json',
        # '13.2/exp_1-149-6_type_13.2_mask_15_35_non-normalized_fixed.json',

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
    #
    # # corrs = []
    # #####
    exps = []
    # list_dicts = []
    # first_corrs = []
    # second_corrs = []
    # center_corrs = []
    # periph_corrs = []
    # all_exps_features = []
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
        # nbrs_avg_correlation, non_nbrs_avg_correlation = get_experiment_results_data(Consts.RESULT_FOLDER_PATH + '/results.csv',
        #                                    ['nbrs_avg_correlation', 'non_nbrs_avg_correlation'])
        # nbrs_corrs.append(nbrs_avg_correlation)
        # non_nbrs_corrs.append(non_nbrs_avg_correlation)
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
    # plot_average_correlation_neighbors_vs_non_neighbors(non_nbrs_corrs, nbrs_corrs, labels=exps)
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
