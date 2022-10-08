import numpy
from matplotlib.pyplot import axline

from Pillars.visualization import *
from Pillars.granger_causality import *
from Pillars.repositioning import *
from Pillars.analyzer import *
from pathlib import Path
from Pillars.runner_helper import *

import json
import math


def update_const_by_config(config_data):
    # Update experiment configurations
    experiment_data = config_data["experiment"]
    experiment_id = experiment_data['id']
    experiment_tif_path = experiment_data['tif_path']
    perturbation = config_data["perturbation"]
    perturbation_type = perturbation['type']

    Consts.USE_CACHE = config_data.get('use_cache', True)

    Consts.fixed_images_path = Consts.PILLARS + '\\FixedImages\\Fixed_' + perturbation_type + '\\new_fixed_' + experiment_id + '.tif'
    if not Path(Consts.fixed_images_path).exists():
        reposition(Consts.PILLARS + '\\' + perturbation_type + experiment_tif_path, Consts.fixed_images_path)
    Consts.images_path = Consts.PILLARS + experiment_tif_path
    Consts.last_image_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/last_image_' + experiment_id + '.npy'

    images = get_images(get_images_path())

    if not Path(Consts.last_image_path).exists():
        with open(Consts.last_image_path, 'wb') as f:
            np.save(f, images[-1])

    Consts.IMAGE_SIZE = len(images[-1])

    Consts.normalized = config_data.get('normalized', False)
    Consts.fixed = config_data.get('fixed', True)
    Consts.use_otsu = config_data.get('use_otsu', True)
    Consts.pixel_to_whiten = config_data.get('pixel_to_whiten', 10)
    Consts.MAX_DISTANCE_PILLAR_FIXED = config_data.get('max_distance_pillar_fixed', 11)

    # Update mask radius
    mask_radius = config_data.get('mask_radius', {
        "small_radius": 15,
        "large_radius": 35
    })

    small_mask_radius_ratio = mask_radius['small_radius'] / 20
    large_mask_radius_ratio = mask_radius['large_radius'] / 20

    Consts.percentage_from_perfect_circle_mask = config_data.get('percentage_from_perfect_circle_mask', 1)

    Consts.last_img_alive_centers_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + 'alive_centers_cached.pickle'

    radius = get_circle_radius(config_data)
    Consts.SMALL_MASK_RADIUS = math.floor(radius * small_mask_radius_ratio)
    Consts.LARGE_MASK_RADIUS = math.floor(radius * large_mask_radius_ratio)

    Consts.CIRCLE_RADIUS = radius
    micron = config_data["metadata"]["micron"]
    validation_ratio = int(micron / Consts.RELATIVE_TO)
    Consts.CHECK_VALID_CENTER = math.ceil(Consts.CHECK_VALID_CENTER / validation_ratio)
    Consts.CIRCLE_INSIDE_VALIDATE_SEARCH_LENGTH = Consts.CIRCLE_RADIUS - Consts.CHECK_VALID_CENTER
    Consts.CIRCLE_OUTSIDE_VALIDATE_SEARCH_LENGTH = Consts.CIRCLE_RADIUS + Consts.CHECK_VALID_CENTER

    # TODO: if we want to cache by other params (like mask radius) we need to add them to the path
    path_postfix = str(Consts.SMALL_MASK_RADIUS) + '_' + str(Consts.LARGE_MASK_RADIUS) + '_fully_' + str(
        Consts.inner_cell) + '_fixed_' + str(Consts.fixed) + '_normalized_' + str(Consts.normalized) + '_'

    # Update caches path.
    Consts.pillar_to_intensities_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'pillar_to_intensities_cached.pickle'
    Consts.correlation_alive_normalized_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'alive_pillar_correlation_normalized_cached.pickle'
    Consts.correlation_alive_not_normalized_cache_path = '..SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'alive_pillar_correlation_cached.pickle'
    Consts.all_pillars_correlation_normalized_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'all_pillar_correlation_normalized_cached.pickle'
    Consts.all_pillars_correlation_not_normalized_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'all_pillar_correlation_cached.pickle'
    Consts.frame2pillar_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'frames2pillars_cached.pickle'
    Consts.frame2alive_pillars_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'frames2alive_pillars_cached.pickle'
    Consts.gc_df_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'gc_df_cached.pickle'
    Consts.alive_pillars_corr_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'alive_pillars_corr_cached.pickle'
    Consts.centers_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'centers_cached.pickle'
    Consts.pillar_to_neighbors_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'pillar_to_neighbors_cached.pickle'
    Consts.mask_for_each_pillar_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'mask_for_each_pillar_cached.pickle'


if __name__ == '__main__':

    perturbation_type = "REF5.3"
    exp_number = "69.1"
    config_path = "exp_" + exp_number + "_type_" + perturbation_type + "_mask_15_35_non-normalized_fixed.json"
    f = open("configs/" + perturbation_type + "/" + config_path)
    config_data = json.load(f)
    update_const_by_config(config_data)
    random.seed(10)

    WRITE_OUTPUT = True
    gc_df = get_gc_df()

    # Here new analyzes
    # gc_df_test = get_gc_df_test()

    # lst_origin = np.array([0.048, 0.17, 0.214, 0.146, 0.116, -0.035, 0.005])
    # lst_random = np.array([0.013, 0.056, 0.087, 0.094, 0.042, -0.007, -0.008])
    # f, ax = plt.subplots(figsize=(6, 6))
    # plt.plot(lst_random, lst_origin, 'bo')
    # axline([ax.get_xlim()[0], ax.get_ylim()[0]], [ax.get_xlim()[1], ax.get_ylim()[1]], ls='--')
    # # plt.plot(lst_random, 'ro', label='random')
    # plt.title('Average Correlation 9.4')
    # plt.ylabel('Neighbor pair correlation')
    # plt.xlabel('Random pair correlation')
    # plt.show()
    #
    # lst_origin = np.array([0.197, 0.203, 0.182, 0.03])
    # lst_random = np.array([0.017, 0.012, 0.033, 0.01])
    # f, ax = plt.subplots(figsize=(6, 6))
    # plt.plot(lst_random, lst_origin, 'bo')
    # axline([ax.get_xlim()[0], ax.get_ylim()[0]], [ax.get_xlim()[1], ax.get_ylim()[1]], ls='--')
    # # plt.plot(lst_random, 'ro', label='random')
    # plt.title('Average Correlation 13.2')
    # plt.ylabel('Neighbor pair correlation')
    # plt.xlabel('Random pair correlation')
    # plt.show()
    #
    # lst_origin = np.array([0.157, 0.104, 0.138, 0.202, 0.228])
    # lst_random = np.array([0.032, 0.031, 0.062, 0.001, 0.021])
    # f, ax = plt.subplots(figsize=(6, 6))
    # plt.plot(lst_random, lst_origin, 'bo')
    # axline([ax.get_xlim()[0], ax.get_ylim()[0]], [ax.get_xlim()[1], ax.get_ylim()[1]], ls='--')
    # # plt.plot(lst_random, 'ro', label='random')
    # plt.title('Average Correlation KD_9.4')
    # plt.ylabel('Neighbor pair correlation')
    # plt.xlabel('Random pair correlation')
    # # plt.legend()
    # plt.show()
    #
    #
    # lst_origin = np.array([0.305, 0.123, 0.202, 0.391, 0.27])
    # lst_random = np.array([-0.014, 0.021, 0.0008, 0.034, 0.002])
    # f, ax = plt.subplots(figsize=(6, 6))
    # plt.plot(lst_random, lst_origin, 'bo')
    # axline([ax.get_xlim()[0], ax.get_ylim()[0]], [ax.get_xlim()[1], ax.get_ylim()[1]], ls='--')
    # # plt.plot(lst_random, 'ro', label='random')
    # plt.title('Average Correlation KD_13.2')
    # plt.ylabel('Neighbor pair correlation')
    # plt.xlabel('Random pair correlation')
    # plt.show()
    #
    # lst_origin = np.array([0.444, 0.369, 0.19, 0.333])
    # lst_random = np.array([0, 0, 0.125, 0.248, 0.286])
    # plt.plot(lst_origin, 'b', label='13.2')
    # plt.plot(lst_random, 'r', label='KD_13.2')
    # plt.title('reciprocity 13.2 vs KD_13.2')
    # plt.ylabel('reciprocity')
    # plt.legend()
    # plt.show()
    #
    # factor9_4 = [0.197, 0.203, 0.182, 0.03]
    # factor_KD9_4 = [0.305, 0.123, 0.202, 0.391, 0.27]
    # stat_factor_94, pval_factor94 = t_test(factor9_4, factor_KD9_4)
    #
    #
    # output_path_type = "9_4"
    # targets_list_9_4 = ["9.4 - 04.1", "9.4 - 04.2", "9.4 - 06.1", "9.4 - 06.2", "9.4 - 11.1", "9.4 - 11.2", "9.4 - 12",
    #                     "KD9.4 - 02", "KD9.4 - 08", "KD9.4 - 11", "KD9.4 - 17", "KD9.4 - 53"]
    # targets_list_13_2 = ["13.2 - 01", "13.2 - 05", "13.2 - 06", "13.2 - 20",
    #                      "KD13.2 - 01", "KD13.2 - 02", "KD13.2 - 17", "KD13.2 - 46", "KD13.2 - 49.1"]
    # two_features = False
    # output_df = get_output_df(output_path_type)
    # features_correlations_heatmap(output_path_type)
    #
    # features = list(output_df.columns)
    # f = ['outwards_edges', 'heterogeneity']
    # leave_1_out_df = output_df
    # leave_1_out_df = leave_1_out_df.drop(columns=f, axis=1)
    # features_correlations_heatmap(output_path_type, custom_df=leave_1_out_df)
    # plot_2d_pca_components(targets_list_9_4, output_path_type, n_components=3, custom_df=leave_1_out_df)
    # pca, principle_comp = get_pca(output_path_type, n_components=3, custom_df=leave_1_out_df)
    # features_coefficient_heatmap(pca, output_path_type, custom_df=leave_1_out_df)
    # k_means(principle_comp, output_path_type, n_clusters=2, custom_df=leave_1_out_df)
    #
    # if two_features:
    #     features = list(output_df.columns)
    #     for i in range(len(features)):
    #         f1 = features[i]
    #         for j in range(i + 1, len(features)):
    #             f2 = features[j]
    #             two_features_df = output_df
    #             to_drop = [f for f in features if f != f1 and f != f2]
    #             two_features_df = two_features_df.drop(columns=to_drop, axis=1)
    #             plot_2d_pca_components(targets_list_13_2, output_path_type, n_components=2,
    #                                    custom_df=two_features_df)
    #             pca, principle_comp = get_pca(output_path_type, n_components=2, custom_df=two_features_df)
    #             # components_feature_weight(pca)
    #             features_coefficient_heatmap(pca, output_path_type, custom_df=two_features_df)
    #             k_means(principle_comp, output_path_type, n_clusters=2, custom_df=two_features_df)
    #
    # leave_1_out = True
    # if leave_1_out:
    #     features = list(output_df.columns)
    #     for i, f in enumerate(features):
    #         leave_1_out_df = output_df
    #         leave_1_out_df = leave_1_out_df.drop(columns=f, axis=1)
    #         features_correlations_heatmap(output_path_type, custom_df=leave_1_out_df)
    #         plot_2d_pca_components(targets_list_13_2, output_path_type, n_components=3, custom_df=leave_1_out_df)
    #         pca, principle_comp = get_pca(output_path_type, n_components=3, custom_df=leave_1_out_df)
    #         features_coefficient_heatmap(pca, output_path_type, custom_df=leave_1_out_df)
    #         k_means(principle_comp, output_path_type, n_clusters=2, custom_df=leave_1_out_df)

    # features_correlations_heatmap(output_path_type)
    # pca_number_of_components(output_path_type)
    # plot_2d_pca_components(targets_list_13_2, output_path_type, n_components=3)
    # pca, principle_comp = get_pca(output_path_type, n_components=3)
    # components_feature_weight(pca)
    # features_coefficient_heatmap(pca, output_path_type)
    # number_clusters_kmeans(principle_comp)
    # k_means(principle_comp, output_path_type, n_clusters=2)

    # factor9_4 = [0.25, 1.50, 0.25, 0.00, 0.00, 0.50, 1.00]
    # factor_KD9_4 = [0.67, 0.50, 0.00, 0.75, 2.5]
    # stat_factor_94, pval_factor94 = t_test(factor9_4, factor_KD9_4)
    #
    # gc_prob_94 = [0.198, 0.17, 0.071, 0.05, 0.068, 0.143, 0.063]
    # gc_prob_KD94 = [0.375, 0.154, 0.046, 0.209, 0.192]
    # stat_gc_94, pval_gc_94 = t_test(gc_prob_94, gc_prob_KD94)
    #
    # reciprocity_94 = [0.286, 0.333, 0, 0, 0.182, 0.333, 0.4]
    # reciprocity_KD94 = [0.333, 0, 0.4, 0.222, 0.4]
    # stat_rec_94, pval_rec_94 = t_test(reciprocity_94, reciprocity_KD94)
    #
    # heterogeneity_94 = [0.249, 0.204, 0.11, 0.063, 0.143, 0, 0]
    # heterogeneity_KD94 = [0.218, 0.112, 0.082, 0.16, 0.13]
    # stat_het_94, pval_het_94 = t_test(heterogeneity_94, heterogeneity_KD94)

    # gc_prob_132 = [0.27, 0.248, 0.108, 0.267]
    # gc_prob_KD132 = [0.076, 0.081, 0.186, 0.178, 0.175]
    # stat_gc_132, pval_gc_132 = t_test(gc_prob_132, gc_prob_KD132)
    #
    # factor13_2 = [1.00, 0.74, 0.67, 0.50]
    # factor_KD13_2 = [0.33, 2.00, 1.00, 0.96, 0.00]
    # stat_factor_132, pval_factor_132 = t_test(factor13_2, factor_KD13_2)
    #
    # reciprocity_132 = [0.444, 0.369, 0.19, 0.333]
    # reciprocity_KD132 = [0.00, 0.00, 0.125, 0.248, 0.286]
    # stat_rec_132, pval_rec_132 = t_test(reciprocity_132, reciprocity_KD132)
    #
    # heterogeneity_132 = [0.182, 0.154, 0.058, 0.158]
    # heterogeneity_KD132 = [0.025, 0.103, 0.132, 0.14, 0.17]
    # stat_het_132, pval_het_132 = t_test(heterogeneity_132, heterogeneity_KD132)
    # x=1

    # imgs = get_images('C:\\Users\\Sarit Hollander\\Desktop\\Study\\MSc\\Research\\Project\\Cell2CellComunicationAnalyzer\\Data\\Pillars\\9.4\\New-04-Airyscan Processing-1 MEF9.4.tif')
    # plt.imshow(imgs[-1])
    # plt.show()

    # time_res_concat = [19.94, 19.83, 19.96, 20.03, 31.33, 31.38, 19.87, 21.36]
    # time_res_height_53 = [19.94, 19.83, 19.96, 20.03]
    # time_res_height_132 = [31.33, 31.38, 19.87, 21.36]
    # gc_prob_5_concat = [0.429, 0.131, 0.2198, 0.571, 0.281, 0.371, 0.167, 0.5]
    # gc_prob_5_height_53 = [0.429, 0.131, 0.2198, 0.571]
    # gc_prob_5_height_132 = [0.281, 0.371, 0.167, 0.5]
    # gc_prob_1_concat = [0.143, 0.069, 0.104, 0.386, 0.181, 0.223, 0.063, 0.319]
    # gc_prob_1_height_53 = [0.143, 0.069, 0.104, 0.386]
    # gc_prob_1_height_132 = [0.181, 0.223, 0.063, 0.319]
    #
    # print("height: 5.3")
    # print("correlation of time res and gc prob - 5.3, 5% " + str(numpy.corrcoef(time_res_height_53, gc_prob_5_height_53)))
    # print("correlation of time res and gc prob - 5.3, 1% " + str(numpy.corrcoef(time_res_height_53, gc_prob_1_height_53)))
    # print("---------------------------------------------------")
    # print("height: 13.2")
    # print("correlation of time res and gc prob - 13.2, 5% " + str(numpy.corrcoef(time_res_height_132, gc_prob_5_height_132)))
    # print("correlation of time res and gc prob - 13.2, 1% " + str(numpy.corrcoef(time_res_height_132, gc_prob_1_height_132)))
    # print("--------------------------------------------------")
    # print("concat")
    # print("correlation of time res and gc prob - concat, 5% " + str(numpy.corrcoef(time_res_concat, gc_prob_5_concat)))
    # print("correlation of time res and gc prob - concat, 1% " + str(numpy.corrcoef(time_res_concat, gc_prob_1_concat)))

    # x = get_pillar_to_intensities(get_images_path())
    # alive_p = get_alive_pillars(get_mask_for_each_pillar())
    # alive_p2i = {k: v for k, v in x.items() if k in alive_p}
    # p_names = [str(k) for k in alive_p2i.keys()]
    # df = pd.DataFrame.from_dict(alive_p2i)
    # df.columns = p_names
    # df.to_pickle('./features output/time_series_exp05_type13-2.pkl')
    # y=1

    operations = config_data.get("operations", [])
    for op in operations:
        op_key = list(op.keys())[0]
        op_values = op[op_key]
        if op_key == 'correlation_plot':
            correlation_plot(
                op_values.get("only_alive", True),
                op_values.get("neighbors", 'all'),
                op_values.get("alive_correlation_type", 'symmetric')
            )
        elif op_key == "correlation_histogram":
            alive = op_values.get("alive", True)
            if alive:
                correlations = get_alive_pillars_symmetric_correlation()
                print("Correlation between all alive pillars:")
            else:
                correlations = get_all_pillars_correlations()
                print("Correlation between all pillars:")
            correlation_histogram(correlations)
        elif op_key == "neighbors_correlation_histogram":
            neighbors = op_values.get("neighbors", "real_neighbors")
            if neighbors == "real_neighbors":
                neighbors_dict = get_alive_pillars_to_alive_neighbors()
                original = True
            elif neighbors == "random":
                neighbors_dict = get_random_neighbors()
                original = False
            else:
                neighbors_dict = {}
                original = True
            if original:
                avg_correlation = neighbors_correlation_histogram(get_alive_pillars_symmetric_correlation(),
                                                                  neighbors_dict,
                                                                  original_neighbors=original)
            else:
                neighbors_correlation_histogram(get_alive_pillars_symmetric_correlation(), neighbors_dict,
                                                original_neighbors=original)
        elif op_key == "compare_neighbors_corr_histogram_random_vs_real":
            random_amount = op_values.get("random_amount", 5)
            compare_neighbors_corr_histogram_random_vs_real(random_amount)
        elif op_key == "show_pillars_mask":
            show_last_image_masked(pillars_mask=build_pillars_mask())
        elif op_key == "edges_distribution_plots":
            edges_distribution_plots(gc_df)
        elif op_key == "gc_graph":
            non_stat_lst, passed_stationary = get_non_stationary_pillars_lst()
            _, _, inwards_edges, outwards_edges, inwards_percentage, outwards_percentage = get_number_of_inwards_outwards_gc_edges(
                gc_df)
            build_gc_directed_graph(gc_df, non_stationary_pillars=non_stat_lst, inwards=inwards_edges,
                                    outwards=outwards_edges, random_neighbors=False)
        # elif op_key == "in_out_gc_edges":
        #     _, _, in_lst, out_lst = get_number_of_inwards_outwards_gc_edges(gc_df)
        #     build_gc_directed_graph(gc_df, edges_direction_lst=in_lst, draw=True)
        #     build_gc_directed_graph(gc_df, edges_direction_lst=out_lst, draw=True)
        elif op_key == "gc_edge_prob":
            gc_edge_prob = probability_for_gc_edge(gc_df, random_neighbors=False)
        elif op_key == "gc_edge_prob_original_vs_random":
            gc_edge_probability_original_vs_random(gc_df)
        elif op_key == "in_out_degree":
            in_d, out_d, _ = get_pillar_in_out_degree(gc_df)
            in_out_degree_distribution(in_d, out_d)
        elif op_key == "reciprocity":
            reciprocity = get_network_reciprocity(gc_df)
        elif op_key == "heterogeneity":
            heterogeneity = get_network_heterogeneity(gc_df)

        # features_name_list = ['avg_correlation', 'passed_stationary', 'inwards_edges', 'outwards_edges', 'gc_edge_prob',
        #                       'reciprocity', 'heterogeneity']
    if WRITE_OUTPUT:
        features_dict = {'avg_correlation': avg_correlation,
                         'passed_stationary': passed_stationary,
                         'inwards_edges': inwards_percentage,
                         'outwards_edges': outwards_percentage,
                         'gc_edge_prob': gc_edge_prob,
                         'reciprocity': reciprocity,
                         'heterogeneity': heterogeneity
                         }
        experiment_id = config_data.get("experiment")['id']
        perturbation = config_data.get("perturbation")['type']
        output_path_type = config_data.get("output_path_type")
        index = perturbation + "_" + experiment_id
        output_path = get_output_path(output_path_type)
        if os.path.exists(output_path):
            output_df = pd.read_csv(output_path, index_col=0)
            features_df = pd.DataFrame(features_dict, index=[index])
            output_df = output_df.append(features_df)
            output_df.to_csv(output_path)
        else:
            output_df = pd.DataFrame(features_dict, index=[index])
            output_df.to_csv(output_path)
