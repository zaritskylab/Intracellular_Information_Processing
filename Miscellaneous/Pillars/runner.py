import numpy

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

    # Update mask radius
    mask_radius = config_data.get('mask_radius', {
        "small_radius": 15,
        "large_radius": 35
    })

    small_mask_radius_ratio = mask_radius['small_radius'] / 20
    large_mast_radius_ratio = mask_radius['large_radius'] / 20

    Consts.percentage_from_perfect_circle_mask = config_data.get('percentage_from_perfect_circle_mask', 1)

    Consts.alive_centers_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + 'alive_centers_cached.pickle'

    radius = get_circle_radius(config_data)
    Consts.SMALL_MASK_RADIUS = math.floor(radius * small_mask_radius_ratio)
    Consts.LARGE_MASK_RADIUS = math.floor(radius * large_mast_radius_ratio)

    # Consts.SMALL_MASK_RADIUS = mask_radius['small_radius']
    # Consts.LARGE_MASK_RADIUS = mask_radius['large_radius']

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
    Consts.gc_df_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'gc_df_cached.pickle'
    Consts.alive_pillars_corr_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'alive_pillars_corr_cached.pickle'
    Consts.centers_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'centers_cached.pickle'
    Consts.pillar_to_neighbors_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'pillar_to_neighbors_cached.pickle'
    Consts.mask_for_each_pillar_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'mask_for_each_pillar_cached.pickle'



if __name__ == '__main__':

    perturbation_type = "13.2"
    exp_number = "01"
    config_path = "exp_" + exp_number + "_type_" + perturbation_type + "_mask_15_35_non-normalized_fixed.json"
    f = open("configs/" + perturbation_type + "/" + config_path)
    config_data = json.load(f)
    update_const_by_config(config_data)

    WRITE_OUTPUT = False
    gc_df = get_gc_df()

    # Here new analyzes
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
    # print("---------------------------------------------------")
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
    # features_correlations_heatmap()
    # pca_number_of_components()
    # plot_2d_pca_components(n_components=3)
    # pca, principle_comp = get_pca(n_components=3)
    # components_feature_weight(pca)
    # features_coefficient_heatmap(pca)

    # exp_13_2 = ["01", "05", "06", "20"]
    # original_13_2 = [0.281, 0.371, 0.167, 0.5]
    # random_13_2 = [0.192, 0.265, 0.206, 0.47]
    # fig, ax = plt.subplots()
    # ax.scatter(exp_13_2, original_13_2, label="original")
    # ax.scatter(exp_13_2, random_13_2, label="random")
    # plt.ylabel("Edge Probability")
    # plt.xlabel("Experiment")
    # plt.title("GC Edge Probability - Original vs. Random Neighbors (13.2)")
    # ax.legend()
    # plt.show()
    # exp_9_4 = ["08", "09", "12", "30"]
    # original_9_4 = [0.429, 0.131, 0.2198, 0.571]
    # random_9_4 = [0.414, 0.117, 0.345, 0.333]
    # fig, ax = plt.subplots()
    # ax.scatter(exp_9_4, original_9_4, label="original")
    # ax.scatter(exp_9_4, random_9_4, label="random")
    # plt.ylabel("Edge Probability")
    # plt.xlabel("Experiment")
    # plt.title("GC Edge Probability - Original vs. Random Neighbors (5.3)")
    # ax.legend()
    # plt.show()

    # non_stat_lst, passed_stationary = get_non_stationary_pillars_lst()
    # _, _, inwards_edges, outwards_edges, inwards_percentage, outwards_percentage = get_number_of_inwards_outwards_gc_edges(
    #     gc_df)
    # _, pvals_originals = build_gc_directed_graph(gc_df, non_stationary_pillars=non_stat_lst, inwards=inwards_edges,
    #                         outwards=outwards_edges, random_neighbors=False, draw=False)
    # _, pvals_random = build_gc_directed_graph(gc_df, non_stationary_pillars=non_stat_lst, inwards=inwards_edges,
    #                                    outwards=outwards_edges, random_neighbors=True, draw=False)
    # fig, ax1 = plt.subplots()
    # g = sns.histplot([pvals_originals, pvals_random], kde=True, stat='probability', element="step", multiple="stack")
    # legend = ax1.get_legend()
    # handles = legend.legendHandles
    # legend.remove()
    # ax1.legend(handles, ['original', 'random'])
    # # sns.displot(data=pvals_random, kind="kde")
    # plt.xlabel("P-values")
    # plt.show()



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
            _, _, inwards_edges, outwards_edges, inwards_percentage, outwards_percentage = get_number_of_inwards_outwards_gc_edges(gc_df)
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
        index = perturbation + "_" + experiment_id
        output_path = get_output_path()
        if os.path.exists(output_path):
            output_df = pd.read_csv(output_path, index_col=0)
            features_df = pd.DataFrame(features_dict, index=[index])
            output_df = output_df.append(features_df)
            output_df.to_csv(output_path)
        else:
            output_df = pd.DataFrame(features_dict, index=[index])
            output_df.to_csv(output_path)
