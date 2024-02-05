import time

import numpy
from matplotlib.pyplot import axline
from pathlib import Path

from Pillars.visualization import *
from Pillars.granger_causality_old import *
from Pillars.repositioning import *
from Pillars.analyzer import *
from pathlib import Path
from Pillars.runner_helper import *
from Pillars.cross_correlation import *

import json
import math


def update_const_by_config(config_data):
    # Update experiment configurations
    experiment_data = config_data["experiment"]
    experiment_id = experiment_data['id']
    experiment_tif_path = experiment_data['tif_path']
    perturbation = config_data["perturbation"]
    perturbation_type = perturbation['type']
    Consts.ignore_first_image = config_data.get('ignore_first_image', False)
    Consts.ignore_last_image = config_data.get('ignore_last_image', False)
    Consts.tagged_centers = config_data.get('tagged_centers')
    if Consts.tagged_centers is not None:
        Consts.tagged_centers = [eval(tup) for tup in Consts.tagged_centers]
        Consts.ignore_centers = config_data.get('ignore_centers')
    if Consts.ignore_centers is not None:
        Consts.ignore_centers = [eval(tup) for tup in Consts.ignore_centers]
    if 'metadata' in config_data and 'is_spreading' in config_data["metadata"]:
        Consts.is_spreading = config_data["metadata"]["is_spreading"]
    else:
        Consts.is_spreading = True

    Consts.USE_CACHE = config_data.get('use_cache', True)

    Consts.fixed_images_path = Consts.PILLARS + '\\FixedImages\\Fixed_' + perturbation_type + '\\new_fixed_' + experiment_id + '.tif'
    if not Path(Consts.fixed_images_path).exists():
        reposition(Consts.PILLARS + '\\' + perturbation_type + experiment_tif_path, Consts.fixed_images_path)
    Consts.images_path = Consts.PILLARS + experiment_tif_path
    Consts.last_image_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/last_image_' + experiment_id + '.npy'

    images = get_images(get_images_path())

    if not Path(Consts.last_image_path).exists():
        os.makedirs(os.path.dirname(Consts.last_image_path), exist_ok=True)
        with open(Consts.last_image_path, 'wb') as f:
            np.save(f, images[-1])

    Consts.IMAGE_SIZE_ROWS = len(images[-1])
    Consts.IMAGE_SIZE_COLS = len(images[-1][0])

    Consts.normalized = config_data.get('normalized', False)
    Consts.fixed = config_data.get('fixed', True)
    Consts.use_otsu = config_data.get('use_otsu', True)
    Consts.ALL_TAGGED_ALWAYS_ALIVE = config_data.get('all_tagged_always_alive', False)
    Consts.pixel_to_whiten = config_data.get('pixel_to_whiten', 10)
    Consts.MAX_DISTANCE_PILLAR_FIXED = config_data.get('max_distance_pillar_fixed', 11)
    Consts.INTENSITIES_RATIO_OUTER_INNER = config_data.get('outer_inner_intensity_ratio', 1.9)
    radius = get_circle_radius(config_data)
    Consts.CIRCLE_RADIUS_FOR_MASK_CALCULATION = radius
    Consts.CIRCLE_RADIUS = round(radius)

    # Update mask radius
    mask_radius = config_data.get('mask_radius', {
        "small_radius": 15,
        "large_radius": 35
    })

    mask_radiuses_after_ratio = get_mask_radiuses(mask_radius)

    Consts.SMALL_MASK_RADIUS = mask_radiuses_after_ratio['small']
    Consts.LARGE_MASK_RADIUS = mask_radiuses_after_ratio['large']

    # circle_validation = config_data.get('circle_validation', False)
    # if circle_validation:

    # Consts.FIND_BETTER_CENTER_IN_RANGE = math.ceil(Consts.CIRCLE_RADIUS / 3)
    Consts.FIND_BETTER_CENTER_IN_RANGE = Consts.CIRCLE_RADIUS
    Consts.percentage_from_perfect_circle_mask = config_data.get('percentage_from_perfect_circle_mask', 1)
    Consts.MAX_CIRCLE_AREA = (math.pi * Consts.CIRCLE_RADIUS ** 2) * 2
    micron = config_data["metadata"]["micron"]
    validation_ratio = int(micron / Consts.RELATIVE_TO)
    if validation_ratio == 0:
        validation_ratio = micron / Consts.RELATIVE_TO
    Consts.CHECK_VALID_CENTER = math.ceil(Consts.CHECK_VALID_CENTER / validation_ratio)
    Consts.CIRCLE_INSIDE_VALIDATE_SEARCH_LENGTH = config_data.get("CIRCLE_INSIDE_VALIDATE_SEARCH_LENGTH",
                                                                  Consts.CIRCLE_RADIUS - Consts.CHECK_VALID_CENTER)
    Consts.CIRCLE_OUTSIDE_VALIDATE_SEARCH_LENGTH = Consts.CIRCLE_RADIUS + Consts.CHECK_VALID_CENTER

    # TODO: add Consts.CORRELATION to path
    # TODO: if we want to cache by other params (like mask radius) we need to add them to the path
    path_postfix = str(Consts.SMALL_MASK_RADIUS) + '_' + str(Consts.LARGE_MASK_RADIUS) + '_fully_' + str(
        Consts.inner_cell) + '_fixed_' + str(Consts.fixed) + '_normalized_' + str(Consts.normalized) + '_'

    # Update caches path.
    Consts.pillar_to_intensities_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'pillar_to_intensities_cached.pickle'
    Consts.pillar_to_intensities_norm_by_noise_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'pillar_to_intensities_norm_by_noise_cached.pickle'
    Consts.inner_pillar_noise_series_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'inner_pillar_noise_series_cached.pickle'
    Consts.correlation_alive_normalized_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'alive_pillar_correlation_normalized_cached.pickle'
    Consts.correlation_alive_not_normalized_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'alive_pillar_correlation_cached.pickle'
    Consts.all_pillars_correlation_normalized_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'all_pillar_correlation_normalized_cached.pickle'
    Consts.all_pillars_correlation_not_normalized_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'all_pillar_correlation_cached.pickle'
    Consts.gc_df_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'gc_df_cached.pickle'
    Consts.alive_pillars_sym_corr_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'alive_pillars_corr_cached.pickle'
    Consts.alive_pillars_sym_corr_norm_by_inner_p_noise_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'alive_pillars_corr_norm_by_noise_cached.pickle'
    Consts.mask_for_each_pillar_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'mask_for_each_pillar_cached.pickle'
    Consts.gc_graph_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'gc_cached.pickle'
    Consts.alive_pillars_correlations_frame_windows_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'alive_pillars_correlations_frame_windows_cache.pickle'
    Consts.alive_pillars_correlations_with_running_frame_windows_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'alive_pillars_correlations_with_running_frame_windows_cache.pickle'

    Consts.pillars_alive_location_by_frame_to_gif_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'pillars_alive_location_by_frame_to_gif_cache.pickle'
    Consts.alive_pillars_by_frame_reposition_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'alive_pillars_by_frame_reposition_cache.pickle'
    Consts.frame2alive_pillars_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'frames2alive_pillars_cached.pickle'
    Consts.last_img_alive_centers_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'last_img_seen_centers_cache.pickle'
    Consts.frame2pillar_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'frames2pillars_cached.pickle'
    Consts.pillar_to_neighbors_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'pillar_to_neighbors_cached.pickle'
    Consts.centers_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'centers_cached.pickle'
    Consts.alive_pillars_to_alive_neighbors_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'alive_pillars_to_alive_neighbors_cache.pickle'
    Consts.alive_pillars_overall = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'alive_pillars_overall_cache.pickle'
    Consts.alive_center_ids_by_frame_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'alive_center_ids_by_frame_cache.pickle'
    Consts.alive_center_real_locations_by_frame_cache_path = '../SavedPillarsData/' + perturbation_type + '/SavedPillarsData_' + experiment_id + '/' + path_postfix + 'alive_center_real_locations_by_frame_cache.pickle'


def run_config(config_name):
    nbrs_avg_correlation = None
    non_nbrs_avg_correlation = None
    passed_stationary = None
    total_edges = None
    inwards_percentage = None
    outwards_percentage = None
    out_in_factor = None
    gc_edge_prob = None
    avg_random_gc_edge_prob = None
    std = None
    reciprocity = None
    heterogeneity = None
    num_of_neighboring_pairs_in_top_corrs = None
    neighbors_avg_movement_correlation = None
    not_neighbors_avg_movement_correlation = None
    avg_intensity = None
    map_radius_to_corr = None
    map_radius_to_norm_corr_by_0_10 = None
    map_radius_to_norm_corr_by_0_15 = None

    f = open("../configs/" + config_name)
    config_data = json.load(f)
    update_const_by_config(config_data)
    random.seed(10)
    gc_df = None
    operations = config_data.get("operations", [])

    # TODO: delete
    # if os.path.exists(Consts.pillar_to_intensities_norm_by_noise_cache_path):
    #     os.remove(Consts.pillar_to_intensities_norm_by_noise_cache_path)

    # show_last_image_masked(pillars_mask=build_pillars_mask(get_all_center_generated_ids()), save_mask=False, frame=get_images(get_images_path())[174])
    # get_alive_pillars_to_alive_neighbors()
    # return

    # show_peripheral_pillars_in_video()
    # x = 1
    # return get_neighbors_avg_correlation(get_alive_pillars_symmetric_correlation(),
    #                                                             get_alive_pillars_to_alive_neighbors())

    # p_to_intns = get_pillar_to_intensity_norm_by_inner_pillar_noise()
    # core, periph = get_core_periphery_pillars()
    # return p_to_intns

    # p_to_diff_intens = differencing_time_series(get_overall_alive_pillars_to_intensities())
    # stationary_p_to_intens, non_stationary_pillars = get_stationary_and_non_stationary_pillars(p_to_diff_intens)
    # gc_dict = perform_granger_test(stationary_p_to_intens, maxlag=3)
    # G = build_gc_graph(gc_dict, threshold=0.05)
    # plot_graph(G)
    # in_degree_nodes = in_degree_centrality(G)
    # plot_graph_in_degree_label(G, in_degree_nodes, non_stationary_pillars)
    # out_degree_nodes = out_degree_centrality(G)
    # plot_graph_out_degree_label(G, out_degree_nodes, non_stationary_pillars)
    # connected_component(G)
    # plot_main_in_out_centrality_pillars(G, non_stationary_pillars, in_degree_nodes, out_degree_nodes)

    # G_rand_nbrs = build_pillars_graph(random_neighbors=True, draw=False)
    # ns_rand, strong_nodes_rand, _ = nodes_strengths(G_rand_nbrs, draw=False)
    # strong_nodes_avg_distance_rand = strong_nodes_avg_distance_from_center(G_rand_nbrs, alive_centers=get_seen_centers_for_mask(),
    #                                                                   strong_nodes=strong_nodes_rand, all_nodes_strength=ns_rand,
    #                                                                   draw=False)
    G = build_pillars_graph(random_neighbors=False, shuffle_ts=False, draw=False)
    # ns, strong_nodes, _ = nodes_strengths(G, draw=False, color_map_nodes=False)
    # clustering_strong_nodes_by_Louvain(G, ns, strong_nodes, draw=False)
    # centrality_measure_strong_nodes(G, ns, strong_nodes, draw=True)
    # strong_nodes_avg_distance = strong_nodes_avg_distance_from_center(G, alive_centers=get_seen_centers_for_mask(), strong_nodes=strong_nodes, all_nodes_strength=ns, draw=False)
    # strong_nodes_avg_distance = strong_nodes_avg_distance_from_center_in_different_distance_categories(G, alive_centers=get_seen_centers_for_mask(), strong_nodes=strong_nodes, draw=False)
    # plot_node_strengths_distribution(ns)
    # strong_nodes_avg_distance = strong_nodes_avg_distance_from_center_by_hops(G, alive_centers=get_seen_centers_for_mask(), strong_nodes=strong_nodes, removed_nodes=removed_nodes)
    # return strong_nodes_avg_distance, strong_nodes_avg_distance_rand
    # cc, largest_cc = strong_nodes_connected_components(G, ns, strong_nodes, draw=True)
    # return len(largest_cc)
    # return cc, largest_cc
    # return strong_nodes_avg_distance, ns, strong_nodes
    # return ns
    # nbrs_sim_dict = nbrs_nodes_similarity_by_strength(G, ns)
    # all_nodes_sim_dict = non_nbrs_similarity_by_strength(G, ns)
    # core_pillars, periphery_pillars = get_core_periphery_pillars()
    # core_strength, periphery_strength = get_core_periphery_values_by_strength(core_pillars, periphery_pillars, G, ns)
    # nbrs_sims = [v[1] for v in nbrs_sim_dict.values()]
    # core_sims, periphery_sims, border_sims = get_core_periphery_values_by_similarity(core_pillars, periphery_pillars, nbrs_sim_dict)
    # core_corrs, periphery_corrs, border_corrs = get_core_periphery_values_by_correlation(core_pillars, periphery_pillars)
    # plot_core_periphery_pillars_in_graph(G, core_pillars, periphery_pillars)
    # plot_distribution_similarity_core_vs_periphery(core_sims, periphery_sims, np.mean(nbrs_sims))
    # similarities_above_avg_graph(G_sims, pillars_pair_to_sim_above_avg)
    # nbrs_vals = nbrs_sim_dict.values()
    # noon_nbrs_vals = all_nodes_sim_dict.values()
    # nbrs_sim = [v[1] for v in nbrs_vals]
    # non_nbrs_sim = [v[1] for v in noon_nbrs_vals]
    # avg_nbrs_sim = np.mean(nbrs_sim)
    # avg_non_nbrs_sim = np.mean(non_nbrs_sim)
    # return avg_nbrs_sim, avg_non_nbrs_sim
    # return nbrs_sim, non_nbrs_sim
    # nbrs_to_corrs_dict = get_neighbors_to_correlation(get_alive_pillars_symmetric_correlation(),
    #                               get_alive_pillars_to_alive_neighbors())
    # non_nbrs_to_corrs_dict = get_non_neighbors_to_correlation_dict(get_alive_pillars_symmetric_correlation(),
    #                               get_alive_pillars_to_alive_neighbors())
    # level_to_similarities = nbrs_level_to_correlation(G, nbrs_to_corrs_dict, non_nbrs_to_corrs_dict)
    # plot_avg_similarity_by_nbrhood_degree(level_to_similarities)
    # return level_to_similarities
    # return avg_strength, avg_sim
    # p_2_sim_dict = get_pillar_to_avg_similarity_dict(nbrs_sim_dict, all_nodes_sim_dict)
    # return p_2_sim_dict
    # return test_strong_nodes_distance_significance(num_permutations=100, alpha=0.05)
    # return test_strong_nodes_clusters_louvain_significance(num_permutations=1000, alpha=0.05)
    # return test_strong_nodes_distance_hops_significance(num_permutations=1000, alpha=0.05)
    # return test_strong_nodes_number_of_cc_significance(num_permutations=100, alpha=0.05)
    # return test_strong_nodes_largest_cc_significance(num_permutations=100, alpha=0.05)
    # return test_avg_similarity_significance(num_permutations=500, alpha=0.05)
    # return core_strength, periphery_strength
    # return core_corrs, periphery_corrs, border_corrs
    # return nbrs_sims, core_sims, periphery_sims, border_sims
    # return np.mean(core_sims), np.mean(periphery_sims)
    # return test_avg_core_periphery_similarity_significance(num_permutations=500, alpha=0.05)
    # correlations = cross_correlations(get_overall_alive_pillars_to_intensities(), get_alive_pillars_to_alive_neighbors(), max_lag=3)
    # lag_correlations_heatmap(correlations)
    # lag_to_avg_corr_dict = cross_correlation_avg_each_lag(correlations)
    # plot_total_avg_correlation_in_lag(lag_to_avg_corr_dict)
    # peak_lags = identify_peaks(correlations)
    # DG = plot_cross_correlation_directed_graph(peak_lags)
    # lag_distribution_plot(peak_lags)
    # plot_peak_correlation_vs_lag(peak_lags)
    # leading_nodes, lagging_nodes = categorize_patterns(peak_lags)
    # plot_nodes_correlations_through_lags(G, strong_nodes, correlations)
    # plot_only_in_or_only_out_degree_in_graph(DG)
    # return lag_to_avg_corr_dict
    # pillars_strength_by_intens_similarity_in_time(get_pillar_to_intensity_norm_by_inner_pillar_noise(), get_alive_pillars_to_alive_neighbors(), G, n_frames_to_avg=20, show_above_avg=True)
    # return

    # nbrs_and_non_nbrs_corrs_gistogram(get_alive_pillars_symmetric_correlation(), get_alive_pillars_to_alive_neighbors())
    # superpixel()
    # G = build_graph(random_neighbors=False, shuffle_ts=False, draw=False)
    # G = build_fully_connected_graph(draw=False)
    # louvain_cluster_nodes(G, draw=True)
    # gcn(G)
    pillar_ids = list(get_alive_pillar_ids_overall_v3())
    p_to_intens = get_pillar_to_intensity_norm_by_inner_pillar_noise()
    # norm_p_to_intens = min_max_intensity_normalization(p_to_intens)

    # TODO: intensity animation
    # show_superpixel_intensity_gif(n_segments)
    # TODO: avg time series of each cluster
    # plot_avg_cluster_time_series(segments, matrix_3d, pillars_id_matrix_2d)
    # TODO: superpixel on video with stuck of 20 frames
    # for i in range(0, len(list(p_to_intens.values())[0])+1, 20):
    #     start = i
    #     end = i + 20
    #     if start == 100:
    #         end = 122
        # TODO: the problem is that its always picks the same centroids and never change them. why?
        # print("start frame:", start)
        # print("end frame:", end)
        # val = int(np.ceil(len(pillar_ids) / 15))
        # n_vals = [[val]]
    # n_vals = [[6]]
    custom_p_to_intensity = None
    n_vals = [[int(np.ceil(len(pillar_ids) / 22))], [int(np.ceil(len(pillar_ids) / 15))], [int(np.ceil(len(pillar_ids) / 11))], [int(np.ceil(len(pillar_ids) / 8))]]
    for n in n_vals:
        print("number of clusters:", n[0], "avg number of pillars in cluster:", str(len(pillar_ids)/n[0]))
        print("######### No ts shuffle #########")
        # custom_p_to_intensity = {k: v[start:end] for k,v in p_to_intens.items()}
        intra_variance, intra_dtw_distance, intra_vec_dist, inter_dtw_distance, inter_vec_dist = superpixel_segmentation_evaluation(n, shuffle_ts=False, custom_p_to_intensity=custom_p_to_intensity, channel='correlation', save_clusters_fig=n)
        print("intra variance:", intra_variance)
        print("intra DTW distance:", intra_dtw_distance)
        print("intra vec distance:", intra_vec_dist)
        print("inter DTW distance:", inter_dtw_distance)
        print("inter vec distance:", inter_vec_dist)
        print("######### ts shuffle #########")
        num_permutations = 100
        alpha = 0.05
        permuted_test_intra_var_statistics = np.zeros(num_permutations)
        permuted_test_intra_dtw_distance_statistics = np.zeros(num_permutations)
        permuted_test_intra_vec_dist_statistics = np.zeros(num_permutations)
        permuted_test_inter_dtw_distance_statistics = np.zeros(num_permutations)
        permuted_test_inter_vec_dist_statistics = np.zeros(num_permutations)
        for i in range(num_permutations):
            p_to_intensity = get_pillar_to_intensity_for_shuffle_ts()
            # custom_p_to_intensity = {k: v[start:end] for k,v in p_to_intensity.items()}
            rand_intra_variance, rand_intra_dtw_distance, rand_intra_vec_dist, rand_inter_dtw_distance, rand_inter_vec_dist = superpixel_segmentation_evaluation(n, shuffle_ts=True, custom_p_to_intensity=custom_p_to_intensity, channel='correlation')
            permuted_test_intra_var_statistics[i] = list(rand_intra_variance.values())[0]
            permuted_test_intra_dtw_distance_statistics[i] = list(rand_intra_dtw_distance.values())[0]
            permuted_test_intra_vec_dist_statistics[i] = list(rand_intra_vec_dist.values())[0]
            permuted_test_inter_dtw_distance_statistics[i] = list(rand_inter_dtw_distance.values())[0]
            permuted_test_inter_vec_dist_statistics[i] = list(rand_inter_vec_dist.values())[0]

        # Calculate the p-value
        p_value = np.sum(permuted_test_intra_var_statistics <= list(intra_variance.values())[0]) / num_permutations
        print("p-value for intra variance:", p_value)
        if p_value < alpha:
            print("Reject the null hypothesis for intra var with n=" + str(n[0]) + ". The original values are statistically significant.")
        else:
            print("Fail to reject the null hypothesis for intra var n=" + str(n[0]) + ".")

        p_value = np.sum(permuted_test_intra_dtw_distance_statistics <= list(intra_dtw_distance.values())[0]) / num_permutations
        print("p-value for intra DTW distance:", p_value)
        if p_value < alpha:
            print("Reject the null hypothesis for intra DTW distance with n=" + str(n[0]) + ". The original values are statistically significant.")
        else:
            print("Fail to reject the null hypothesis for intra DTW distance n=" + str(n[0]) + ".")

        p_value = np.sum(permuted_test_intra_vec_dist_statistics <= list(intra_vec_dist.values())[0]) / num_permutations
        print("p-value for intra vector distance:", p_value)
        if p_value < alpha:
            print("Reject the null hypothesis for intra vector distance with n=" + str(
                n[0]) + ". The original values are statistically significant.")
        else:
            print("Fail to reject the null hypothesis for intra vector distance n=" + str(n[0]) + ".")

        p_value = np.sum(permuted_test_inter_dtw_distance_statistics >= list(inter_dtw_distance.values())[0]) / num_permutations
        print("p-value for inter DTW distance:", p_value)
        if p_value < alpha:
            print("Reject the null hypothesis for inter DTW distance with n=" + str(n[0]) + ". The original values are statistically significant.")
        else:
            print("Fail to reject the null hypothesis for inter DTW distance n=" + str(n[0]) + ".")

        p_value = np.sum(permuted_test_inter_vec_dist_statistics >= list(inter_vec_dist.values())[0]) / num_permutations
        print("p-value for inter vector distance:", p_value)
        if p_value < alpha:
            print("Reject the null hypothesis for inter vector distance with n=" + str(
                n[0]) + ". The original values are statistically significant.")
        else:
            print("Fail to reject the null hypothesis for inter vector distance n=" + str(n[0]) + ".")
    return

    # plot_significance_bar(53, 81, 34, 72, ['5.3', '13.2'])
    # plot_pillar_time_series((511,537), temporal_res=19.87, img_res=0.0519938, inner_pillar=False)
    # plot_pillar_time_series((511, 537), temporal_res=19.87, img_res=0.0519938, inner_pillar=True)
    # plot_pillar_time_series((511, 537), temporal_res=19.87, img_res=00.0519938, inner_pillar=False, ring_vs_inner=True)
    # return


    # pillars_under_diagonal = []
    # pillars = list(p_2_sim_dict.keys())
    # nbrs_sim = [tup[0] for tup in list(p_2_sim_dict.values())]
    # non_nbrs_sim = [tup[1] for tup in list(p_2_sim_dict.values())]
    # for i in range(len(pillars)):
    #     if nbrs_sim[i] < non_nbrs_sim[i]:
    #         pillars_under_diagonal.append(pillars[i])
    #
    # ns, strong_nodes, _ = nodes_strengths(G, draw=True, color_map_nodes=False, group_nodes_colored=pillars_under_diagonal)

    # pillar_strength = {}
    # for node_idx in ns:
    #     coor = G.nodes[node_idx]['pos']
    #     pillar_strength[coor] = ns[node_idx]
    # total_avg_intensity, pillars_avg_intens = get_cell_avg_intensity()
    # # return total_avg_intensity, pillars_avg_intens, pillar_strength
    # p_to_intens = get_overall_alive_pillars_to_intensities()
    # avg_ts = get_cell_avg_ts()
    # return p_to_intens, avg_ts, total_avg_intensity, pillars_avg_intens, pillar_strength
    # pillars_movements_dict = get_alive_centers_movements()
    # movement_corr_df = get_pillars_movement_correlation_df(pillars_movements_dict)
    # intensity_corr_df = get_alive_pillars_symmetric_correlation()
    # return total_movements_percentage(pillars_movements_dict)
    # plot_pair_of_pillars_movement_corr_and_intensity_corr(movement_corr_df, intensity_corr_df, neighbors_only=False)
    # get_avg_correlation_pillars_intensity_movement()
    # get_alive_centers_movements()
    # plot_average_intensity_by_distance()
    # plot_pillars_average_intensity_by_movement()

    # nbrs_avg_corr, nbrs_corrs_list = get_neighbors_avg_correlation(get_alive_pillars_symmetric_correlation(),
    #                                                    get_alive_pillars_to_alive_neighbors())
    # non_nbrs_avg_corr, non_nbrs_correlation_list = get_non_neighbors_mean_correlation(get_alive_pillars_symmetric_correlation(),
    #                                                                   get_alive_pillars_to_alive_neighbors())
    #
    # x = 1
    #

    # print(get_avg_correlation_pillars_intensity_movement_peripheral_vs_central())
    # show_peripheral_pillars_in_video(get_peripheral_and_center_pillars_by_frame_according_revealing_pillars_and_nbrs(pillars_frame_zero_are_central=True))
    # get_avg_correlation_pillars_intensity_movement_peripheral_vs_central(get_peripheral_and_center_pillars_by_frame_according_revealing_pillars_and_nbrs(pillars_frame_zero_are_central=True))
    # frame_to_peripheral_center_dict = get_peripheral_and_center_pillars_by_frame()
    # last_frame = list(frame_to_peripheral_center_dict.keys())[-1]
    # periph_pillars = frame_to_peripheral_center_dict[last_frame]["peripherals"]
    # plt.imshow(get_last_image(), cmap=plt.cm.gray)
    # y = [p[0] for p in periph_pillars]
    # x = [p[1] for p in periph_pillars]
    # scatter_size = [3 for center in periph_pillars]
    #
    # plt.scatter(x, y, s=scatter_size)
    # plt.text(10, 10, str(len(periph_pillars)), fontsize=8)
    # plt.show()
    # pillar2middle_img_steps = get_pillar2_middle_img_steps(get_alive_pillars_to_alive_neighbors())
    # show_nbrs_distance_graph(get_alive_pillars_to_alive_neighbors(), pillar2middle_img_steps)
    # return
    # return get_neighbours_correlations_by_distance_from_cell_center()
    # first_half_corrs, second_half_corrs, overall_corrs = get_correlations_in_first_and_second_half_of_exp()
    # mean_corr_1st, _ = get_neighbors_avg_correlation(get_correlation_df_with_only_alive_pillars(first_half_corrs), get_alive_pillars_to_alive_neighbors())
    # mean_corr_2nd, _ = get_neighbors_avg_correlation(get_correlation_df_with_only_alive_pillars(second_half_corrs), get_alive_pillars_to_alive_neighbors())
    # plot_nbrs_correlations_heatmap(first_half_corrs, get_alive_pillars_to_alive_neighbors())
    # plot_nbrs_correlations_heatmap(second_half_corrs, get_alive_pillars_to_alive_neighbors())
    # plot_nbrs_correlations_heatmap(overall_corrs, get_alive_pillars_to_alive_neighbors())
    # correlation_plot(alive_correlation_type='custom', pillars_corr_df=first_half_corrs, frame_to_show=len(get_images(get_images_path()))//2)
    # correlation_plot(alive_correlation_type='custom', pillars_corr_df=second_half_corrs)
    # correlation_plot(alive_correlation_type='custom', pillars_corr_df=overall_corrs)
    # return mean_corr_1st, mean_corr_2nd
    # return nbrs_avg_corr, non_nbrs_avg_corr
    # return get_cc_pp_cp_correlations()
    # get_cell_avg_intensity
    # return get_alive_pillars_symmetric_correlation()

    # print_tagged_centers()
    # with open(Consts.pillars_alive_location_by_frame_to_gif_cache_path, 'rb') as handle:
    #     pillars_alive_location_by_frame = pickle.load(handle)
    #     show_pillars_location_by_frame(pillars_alive_location_by_frame)
    #
    # return

    for op in operations:
        op_key = list(op.keys())[0]
        op_values = op[op_key]
        if op_key == 'neighbors_avg_correlation':
            nbrs_avg_correlation, _ = get_neighbors_avg_correlation(get_alive_pillars_symmetric_correlation(),
                                                                    get_alive_pillars_to_alive_neighbors())
        if op_key == 'non_neighbors_avg_correlation':
            non_nbrs_avg_correlation, _ = get_non_neighbors_mean_correlation(get_alive_pillars_symmetric_correlation(),
                                                                             get_alive_pillars_to_alive_neighbors())
        if op_key == 'correlation_plot':
            correlation_plot(
                op_values.get("only_alive", True),
                op_values.get("neighbors", 'all'),
                op_values.get("alive_correlation_type", 'symmetric')
            )
        elif op_key == "correlation_histogram":
            alive = op_values.get("alive", True)
            if alive:
                all_correlations_df = get_alive_pillars_symmetric_correlation()
                print("Correlation between all alive pillars:")
            else:
                all_correlations_df = get_all_pillars_correlations()
                print("Correlation between all pillars:")
            all_pillars_correlation = correlation_histogram(all_correlations_df)
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
                mean_corr, corrs_list = get_neighbors_avg_correlation(get_alive_pillars_symmetric_correlation(),
                                                                      neighbors_dict)
                neighbors_correlation_histogram(corrs_list)
            else:
                mean_corr, corrs_list = get_neighbors_avg_correlation(get_alive_pillars_symmetric_correlation(),
                                                                      neighbors_dict)
                neighbors_correlation_histogram(corrs_list)
        elif op_key == "non_neighbors_correlation_histogram":
            mean_corr, corrs_list = get_non_neighbors_mean_correlation(get_alive_pillars_symmetric_correlation(),
                                                                       get_alive_pillars_to_alive_neighbors())
            non_neighbors_correlation_histogram(corrs_list)
        elif op_key == "number_of_neighboring_pairs_in_top_correlations":
            top = op_values.get("top", 10)
            num_of_neighboring_pairs_in_top_corrs = get_number_of_neighboring_pillars_in_top_correlations(top=top)
        elif op_key == "compare_neighbors_corr_histogram_random_vs_real":
            random_amount = op_values.get("random_amount", 5)
            compare_neighbors_corr_histogram_random_vs_real(random_amount)
        elif op_key == "show_pillars_mask":
            show_last_image_masked(pillars_mask=build_pillars_mask(get_all_center_generated_ids()))
        elif op_key == "edges_distribution_plots":
            if gc_df is None:
                gc_df = get_gc_df()
            edges_distribution_plots(gc_df)
        elif op_key == "gc_graph":
            if gc_df is None:
                gc_df = get_gc_df()
            non_stat_lst, passed_stationary = get_non_stationary_pillars_lst()
            total_edges, _, _, inwards_edges, outwards_edges, inwards_percentage, outwards_percentage, out_in_factor = get_number_of_inwards_outwards_gc_edges(
                gc_df)
            build_gc_directed_graph(gc_df, non_stationary_pillars=non_stat_lst, inwards=inwards_edges,
                                    outwards=outwards_edges, random_neighbors=False)
        # elif op_key == "in_out_gc_edges":
        #     _, _, in_lst, out_lst = get_number_of_inwards_outwards_gc_edges(gc_df)
        #     build_gc_directed_graph(gc_df, edges_direction_lst=in_lst, draw=True)
        #     build_gc_directed_graph(gc_df, edges_direction_lst=out_lst, draw=True)
        elif op_key == "gc_edge_prob":
            if gc_df is None:
                gc_df = get_gc_df()
            gc_edge_prob = probability_for_gc_edge(gc_df, random_neighbors=False)
            gc_edge_prob = format(gc_edge_prob, ".3f")
        elif op_key == "gc_edge_prob_original_vs_random":
            if gc_df is None:
                gc_df = get_gc_df()
            gc_edge_probs_lst, avg_random_gc_edge_prob, std = avg_gc_edge_probability_original_vs_random(gc_df)
            gc_edge_probability_original_vs_random(gc_df, gc_edge_probs_lst)
        elif op_key == "in_out_degree":
            if gc_df is None:
                gc_df = get_gc_df()
            in_d, out_d, _ = get_pillar_in_out_degree(gc_df)
            in_out_degree_distribution(in_d, out_d)
        elif op_key == "reciprocity":
            if gc_df is None:
                gc_df = get_gc_df()
            reciprocity = get_network_reciprocity(gc_df)
        elif op_key == "heterogeneity":
            if gc_df is None:
                gc_df = get_gc_df()
            heterogeneity = get_network_heterogeneity(gc_df)
        elif op_key == "show_correlated_pairs_in_video":
            num_of_pairs = op_values.get("num_of_pairs", 5)
            show_correlated_pairs_in_video(n=num_of_pairs, neighbor_pairs=False)
        elif op_key == "show_correlated_neighboring_pairs_in_video":
            num_of_pairs = op_values.get("num_of_pairs", 5)
            show_correlated_pairs_in_video(n=num_of_pairs, neighbor_pairs=True)
        elif op_key == "plot_pillar_pairs_correlation_frames_window":
            num_of_pairs = op_values.get("num_of_pairs", 5)
            pillars_corrs_frame_wind = get_top_pairs_corr_in_each_frames_window(n=num_of_pairs, neighbor_pairs=False)
            plot_pillar_pairs_correlation_frames_window(pillars_corrs_frame_wind, neighbor_pairs=False)
        elif op_key == "plot_neighboring_pillar_pairs_correlation_frames_window":
            num_of_pairs = op_values.get("num_of_pairs", 5)
            pillars_corrs_frame_wind = get_top_pairs_corr_in_each_frames_window(n=num_of_pairs, neighbor_pairs=True)
            plot_pillar_pairs_correlation_frames_window(pillars_corrs_frame_wind, neighbor_pairs=True)
        elif op_key == "show_neighboring_correlated_pairs_in_last_image":
            num_of_pairs = op_values.get("num_of_pairs", 5)
            show_correlated_pairs_in_last_image(n=num_of_pairs, neighbor_pairs=True)
        elif op_key == "neighbors_avg_movement_correlation":
            pillars_movements_dict = get_alive_centers_movements_v2()
            movement_correlations_df = get_pillars_movement_correlation_df(pillars_movements_dict)
            neighbors_avg_movement_correlation = get_avg_movement_correlation(movement_correlations_df, neighbors=True)
        elif op_key == "not_neighbors_avg_movement_correlation":
            pillars_movements_dict = get_alive_centers_movements_v2()
            movement_correlations_df = get_pillars_movement_correlation_df(pillars_movements_dict)
            not_neighbors_avg_movement_correlation = get_avg_movement_correlation(movement_correlations_df,
                                                                                  neighbors=False)
        elif op_key == "avg_intensity":
            avg_intensity, _ = get_cell_avg_intensity()
        elif Consts.MULTI_COMPONENT and op_key == "change_mask_radius":
            print("change_mask_radius")
            temp_small_mask_radius = Consts.SMALL_MASK_RADIUS
            temp_large_mask_radius = Consts.LARGE_MASK_RADIUS

            mask_radiuses_tuples = [(0, 10), (0, 15), (10, 30), (20, 40), (15, 40), (10, 40)]
            map_radius_to_corr = {}
            # original_corr, _ = get_neighbors_avg_correlation(get_alive_pillars_symmetric_correlation(), get_alive_pillars_to_alive_neighbors())
            # print('original_corr', original_corr)

            for mask_radius in mask_radiuses_tuples:
                ratio_radiuses = get_mask_radiuses({'small_radius': mask_radius[0], 'large_radius': mask_radius[1]})
                Consts.SMALL_MASK_RADIUS = ratio_radiuses['small']
                Consts.LARGE_MASK_RADIUS = ratio_radiuses['large']

                # p_to_intens = get_pillar_to_intensities(get_images_path(), use_cache=False)

                # print("shows new mask", mask_radius)
                # show_last_image_masked(pillars_mask=get_mask_for_center(get_all_center_generated_ids()[172]), save_mask=False)
                # show_last_image_masked(pillars_mask=build_pillars_mask(get_all_center_generated_ids()),
                #                        save_mask=False)

                change_mask_radius_nbrs_avg_correlation, _ = get_neighbors_avg_correlation(
                    get_alive_pillars_symmetric_correlation(use_cache=False), get_alive_pillars_to_alive_neighbors())

                change_mask_radius_non_nbrs_avg_correlation, _ = get_non_neighbors_mean_correlation(
                    get_alive_pillars_symmetric_correlation(use_cache=False),
                    get_alive_pillars_to_alive_neighbors())

                map_radius_to_corr[mask_radius] = {}

                map_radius_to_corr[mask_radius]['nbrs_corrs'] = change_mask_radius_nbrs_avg_correlation
                map_radius_to_corr[mask_radius]['non_nbrs_corrs'] = change_mask_radius_non_nbrs_avg_correlation

                # print("change_mask_radius_nbrs_avg_correlation", mask_radius, change_mask_radius_nbrs_avg_correlation)
                # print("change_mask_radius_non_nbrs_avg_correlation", mask_radius, change_mask_radius_non_nbrs_avg_correlation)

            Consts.SMALL_MASK_RADIUS = temp_small_mask_radius
            Consts.LARGE_MASK_RADIUS = temp_large_mask_radius
        # elif op_key == "correlations_norm_by_noise_(0,10)":
        #     mask_radius = (0, 10)
        #     radiuses = [(15, 35), (0, 10), (10, 30), (20, 40), (15, 40), (10, 40)]
        #
        #     map_radius_to_norm_corr_by_0_10 = get_correlations_norm_by_noise(mask_radius, radiuses)
        # elif op_key == "correlations_norm_by_noise_(0,15)":
        #     mask_radius = (0, 15)
        #     radiuses = [(15, 35), (0, 15), (10, 30), (20, 40), (15, 40), (10, 40)]
        #
        #     map_radius_to_norm_corr_by_0_15 = get_correlations_norm_by_noise(mask_radius, radiuses)

    if Consts.MULTI_COMPONENT and Consts.WRITE_OUTPUT:
        features_dict = {'passed_stationary': passed_stationary,
                         'nbrs_avg_correlation': nbrs_avg_correlation,
                         'non_nbrs_avg_correlation': non_nbrs_avg_correlation,
                         'total_gc_edges': total_edges,
                         'inwards_edges': inwards_percentage,
                         'outwards_edges': outwards_percentage,
                         'out/in_factor': out_in_factor,
                         'gc_edge_prob': gc_edge_prob,
                         'avg_random_gc_edge_prob': avg_random_gc_edge_prob,
                         'std': std,
                         'reciprocity': reciprocity,
                         'heterogeneity': heterogeneity,
                         'num_of_neighboring_pairs_in_top_corrs': num_of_neighboring_pairs_in_top_corrs,
                         'neighbors_avg_movement_correlation': neighbors_avg_movement_correlation,
                         'not_neighbors_avg_movement_correlation': not_neighbors_avg_movement_correlation,
                         'avg_intensity': avg_intensity,
                         'change_mask_radius': str(map_radius_to_corr),
                         'correlations_norm_by_noise_(0,10)': str(map_radius_to_norm_corr_by_0_10),
                         'correlations_norm_by_noise_(0,15)': str(map_radius_to_norm_corr_by_0_15)
                         }
        experiment_id = config_data.get("experiment")['id']
        perturbation = config_data.get("perturbation")['type']
        output_path_type = config_data.get("output_path_type")
        index = perturbation + "_" + experiment_id
        if Consts.RESULT_FOLDER_PATH is not None:
            output_path = Consts.RESULT_FOLDER_PATH + "/results.csv"
        else:
            output_path = get_output_path(output_path_type)
        if os.path.exists(output_path):
            output_df = pd.read_csv(output_path, index_col=0)
            features_df = pd.DataFrame(features_dict, index=[index])
            output_df = output_df.append(features_df)
            output_df.to_csv(output_path)
        else:
            output_df = pd.DataFrame(features_dict, index=[index])
            output_df.to_csv(output_path)
        print("Writen output")


if __name__ == '__main__':
    perturbation_type = "13.2"
    exp_number = "1-149_bottom_right"
    config_name = perturbation_type + "/exp_" + exp_number + "_type_" + perturbation_type + "_mask_15_35_non-normalized_fixed.json"
    run_config(config_name)

    # lst_real_REF_5_3 = np.array([0.178, 0.100, 0.351, -0.023, 0.406, 0.468, 0.017])
    # lst_random_REF_5_3 = np.array([0.178, 0.059, 0.121, 0.004, 0.377, 0.358, -0.004])
    # labelsref53 = ['37.41.1', '37.41.2', '37.41.3', '37.41.4', '69.1', '69.2', '69.3']
    #
    # lst_real_MEF_9_4 = np.array([0.057, 0.116, 0.125, 0.125, 0.088, 0.026, 0.036])
    # lst_random_MEF_9_4 = np.array([-0.031, 0.119, 0.030, 0.065, 0.024, -0.018, 0.011])
    # labels94 = ['04.1', '04.2', '06.1', '06.2', '11.1', '11.2', '12']
    #
    # lst_real_MEF_KD_9_4 = np.array([0.257, 0.172, -0.105, 0.264, 0.360])
    # lst_random_MEF_KD_9_4 = np.array([0.080, 0.034, -0.039, 0.045, 0.403])
    # labelskd94 = ['02', '08', '11', '17', '53']
    #
    # lst_real_MEF_13_2 = np.array([0.196, 0.097, 0.030, 0.093])
    # lst_random_MEF_13_2 = np.array([0.132, 0.000, 0.038, 0.107])
    # labels132 = ['01', '05', '06', '20']
    #
    # lst_real_MEF_KD_13_2 = np.array([0.254, 0.121, 0.262, 0.398, 0.217])
    # lst_random_MEF_KD_13_2 = np.array([0.021, 0.019, 0.042, 0.029, 0.076])
    # labelskd132 = ['01', '02', '17', '46', '49.1']
    #
    # lst_real_MEF_5_3 = np.array([0.524, 0.144, 0.443, 0.421, 0.343, 0.269])
    # lst_random_MEF_5_3 = np.array([0.043, 0.062, 0.089, 0.313, 0.182, -0.027])
    # labels53 = ['08', '09', '12', '27.1', '27.2', '30']
    # #
    # average_correlation_real_vs_random_plot(lst_real_MEF_9_4, lst_random_MEF_9_4,labels94, title='MEF9.4')
    # average_correlation_real_vs_random_plot(lst_real_MEF_KD_9_4, lst_random_MEF_KD_9_4, labelskd94, title='KD_MEF9.4')
    # average_correlation_real_vs_random_plot(lst_real_MEF_13_2, lst_random_MEF_13_2, labels132, title='MEF13.2')
    # average_correlation_real_vs_random_plot(lst_real_MEF_KD_13_2, lst_random_MEF_KD_13_2, labelskd132, title='KD_MEF13.2')
    # average_correlation_real_vs_random_plot(lst_real_MEF_5_3, lst_random_MEF_5_3, labels53, title='MEF5.3')
    # average_correlation_real_vs_random_plot(lst_real_REF_5_3, lst_random_REF_5_3, labelsref53, title='REF5.3')

    # names = ['REF5.3', 'MEF9.4', 'KD_MEF9.4', 'MEF13.2', 'KD_MEF13.2', 'MEF5.3']
    # avg = [0.05471428571, 0.05528571429, 0.10442, 0.125, 0.24902, 0.2383666667]
    # plt.bar(names, avg)
    # plt.title('Average of (neighbors_pair - random_pair)')
    # plt.ylabel('Average of Differences')
    # plt.show()
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

# def get_dead_pillars_movements():
#     all_center_ids = get_all_center_ids()
#     alive_pillars = get_alive_pillars_overall_v2()
#     dead_pillars = [p for p in all_center_ids if p not in alive_pillars]
#
#     frame_images = get_images(get_images_path())
#     pillar_id2real_locs = {}
#
#     # TODO: think if should save prev loc before calling get_center_fixed_by_circle_mask_reposition again (not with original ID everytime)
#
#     for dead_pillar in dead_pillars:
#         pillar_id2real_locs[dead_pillar] = []
#         for frame in frame_images:
#             curr_pillar_actual_loc = get_center_fixed_by_circle_mask_reposition(dead_pillar, frame)
#             pillar_id2real_locs[dead_pillar].append(curr_pillar_actual_loc)
#
#     pillar2distances = {}
#
#     for p, locs in pillar_id2real_locs.items():
#         pillar2distances[p] = []
#         for i in range(locs - 1):
#             curr_loc = locs[i]
#             next_loc = locs[i+1]
#
#             distance = math.hypot(curr_loc[1] - next_loc[1], curr_loc[0] - next_loc[0])
#             pillar2distances[p].append(distance)
