from scipy.stats import pearsonr, kendalltau
from scipy import stats
from scipy.stats import ttest_ind
from Pillars.pillar_intensities import *
from Pillars.pillars_utils import *
from Pillars.pillar_neighbors import *
from Pillars.consts import *
# from Pillars.granger_causality_test import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx


def get_correlations_between_neighboring_pillars(pillar_to_pillars_dict):
    """
    Listing all correlations between pillar and its neighbors
    :param pillar_to_pillars_dict:
    :return:
    """
    all_corr = get_all_pillars_correlations()

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

    if Consts.USE_CACHE and os.path.isfile(path):
        with open(path, 'rb') as handle:
            correlation = pickle.load(handle)
            return correlation

    relevant_pillars_dict = get_alive_pillars_to_intensities()

    pillar_intensity_df = pd.DataFrame({str(k): v for k, v in relevant_pillars_dict.items()})
    alive_pillars_corr = pillar_intensity_df.corr()
    if Consts.USE_CACHE:
        with open(path, 'wb') as handle:
            pickle.dump(alive_pillars_corr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return alive_pillars_corr


def get_alive_pillars_correlations_with_running_frame_windows():
    if Consts.USE_CACHE and os.path.isfile(Consts.alive_pillars_correlations_with_running_frame_windows_cache_path):
        with open(Consts.alive_pillars_correlations_with_running_frame_windows_cache_path, 'rb') as handle:
            alive_pillars_correlations_with_running_frame_windows = pickle.load(handle)
            return alive_pillars_correlations_with_running_frame_windows

    pillar_intensity_dict = get_alive_pillars_to_intensities()

    all_pillar_intensity_df = pd.DataFrame({str(k): v for k, v in pillar_intensity_dict.items()})

    all_pillar_intensity_df_array = np.array_split(all_pillar_intensity_df, Consts.FRAME_WINDOWS_AMOUNT)

    alive_pillars_correlations_with_running_frame_windows = [df.corr(method=Consts.CORRELATION) for df in
                                                             all_pillar_intensity_df_array]

    if Consts.USE_CACHE:
        with open(Consts.alive_pillars_correlations_with_running_frame_windows_cache_path, 'wb') as handle:
            pickle.dump(alive_pillars_correlations_with_running_frame_windows, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return alive_pillars_correlations_with_running_frame_windows


def get_alive_pillars_correlations_frame_windows(frame_window=Consts.FRAME_WINDOWS_AMOUNT):
    if Consts.USE_CACHE and os.path.isfile(Consts.alive_pillars_correlations_frame_windows_cache_path):
        with open(Consts.alive_pillars_correlations_frame_windows_cache_path, 'rb') as handle:
            pillars_corrs_frame_window = pickle.load(handle)
            return pillars_corrs_frame_window

    pillar_intensity_dict = get_alive_pillars_to_intensities()

    all_pillar_intensity_df = pd.DataFrame({str(k): v for k, v in pillar_intensity_dict.items()})

    all_pillar_intensity_df_array = np.array_split(all_pillar_intensity_df, frame_window)

    pillars_corrs_frame_window = []
    new_df = None
    for i, df in enumerate(all_pillar_intensity_df_array):
        if i == 0:
            new_df = df
        else:
            new_df = new_df.append(df)

        pillars_corrs_frame_window.append(new_df.corr(method=Consts.CORRELATION))

    if Consts.USE_CACHE:
        with open(Consts.alive_pillars_correlations_frame_windows_cache_path, 'wb') as handle:
            pickle.dump(pillars_corrs_frame_window, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pillars_corrs_frame_window


# Todo: here we have exception
def get_top_pairs_corr_in_each_frames_window(n=5, neighbor_pairs=True):
    correlations = get_alive_pillars_symmetric_correlation()
    neighbors_dict = get_alive_pillars_to_alive_neighbors()
    sorted_correlations = correlations.where(
        np.triu(np.ones(correlations.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
    corr_frames_wind = get_alive_pillars_correlations_frame_windows()
    i = 0
    top_pairs_corr_in_each_frames_window = {}
    for pillars, value in sorted_correlations.items():
        if i == n:
            break
        if neighbor_pairs:
            if eval(pillars[0]) not in neighbors_dict[eval(pillars[1])]:
                continue
        pair_corr_in_each_frames_window = []
        for df in corr_frames_wind:
            corr = df.loc[pillars[0], pillars[1]]
            pair_corr_in_each_frames_window.append(corr)
        top_pairs_corr_in_each_frames_window[pillars] = pair_corr_in_each_frames_window
        i += 1

    return top_pairs_corr_in_each_frames_window


def get_number_of_neighboring_pillars_in_top_correlations(top=10):
    correlations = get_alive_pillars_symmetric_correlation()
    neighbors_dict = get_alive_pillars_to_alive_neighbors()
    sorted_correlations = correlations.where(
        np.triu(np.ones(correlations.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
    num_of_neighboring_pairs_in_top_corrs = 0
    for i in range(top):
        pillars = sorted_correlations.index[i]
        if eval(pillars[0]) in neighbors_dict[eval(pillars[1])]:
            num_of_neighboring_pairs_in_top_corrs += 1

    print('number of neighboring pillars pairs in the top ' + str(top) + ' correlations: ' + str(
        num_of_neighboring_pairs_in_top_corrs))
    return num_of_neighboring_pairs_in_top_corrs


def get_alive_pillars_corr_path():
    """
    Get the cached correlation of alive pillars
    :return:
    """
    if Consts.normalized:
        path = Consts.correlation_alive_normalized_cache_path
    else:
        path = Consts.correlation_alive_not_normalized_cache_path

    return path


def get_all_pillars_correlations():
    """
    Create dataframe of correlation between all pillars
    :return:
    """
    path = get_all_pillars_corr_path()

    if Consts.USE_CACHE and os.path.isfile(path):
        with open(path, 'rb') as handle:
            correlation = pickle.load(handle)
            return correlation

    if Consts.normalized:
        pillar_intensity_dict = normalized_intensities_by_mean_background_intensity()
    else:
        pillar_intensity_dict = get_pillar_to_intensities(get_images_path())

    all_pillar_intensity_df = pd.DataFrame({str(k): v for k, v in pillar_intensity_dict.items()})
    if Consts.CORRELATION == "pearson":
        all_pillars_corr = all_pillar_intensity_df.corr()
    if Consts.CORRELATION == "kendall":
        all_pillars_corr = all_pillar_intensity_df.corr(method='kendall')

    if Consts.USE_CACHE:
        with open(path, 'wb') as handle:
            pickle.dump(all_pillars_corr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_pillars_corr


def get_all_pillars_correlations_frame_windows():
    # TODO: cache

    if Consts.normalized:
        pillar_intensity_dict = normalized_intensities_by_mean_background_intensity()
    else:
        pillar_intensity_dict = get_pillar_to_intensities(get_images_path())

    all_pillar_intensity_df = pd.DataFrame({str(k): v for k, v in pillar_intensity_dict.items()})

    all_pillar_intensity_df_array = np.array_split(all_pillar_intensity_df, Consts.FRAME_WINDOWS_AMOUNT)

    return [df.corr(method=Consts.CORRELATION) for df in all_pillar_intensity_df_array]


def get_all_pillars_corr_path():
    """
    Get the cached correlation of all pillars
    :return:
    """
    if Consts.normalized:
        path = Consts.all_pillars_correlation_normalized_cache_path
    else:
        path = Consts.all_pillars_correlation_not_normalized_cache_path

    return path


def get_alive_pillars_symmetric_correlation(frame_start=None, frame_end=None):
    """
    Create dataframe of alive pillars correlation as the correlation according to
    maximum_frame(pillar a start to live, pillar b start to live) -> correlation(a, b) == correlation(b, a)
    :return:
    """
    origin_frame_start = frame_start
    origin_frame_end = frame_end
    if origin_frame_start is None and origin_frame_end is None and Consts.USE_CACHE and os.path.isfile(
            Consts.alive_pillars_sym_corr_cache_path):
        with open(Consts.alive_pillars_sym_corr_cache_path, 'rb') as handle:
            alive_pillars_symmetric_correlation = pickle.load(handle)
            return alive_pillars_symmetric_correlation

    frame_to_alive_pillars = get_alive_center_ids_by_frame_v2()  # get_frame_to_alive_pillars_by_same_mask(pillar2mask)

    if frame_start is None:
        frame_start = 0
    if frame_end is None:
        frame_end = len(frame_to_alive_pillars)

    alive_pillars_to_frame = {}
    for curr_frame, alive_pillars_in_frame in frame_to_alive_pillars.items():
        if frame_start <= curr_frame <= frame_end:
            for alive_pillar in alive_pillars_in_frame:
                if alive_pillar not in alive_pillars_to_frame:
                    alive_pillars_to_frame[alive_pillar] = curr_frame

    alive_pillars_intens = get_alive_pillars_to_intensities()
    alive_pillars = list(alive_pillars_intens.keys())
    alive_pillars_str = [str(p) for p in alive_pillars]
    pillars_corr = pd.DataFrame(0.0, index=alive_pillars_str, columns=alive_pillars_str)

    # Symmetric correlation - calc correlation of 2 pillars start from the frame they are both alive: maxFrame(A, B)
    for p1 in alive_pillars_to_frame:
        p1_living_frame = alive_pillars_to_frame[p1]
        for p2 in alive_pillars_to_frame:
            p2_living_frame = alive_pillars_to_frame[p2]
            both_alive_frame = max(p1_living_frame, p2_living_frame)
            p1_relevant_intens = alive_pillars_intens[p1][both_alive_frame:frame_end]
            p2_relevant_intens = alive_pillars_intens[p2][both_alive_frame:frame_end]
            # b/c of this, even if pillar is alive for only 2 frames, we will calculate the correlation,
            # if we will increase 1 to X it means it needs to live for at least X frames to calc correlation for
            if len(p1_relevant_intens) > 1 and len(p2_relevant_intens) > 1:
                if Consts.CORRELATION == "pearson":
                    pillars_corr.loc[str(p2), str(p1)] = pearsonr(p1_relevant_intens, p2_relevant_intens)[0]
                if Consts.CORRELATION == "kendall":
                    pillars_corr.loc[str(p2), str(p1)] = kendalltau(p1_relevant_intens, p2_relevant_intens)[0]

    if origin_frame_start is None and origin_frame_end is None and Consts.USE_CACHE:
        with open(Consts.alive_pillars_sym_corr_cache_path, 'wb') as handle:
            pickle.dump(pillars_corr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pillars_corr


def get_indirect_neighbors_correlation(pillar_location, only_alive=True):
    """
    Create dataframe of correlation between pillar and its indirect neighbors (start from neighbors level 2)
    :param pillar_location:
    :param only_alive:
    :return:
    """
    if only_alive:
        pillars_corr = get_alive_pillars_correlation()
    else:
        pillars_corr = get_all_pillars_correlations()

    pillar_directed_neighbors = get_pillar_directed_neighbors(pillar_location)

    pillar_directed_neighbors_str = []
    for tup in pillar_directed_neighbors:
        if tup != pillar_location:
            pillar_directed_neighbors_str.append(str(tup))
    pillars_corr = pillars_corr.drop(pillar_directed_neighbors_str, axis=0)
    pillars_corr = pillars_corr.drop(pillar_directed_neighbors_str, axis=1)

    return pillars_corr


def get_neighbors_avg_correlation(correlations_df, neighbors_dict):
    sym_corr = set()
    for pillar, nbrs in neighbors_dict.items():
        for nbr in nbrs:
            if str(nbr) in correlations_df and str(pillar) in correlations_df:
                sym_corr.add(correlations_df[str(pillar)][str(nbr)])
    corrs = np.array(list(sym_corr))
    mean_corr = np.mean(corrs)
    mean_corr = format(mean_corr, ".3f")
    return mean_corr, corrs


def get_non_neighbors_mean_correlation(correlations_df, neighbors_dict):
    correlation_list = []
    df = correlations_df.mask(np.tril(np.ones(correlations_df.shape)).astype(np.bool))
    for pillar1 in df.columns:
        for pillar2 in df.columns:
            if pillar1 != pillar2 and eval(pillar2) not in neighbors_dict[eval(pillar1)]:
                correlation_list.append(df[str(pillar1)][str(pillar2)])

    mean_corr = np.nanmean(correlation_list)
    mean_corr = format(mean_corr, ".3f")
    return mean_corr, correlation_list


def get_number_of_inwards_outwards_gc_edges(gc_df):
    """
    Count the number of inward and outward gc edges
    :param gc_df:
    :return:
    """
    all_alive_centers = get_seen_centers_for_mask()
    cell_center = get_center_of_points(all_alive_centers)
    neighbors = get_alive_pillars_to_alive_neighbors()
    inwards = 0
    outwards = 0
    total_edges = 0
    in_edges = []
    out_edges = []

    for col in gc_df.keys():
        int_col = eval(col)
        for row, _ in gc_df.iterrows():
            int_row = eval(row)
            if gc_df[col][row] < Consts.gc_pvalue_threshold and int_row in neighbors[int_col]:
                total_edges += 1
                # ang = math.degrees(math.atan2(center[1] - int_col[1], center[0] - int_col[0]) - math.atan2(int_row[1] - int_col[1], int_row[0] - int_col[0]))
                # ang = ang + 360 if ang < 0 else ang
                ang = get_angle(int_col, int_row, cell_center)
                if math.dist(int_col, cell_center) < math.dist(int_row, cell_center) and ang >= 135:
                    outwards += 1
                    out_edges.append((col, row))
                elif math.dist(int_col, cell_center) > math.dist(int_row, cell_center) and ang <= 45:
                    inwards += 1
                    in_edges.append((col, row))
    if total_edges == 0:
        in_percentage, out_percentage = 0, 0
    else:
        in_percentage = inwards / total_edges
        out_percentage = outwards / total_edges
    print("Number of total edges: " + str(total_edges))
    print("Number of inwards gc edges: " + str(inwards) + " (" + format(in_percentage * 100, ".2f") + "%)")
    print("Number of outwards gc edges: " + str(outwards) + " (" + format(out_percentage * 100, ".2f") + "%)")
    if in_percentage > 0:
        out_in_factor = format(out_percentage / in_percentage, ".3f")
        print("out/in factor: " + str(out_in_factor))
    else:
        out_in_factor = 'inf'
        print("out/in factor: in edges = 0")
    in_percentage = format(in_percentage, ".3f")
    out_percentage = format(out_percentage, ".3f")
    return total_edges, inwards, outwards, in_edges, out_edges, in_percentage, out_percentage, out_in_factor


def get_angle(col, row, center):
    a = np.array([row[0], row[1]])
    b = np.array([col[0], col[1]])
    c = np.array([center[0], center[1]])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    return angle


def probability_for_gc_edge(gc_df, random_neighbors=False):
    """

    :param gc_df:
    :param only_alive:
    :return:
    """
    if random_neighbors:
        neighbors = get_random_neighbors()
    else:
        neighbors = get_alive_pillars_to_alive_neighbors()

    num_of_potential_gc_edges = 0
    num_of_actual_gc_edges = 0

    for col in gc_df.keys():
        for row, _ in gc_df.iterrows():
            if eval(row) in neighbors[eval(col)]:
                num_of_potential_gc_edges += 1
                if gc_df[col][row] < Consts.gc_pvalue_threshold:
                    num_of_actual_gc_edges += 1

    prob_for_gc_edge = num_of_actual_gc_edges / num_of_potential_gc_edges

    print("The probability for a gc edge is " + str(prob_for_gc_edge))

    return prob_for_gc_edge


def avg_gc_edge_probability_original_vs_random(gc_df):
    gc_edge_probs_lst = []
    idx = []
    for i in range(10):
        prob = probability_for_gc_edge(gc_df, random_neighbors=True)
        gc_edge_probs_lst.append(prob)
        idx.append(i)
    avg_random_gc_edge_prob = format(np.mean(gc_edge_probs_lst), ".3f")
    std = format(np.std(gc_edge_probs_lst), ".3f")
    print("avg gc edge probability for random " + str(avg_random_gc_edge_prob))
    print("std: " + str(std))
    return gc_edge_probs_lst, avg_random_gc_edge_prob, std


def get_pillar_in_out_degree(gc_df):
    """
    Get from the granger causality network, the in and out degree edges of each pillar
    :param gc_df: granger causality dataframe
    :return: list of in degree edges of all pillars, out degree of all pillars, list of tuples of (pillar, (in degree, out degree))
    """
    gc_edges_only_df = get_gc_edges_df(gc_df)

    pillars_degree_dict = {}
    in_degree_lst = []
    out_degree_lst = []

    for i, pillar in enumerate(gc_df.columns):
        out_degree = gc_edges_only_df[pillar].count()
        in_degree = gc_edges_only_df.iloc[[i]].count().sum()

        out_degree_lst.append(out_degree)
        in_degree_lst.append(in_degree)
        pillars_degree_dict[pillar] = (in_degree, out_degree)

    return in_degree_lst, out_degree_lst, pillars_degree_dict


def get_gc_edges_df(gc_df):
    neighbors = get_alive_pillars_to_alive_neighbors()

    gc_edges_only_df = gc_df.copy()

    for col in gc_df.keys():
        for row, _ in gc_df.iterrows():
            if not (gc_df[col][row] < Consts.gc_pvalue_threshold and eval(row) in neighbors[eval(col)]):
                gc_edges_only_df[col][row] = None

    return gc_edges_only_df


def get_total_gc_edges(gc_df):
    df = get_gc_edges_df(gc_df)

    return df.count().sum()


def get_network_reciprocity(gc_df):
    neighbors = get_alive_pillars_to_alive_neighbors()
    two_sided_edge = 0

    rows = list(gc_df.keys())
    for i, col in enumerate(gc_df.keys()):
        for j in range(i + 1, len(rows)):
            row = rows[j]
            if eval(row) in neighbors[eval(col)] and gc_df[col][row] < Consts.gc_pvalue_threshold and \
                    gc_df[row][col] < Consts.gc_pvalue_threshold:
                two_sided_edge += 2

    total_edges = get_total_gc_edges(gc_df)
    if total_edges == 0:
        print("There are no edges for reciprocity")
        return
    else:
        reciprocity = format(two_sided_edge / total_edges, ".3f")

    print("Reciprocity: " + str(reciprocity))
    return reciprocity


def get_number_of_pillars_with_edges(gc_df):
    df = get_gc_edges_df(gc_df)

    pillars_without_edges = []
    for col in df.columns:
        if df[col].count() == 0:
            if df.loc[col].count() == 0:
                pillars_without_edges.append(col)

    num_pillars_with_edges = len(df) - len(pillars_without_edges)
    return num_pillars_with_edges


def get_network_heterogeneity(gc_df):
    _, _, pillars_degree_rank = get_pillar_in_out_degree(gc_df)
    df = get_gc_edges_df(gc_df)
    sum_hetero = 0

    for col in df.keys():
        for row, _ in gc_df.iterrows():
            if not math.isnan(df[col][row]):
                d_i_out = pillars_degree_rank[col][1]
                d_j_in = pillars_degree_rank[row][0]
                edge_hetero = (1 / d_i_out) + (1 / d_j_in) - (2 / np.sqrt(d_i_out * d_j_in))
                sum_hetero += edge_hetero

    v = get_number_of_pillars_with_edges(gc_df)
    if v == 0:
        print("There are no edges for heterogeneity")
        return
    else:
        g_heterogeneity = (1 / (v - 2 * np.sqrt(v - 1))) * sum_hetero
        g_heterogeneity = format(g_heterogeneity, ".3f")

    print("Heterogeneity: " + str(g_heterogeneity))
    return g_heterogeneity


def get_output_df(output_path_type):
    output_path = get_output_path(output_path_type)

    output_df = pd.read_csv(output_path, index_col=0)
    output_df.drop('passed_stationary', axis=1, inplace=True)

    return output_df


def get_output_path(output_path_type):
    if Consts.inner_cell:
        output_path = './features output/output_inner_cell_' + output_path_type + '.csv'
    else:
        output_path = './features output/output' + output_path_type + '.csv'

    return output_path


def get_pca(output_path_type, n_components, custom_df=None):
    if custom_df is None:
        output_df = get_output_df(output_path_type)
    else:
        output_df = custom_df
    x = StandardScaler().fit_transform(output_df)
    pca = PCA(n_components=n_components)
    principle_comp = pca.fit_transform(x)
    return pca, principle_comp


def t_test(samp_lst_1, samp_lst_2):
    stat, pval = ttest_ind(samp_lst_1, samp_lst_2)
    print("p-value: " + str(pval))
    return stat, pval


def get_pillars_movement_correlation_df(pillars_movements_dict):
    frame_to_df_pillars_movement_corr = get_list_of_frame_df_pillars_movement_correlation(pillars_movements_dict)
    frame_to_alive_pillars = get_alive_center_ids_by_frame_v2()
    alive_pillars_to_frame = {}
    for frame, alive_pillars_in_frame in frame_to_alive_pillars.items():
        for alive_pillar in alive_pillars_in_frame:
            if alive_pillar not in alive_pillars_to_frame:
                alive_pillars_to_frame[alive_pillar] = frame

    alive_pillars = list(pillars_movements_dict.keys())
    alive_pillars_str = [str(p) for p in alive_pillars]
    pillars_movements_corr_df = pd.DataFrame(0.0, index=alive_pillars_str, columns=alive_pillars_str)

    for p1 in alive_pillars_to_frame:
        p1_living_frame = alive_pillars_to_frame[p1]
        for p2 in alive_pillars_to_frame:
            p2_living_frame = alive_pillars_to_frame[p2]
            both_alive_frame = max(p1_living_frame, p2_living_frame)
            pillars_corrs_list = []
            for i in range(both_alive_frame, len(frame_to_df_pillars_movement_corr)):
                df = frame_to_df_pillars_movement_corr[i]
                corr = df[str(p1)][str(p2)]
                pillars_corrs_list.append(corr)

            avg_corr = np.nanmean(pillars_corrs_list)
            pillars_movements_corr_df.loc[str(p1), str(p2)] = avg_corr

    return pillars_movements_corr_df


def total_movements_percentage(pillars_movements_dict):
    frame_to_alive_pillars = get_alive_center_ids_by_frame_v2()
    alive_pillars_to_frame = {}
    for frame, alive_pillars_in_frame in frame_to_alive_pillars.items():
        for alive_pillar in alive_pillars_in_frame:
            if alive_pillar not in alive_pillars_to_frame:
                alive_pillars_to_frame[alive_pillar] = frame

    total_frames = len(frame_to_alive_pillars)
    total_chances_to_move = 0
    for frame in alive_pillars_to_frame.values():
        pillar_chances_to_move = total_frames - frame - 1
        total_chances_to_move += pillar_chances_to_move

    actual_movements = 0
    for moves in pillars_movements_dict.values():
        for move in moves:
            if move['distance'] != 0:
                actual_movements += 1

    total_movements_percentage = actual_movements / total_chances_to_move
    with open(Consts.RESULT_FOLDER_PATH + "/total_movements_percentage.txt", 'w') as f:
        f.write("total possible movements percentage: " + str(total_movements_percentage))

    return total_movements_percentage


def get_pillars_intensity_movement_correlations():
    """
    To each pillar - calculating the correlation between its intensities vector to its movements vector
    :return: dictionary of the correlation between those vectors to each pillar
    """
    pillars_movements_dict = get_alive_centers_movements()
    p_to_distances_dict = {}
    for p, v_list in pillars_movements_dict.items():
        distances = []
        for move_dict in v_list:
            distances.append(move_dict['distance'])
        p_to_distances_dict[p] = distances
    pillars_dist_movements_df = pd.DataFrame({str(k): v for k, v in p_to_distances_dict.items()})

    pillar_to_intens_dict = get_alive_pillars_to_intensities()
    pillar_intens_df = pd.DataFrame({str(k): v for k, v in pillar_to_intens_dict.items()})
    pillar_intens_df = pillar_intens_df[:-1]

    frame_to_alive_pillars = get_alive_center_ids_by_frame_v2()  # get_frame_to_alive_pillars_by_same_mask(pillar2mask)
    alive_pillars_to_frame = {}
    for frame, alive_pillars_in_frame in frame_to_alive_pillars.items():
        for alive_pillar in alive_pillars_in_frame:
            if alive_pillar not in alive_pillars_to_frame:
                alive_pillars_to_frame[alive_pillar] = frame

    pillars = pillar_intens_df.columns
    all_pillars_intens_dist_corrs = {}
    for p in pillars:
        p_corr = None
        alive_frame = alive_pillars_to_frame[eval(p)]
        pillar_intens_series = pillar_intens_df[p]
        pillars_dist_series = pillars_dist_movements_df[p]
        p_relevant_intens = pillar_intens_series[alive_frame:]
        p_relevant_dist = pillars_dist_series[alive_frame:]
        if len(p_relevant_intens) > 1 and len(p_relevant_dist) > 1:
            p_corr = pearsonr(p_relevant_intens, p_relevant_dist)[0]
        all_pillars_intens_dist_corrs[p] = p_corr

    return all_pillars_intens_dist_corrs


def get_avg_correlation_pillars_intensity_movement():
    all_pillars_intens_dist_corrs = get_pillars_intensity_movement_correlations()
    avg_corr = np.nanmean(list(all_pillars_intens_dist_corrs.values()))

    with open(Consts.RESULT_FOLDER_PATH + "/intens_movement_sync.txt", 'w') as f:
        f.write("The correlation between the actin signal and the pillar movement is: " + str(avg_corr))

    return avg_corr


def get_average_intensity_by_distance():
    pillars_movements_dict = get_alive_centers_movements()
    p_to_distances_dict = {}
    for p, v_list in pillars_movements_dict.items():
        distances = []
        for move_dict in v_list:
            distances.append(move_dict['distance'])
        p_to_distances_dict[p] = distances
    pillars_dist_movements_df = pd.DataFrame({str(k): v for k, v in p_to_distances_dict.items()})

    pillar_to_intens_dict = get_alive_pillars_to_intensities()
    pillar_intens_df = pd.DataFrame({str(k): v for k, v in pillar_to_intens_dict.items()})
    pillar_intens_df = pillar_intens_df[:-1]

    frame_to_alive_pillars = get_alive_center_ids_by_frame_v2()  # get_frame_to_alive_pillars_by_same_mask(pillar2mask)
    alive_pillars_to_frame = {}
    for frame, alive_pillars_in_frame in frame_to_alive_pillars.items():
        for alive_pillar in alive_pillars_in_frame:
            if alive_pillar not in alive_pillars_to_frame:
                alive_pillars_to_frame[alive_pillar] = frame

    # pillars = pillar_intens_df.columns
    # pillar_to_avg_intens_by_dist = {}
    # for p in pillars:
    #     alive_frame = alive_pillars_to_frame[eval(p)]
    #     pillar_intens_series = pillar_intens_df[p]
    #     pillars_dist_series = pillars_dist_movements_df[p]
    #     p_relevant_intens = pillar_intens_series[alive_frame:]
    #     p_relevant_intens = p_relevant_intens.reset_index().drop('index', axis=1).squeeze()
    #     p_relevant_dist = pillars_dist_series[alive_frame:]
    #     p_relevant_dist = p_relevant_dist.reset_index().drop('index', axis=1).squeeze()
    #     p_unique_distances = p_relevant_dist.unique()
    #     pillar_to_avg_intens_by_dist[p] = {}
    #     for dist in p_unique_distances:
    #         distance_indexes = list(p_relevant_dist[p_relevant_dist == dist].index)
    #         avg_intens_by_distance = np.mean([p_relevant_intens[i] for i in distance_indexes])
    #         pillar_to_avg_intens_by_dist[p][dist] = avg_intens_by_distance

    pillar_to_avg_intens_by_dist = {}
    pillars = pillar_intens_df.columns
    for p in pillars:

        intens_when_dist_zero = []
        intens_when_dist_zero_not_zero = []

        p_relevant_intens = pillar_intens_df[p]
        p_relevant_dist = pillars_dist_movements_df[p]
        for i, intens in enumerate(p_relevant_intens):
            if p_relevant_dist[i] == 0:
                intens_when_dist_zero.append(intens)
            else:
                intens_when_dist_zero_not_zero.append(intens)
        pillar_to_avg_intens_by_dist[p] = {"avg_intens_when_dist_zero": np.mean(intens_when_dist_zero),
                                           "avg_intens_when_dist_non_zero": np.mean(
                                               intens_when_dist_zero_not_zero)}

    return pillar_to_avg_intens_by_dist


def get_avg_movement_correlation(movement_correlations_df, neighbors=False):
    sym_corr = set()
    neighbors_dict = get_alive_pillars_to_alive_neighbors()
    for p1 in movement_correlations_df.columns:
        for p2 in movement_correlations_df.columns:
            if p1 != p2:
                if neighbors:
                    if eval(p2) in neighbors_dict[eval(p1)]:
                        sym_corr.add(movement_correlations_df[str(p1)][str(p2)])
                else:
                    if eval(p2) not in neighbors_dict[eval(p1)]:
                        sym_corr.add(movement_correlations_df[str(p1)][str(p2)])

    corr = np.array(list(sym_corr))
    mean_corr = np.nanmean(corr)
    return mean_corr


def correlation_graph():
    corr_df = get_alive_pillars_symmetric_correlation()
    pillar_to_neighbor_dict = get_alive_pillars_to_alive_neighbors()
    weighted_graph = {}
    for p, nbrs in pillar_to_neighbor_dict.items():
        nbrs_weight = {}
        for nbr in nbrs:
            nbrs_weight[nbr] = corr_df[p][nbr]
        weighted_graph[p] = nbrs_weight

    return weighted_graph


def get_peripheral_and_center_pillars_by_frame_according_to_nbrs():
    nbrs_dict = get_alive_pillars_to_alive_neighbors()
    frame_to_alive_pillars = get_alive_center_ids_by_frame_v2()

    frame_to_peripheral_center_dict = {}
    for frame, curr_frame_alive_pillars in frame_to_alive_pillars.items():
        frame_to_peripheral_center_dict[frame] = {}
        peripherals = []
        centers = []
        for p in curr_frame_alive_pillars:
            alive_nbrs = 0
            for nbr in nbrs_dict[p]:
                if nbr in curr_frame_alive_pillars:
                    alive_nbrs += 1
            if alive_nbrs <= Consts.NUM_NEIGHBORS_TO_CONSIDER_PERIPHERAL:
                peripherals.append(p)
            else:
                centers.append(p)

        frame_to_peripheral_center_dict[frame]["peripherals"] = peripherals
        frame_to_peripheral_center_dict[frame]["centrals"] = centers

    return frame_to_peripheral_center_dict


def get_type_of_pillars_to_frames(frame_to_peripheral_center_dict, pillars_type="peripherals"):
    specific_pillars_to_relevant_frames = {}
    all_pillars = []
    for pillars in frame_to_peripheral_center_dict.values():
        pillars_from_type = pillars[pillars_type]
        all_pillars.extend(pillars_from_type)

    all_pillars = set(all_pillars)
    for p in all_pillars:
        frames = []
        for frame, pillars in frame_to_peripheral_center_dict.items():
            if p in pillars[pillars_type]:
                frames.append(frame)
        specific_pillars_to_relevant_frames[p] = frames

    return specific_pillars_to_relevant_frames


def get_pillars_intensity_movement_sync_by_frames(pillar_to_frames_dict):
    """
    To each pillar - calculating the correlation between its intensities vector to its movements vector
    :return: dictionary of the correlation between those vectors to each pillar
    """
    pillars_movements_dict = get_alive_centers_movements()
    p_to_distances_dict = {}
    for p, v_list in pillars_movements_dict.items():
        distances = []
        for move_dict in v_list:
            distances.append(move_dict['distance'])
        p_to_distances_dict[p] = distances
    pillars_dist_movements_df = pd.DataFrame({str(k): v for k, v in p_to_distances_dict.items()})

    pillar_to_intens_dict = get_alive_pillars_to_intensities()
    pillar_intens_df = pd.DataFrame({str(k): v for k, v in pillar_to_intens_dict.items()})
    pillar_intens_df = pillar_intens_df[:-1]

    last_frame_possible = len(list(pillar_to_intens_dict.values())[0]) - 1
    pillars_intens_dist_corrs = {}
    for p, frames in pillar_to_frames_dict.items():
        intensities = []
        moves = []
        for frame in frames:
            if frame < last_frame_possible:
                intensities.append(pillar_intens_df.loc[frame][str(p)])
                moves.append(pillars_dist_movements_df.loc[frame][str(p)])
        if len(intensities) > 1 and len(moves) > 1:
            p_corr = pearsonr(intensities, moves)[0]

            pillars_intens_dist_corrs[p] = p_corr

    return pillars_intens_dist_corrs


def get_avg_correlation_pillars_intensity_movement_peripheral_vs_central(frame_to_peripheral_center_dict):
    periph_p_to_frames = get_type_of_pillars_to_frames(frame_to_peripheral_center_dict, pillars_type="peripherals")
    central_p_to_frames = get_type_of_pillars_to_frames(frame_to_peripheral_center_dict, pillars_type="centrals")

    # # TODO: test only
    # periph_p_to_frames_min2max = {}
    # for pillar, frames in periph_p_to_frames.items():
    #     min_frame = min(frames)
    #     max_frame = max(frames)
    #
    #     periph_p_to_frames_min2max[pillar] = list(range(min_frame, max_frame + 1))
    # periph_p_to_frames = periph_p_to_frames_min2max

    periph_sync = get_pillars_intensity_movement_sync_by_frames(periph_p_to_frames)
    central_sync = get_pillars_intensity_movement_sync_by_frames(central_p_to_frames)
    avg_periph_sync = np.nanmean(list(periph_sync.values()))
    avg_central_sync = np.nanmean(list(central_sync.values()))

    avg_corr_peripheral_vs_central = {
        "peripherals": avg_periph_sync,
        "centrals": avg_central_sync
    }

    # print("avg_periph_sync: " + str(avg_periph_sync))
    # print("avg_central_sync: " + str(avg_central_sync))

    with open(Consts.RESULT_FOLDER_PATH + "/avg_corr_peripheral_vs_central.json", 'w') as f:
        json.dump(avg_corr_peripheral_vs_central, f)

    return avg_periph_sync, avg_central_sync


def get_peripheral_and_center_pillars_by_frame_according_revealing_pillars_and_nbrs(
        pillars_frame_zero_are_central=True):
    """
    peripheral pillars = reveal by the cell along the movie and keep considered as peripheral as long as
    their number of neighbors < Consts.NUMBER_OF_NBRS_TO_CONSIDER_CENTRAL
    :return:
    """
    nbrs_dict = get_alive_pillars_to_alive_neighbors()
    frame_to_alive_pillars = get_alive_center_ids_by_frame_v2()
    first_frame_pillars = list(frame_to_alive_pillars.values())[0]
    if pillars_frame_zero_are_central:
        central_pillars = list(frame_to_alive_pillars.values())[0]
    else:
        central_pillars = []

    frame_to_peripheral_center_dict = {}
    frame_to_peripheral_center_dict[list(frame_to_alive_pillars.keys())[0]] = {"peripherals": [],
                                                                               "centrals": central_pillars}

    for frame, curr_frame_alive_pillars in dict(list(frame_to_alive_pillars.items())[1:]).items():
        frame_to_peripheral_center_dict[frame] = {}
        frame_peiph = []
        frame_centrals = []
        for p in curr_frame_alive_pillars:
            if p in first_frame_pillars:
                # 1. seen pillar is exist in first frame & pillars_frame_zero_are_central == true
                #  ->  frame_centrals.append(p)
                # 2. seen pillar is exist in first frame & pillars_frame_zero_are_central == false
                #  -> nowhere
                if pillars_frame_zero_are_central:
                    frame_centrals.append(p)
            # 3. not in this list -> continue...
            else:
                alive_nbrs = 0
                for n in nbrs_dict[p]:
                    if n in curr_frame_alive_pillars:
                        alive_nbrs += 1
                if alive_nbrs < Consts.NUMBER_OF_NBRS_TO_CONSIDER_CENTRAL:
                    frame_peiph.append(p)
                else:
                    frame_centrals.append(p)

        # central_pillars.extend(frame_centrals)

        frame_to_peripheral_center_dict[frame]["peripherals"] = frame_peiph
        frame_to_peripheral_center_dict[frame]["centrals"] = frame_centrals

    return frame_to_peripheral_center_dict


def get_correlations_in_first_and_second_half_of_exp():
    frames_length = len(get_images(get_images_path()))

    first_half_corrs = get_alive_pillars_symmetric_correlation(0, frames_length // 2)
    second_half_corrs = get_alive_pillars_symmetric_correlation(frames_length // 2, frames_length)
    overall_corrs = get_alive_pillars_symmetric_correlation(0, frames_length)

    return first_half_corrs, second_half_corrs, overall_corrs


def get_correlation_df_with_only_alive_pillars(corr_df):
    corr_df = corr_df.loc[:, (corr_df != 0).any(axis=0)]
    corr_df = corr_df.loc[:, (corr_df != 0).any(axis=1)]

    return corr_df


def correlation_diff(corr1_df, corr2_df):
    return corr2_df - corr1_df


def get_cc_pp_cp_correlations():
    nbrs_dict = get_alive_pillars_to_alive_neighbors()
    frame_to_alive_pillars = get_alive_center_ids_by_frame_v2()
    frame_to_periph_center_dict = get_peripheral_and_center_pillars_by_frame_according_revealing_pillars_and_nbrs(
        pillars_frame_zero_are_central=True)

    results = {}

    for p1, nbrs in nbrs_dict.items():
        for p2 in nbrs:
            if (p1, p2) in results or (p2, p1) in results:
                break
            cc_list = []
            cp_list = []
            prev_sequence = None
            for frame, alive_pillars in frame_to_alive_pillars.items():
                if p1 not in alive_pillars or p2 not in alive_pillars:
                    prev_sequence = None
                else:
                    curr_sequence = find_vert_type(p1, p2, frame, frame_to_periph_center_dict)
                    if curr_sequence == prev_sequence:
                        if curr_sequence == 'cc':
                            cc_list[-1].append(frame)
                        elif curr_sequence == 'cp':
                            cp_list[-1].append(frame)
                    else:
                        prev_sequence = curr_sequence
                        if prev_sequence == 'cc':
                            cc_list.append([frame])
                        elif prev_sequence == 'cp':
                            cp_list.append([frame])

        results[(p1, p2)] = {'cc': cc_list, 'cp': cp_list}

    alive_pillars_to_intensities = get_alive_pillars_to_intensities()

    cc_corr_list = []
    cp_corr_list = []

    for pair, series in results.items():
        p1 = pair[0]
        p2 = pair[1]
        cc_corr = get_correlation_of_lists(alive_pillars_to_intensities, p1, p2, series['cc'])
        if cc_corr is not None:
            cc_corr_list.append(cc_corr)

        cp_corr = get_correlation_of_lists(alive_pillars_to_intensities, p1, p2, series['cp'])
        if cp_corr is not None:
            cp_corr_list.append(cp_corr)

    return {'cc_corr': np.mean(cc_corr_list), 'cp_corr': np.mean(cp_corr_list)}


def get_correlation_of_lists(alive_pillars_to_intensities, p1, p2, series):
    SERIES_MIN_LENGTH = 3

    pair_corrs = []
    for ser in series:
        if len(ser) >= SERIES_MIN_LENGTH:
            p1_intens = alive_pillars_to_intensities[p1][ser[0]:ser[-1] + 1]
            p2_intens = alive_pillars_to_intensities[p2][ser[0]:ser[-1] + 1]

            ser_corr = pearsonr(p1_intens, p2_intens)[0]
            pair_corrs.append(ser_corr)
    if len(pair_corrs) == 0:
        return None
    pair_correlation = np.mean(pair_corrs)
    return pair_correlation


def find_vert_type(pillar_1, pillar_2, frame, frame_to_periph_center_dict):
    peripheral = frame_to_periph_center_dict[frame]['peripherals']
    central = frame_to_periph_center_dict[frame]['centrals']

    if pillar_1 in central and pillar_2 in central:
        return 'cc'

    if pillar_1 in peripheral and pillar_2 in peripheral:
        return 'cp'

    if (pillar_1 in central and pillar_2 in peripheral) or (pillar_1 in peripheral and pillar_2 in central):
        return 'cp'

    print("one of the pillars doens't exist", frame, pillar_1, pillar_2)


def get_neighbours_correlations_by_distance_from_cell_center():
    nbrs_dict = get_alive_pillars_to_alive_neighbors()
    all_corr = get_alive_pillars_symmetric_correlation()
    pillar2middle_img_steps = get_pillar2_middle_img_steps(nbrs_dict)

    distance2corrs = {dis: [] for dis in set(pillar2middle_img_steps.values())}

    seen_pairs = set()

    for p1, nbrs in nbrs_dict.items():
        for p2 in nbrs:
            if (p1, p2) not in seen_pairs and (p2, p1) not in seen_pairs:
                distance = find_vertex_distance_from_center(p1, p2, pillar2middle_img_steps)
                distance2corrs[distance].append(all_corr.loc[str(p1)][str(p2)])
                seen_pairs.add((p1, p2))

    result = {dis: np.mean(corrs) for dis, corrs in distance2corrs.items() if dis != 0}

    return result


def get_pillar2_middle_img_steps(nbrs_dict):
    alive_pillars = list(get_alive_pillars_to_alive_neighbors())
    middle_img = (np.mean([p[0] for p in alive_pillars]), np.mean([p[1] for p in alive_pillars]))
    # this returns only 1 center, change to get more
    closest_pillars_to_middle = min(alive_pillars,
                                    key=lambda alive_pillars:
                                    math.hypot(alive_pillars[1] - middle_img[1],
                                               alive_pillars[0] - middle_img[0]))

    pillar2_middle_img_steps = {}
    for p in alive_pillars:
        distance_from_middle_center = get_path_distance(nbrs_dict, p, closest_pillars_to_middle, alive_pillars)
        pillar2_middle_img_steps[p] = distance_from_middle_center

    pillar2_middle_img_steps[closest_pillars_to_middle] = 0

    return pillar2_middle_img_steps


def get_path_distance(nbrs_dict, source_pillar, dest_pillar, alive_pillars):
    pred = dict([(p, None) for p in alive_pillars])
    dist = dict([(p, None) for p in alive_pillars])

    if not BFS_for_distance_from_middle(nbrs_dict, source_pillar, dest_pillar, alive_pillars, pred, dist):
        print("Given source and destination are not connected")
        return 0

    path = []
    crawl = dest_pillar
    path.append(crawl)

    while pred[crawl] != -1:
        path.append(pred[crawl])
        crawl = pred[crawl]

    return len(path) - 1


def BFS_for_distance_from_middle(nbrs_dict, source_pillar, dest_pillar, alive_pillars, pred, dist):
    queue = []

    visited = dict([(p, False) for p in alive_pillars])

    for p in alive_pillars:
        dist[p] = 1000000
        pred[p] = -1

    visited[source_pillar] = True
    dist[source_pillar] = 0
    queue.append(source_pillar)

    while len(queue) != 0:
        u = queue[0]
        queue.pop(0)
        for nbr_p in nbrs_dict[u]:

            if not visited[nbr_p]:
                visited[nbr_p] = True
                dist[nbr_p] = dist[u] + 1
                pred[nbr_p] = u
                queue.append(nbr_p)

                # We stop BFS when we find
                # destination.
                if nbr_p == dest_pillar:
                    return True

    return False


def find_vertex_distance_from_center(p1, p2, pillar2middle_img_steps):
    return max([pillar2middle_img_steps[p1], pillar2middle_img_steps[p2]])






# def get_peripheral_and_center_pillars_by_frame_according_revealing_pillars():
#     frame_to_alive_pillars = get_alive_center_ids_by_frame_v2()
#     peripheral_pillars, central_pillars = get_peripheral_and_center_by_revealing_pillars()
#
#     frame_to_peripheral_center_dict = {}
#     for frame, curr_frame_alive_pillars in frame_to_alive_pillars.items():
#         frame_to_peripheral_center_dict[frame] = {}
#         peripherals = []
#         centers = []
#         for p in curr_frame_alive_pillars:
#             if p in peripheral_pillars:
#                 peripherals.append(p)
#             if p in central_pillars:
#                 centers.append(p)
#
#         frame_to_peripheral_center_dict[frame]["peripherals"] = peripherals
#         frame_to_peripheral_center_dict[frame]["centrals"] = centers
#
#     return frame_to_peripheral_center_dict


# def get_avg_correlation_pillars_intens_move_according_to_revealing_pillars_periph_vs_central():
#     frame_to_peripheral_center_dict = get_peripheral_and_center_pillars_by_frame_according_revealing_pillars_and_nbrs()
#
#     pillars_movements_dict = get_alive_centers_movements()
#     p_to_distances_dict = {}
#     for p, v_list in pillars_movements_dict.items():
#         distances = []
#         for move_dict in v_list:
#             distances.append(move_dict['distance'])
#         p_to_distances_dict[p] = distances
#     pillars_dist_movements_df = pd.DataFrame({str(k): v for k, v in p_to_distances_dict.items()})
#
#     pillar_to_intens_dict = get_alive_pillars_to_intensities()
#     pillar_intens_df = pd.DataFrame({str(k): v for k, v in pillar_to_intens_dict.items()})
#     pillar_intens_df = pillar_intens_df[:-1]
#
#     alive_p_to_frame = get_alive_pillar_to_frame()
#     periph_pillars_intens_dist_corrs = {}
#     for p in peripheral_pillars:
#         p_start_to_live = alive_p_to_frame[p]
#         intens_vec = pillar_intens_df.loc[p_start_to_live:][str(p)]
#         move_vector = pillars_dist_movements_df.loc[p_start_to_live:][str(p)]
#         if len(intens_vec) > 1 and len(move_vector) > 1:
#             p_corr = pearsonr(intens_vec, move_vector)[0]
#             periph_pillars_intens_dist_corrs[p] = p_corr
#
#     central_pillars_intens_dist_corrs = {}
#     for p in central_pillars:
#         p_start_to_live = alive_p_to_frame[p]
#         intens_vec = pillar_intens_df.loc[p_start_to_live:][str(p)]
#         move_vector = pillars_dist_movements_df.loc[p_start_to_live:][str(p)]
#         if len(intens_vec) > 1 and len(move_vector) > 1:
#             p_corr = pearsonr(intens_vec, move_vector)[0]
#             central_pillars_intens_dist_corrs[p] = p_corr
#
#     avg_periph_sync = np.nanmean(list(periph_pillars_intens_dist_corrs.values()))
#     avg_central_sync = np.nanmean(list(central_pillars_intens_dist_corrs.values()))
#
#     avg_corr_peripheral_vs_central = {
#         "peripherals": avg_periph_sync,
#         "centrals": avg_central_sync
#     }
#
#     with open(Consts.RESULT_FOLDER_PATH + "/avg_corr_peripheral_vs_central.json", 'w') as f:
#         json.dump(avg_corr_peripheral_vs_central, f)
#
#     return avg_periph_sync, avg_central_sync
