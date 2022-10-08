from scipy.stats import pearsonr
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
    all_pillars_corr = all_pillar_intensity_df.corr()

    if Consts.USE_CACHE:
        with open(path, 'wb') as handle:
            pickle.dump(all_pillars_corr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_pillars_corr


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


def get_alive_pillars_symmetric_correlation():
    """
    Create dataframe of alive pillars correlation as the correlation according to
    maximum_frame(pillar a start to live, pillar b start to live) -> correlation(a, b) == correlation(b, a)
    :return:
    """
    if Consts.USE_CACHE and os.path.isfile(Consts.alive_pillars_corr_cache_path):
        with open(Consts.alive_pillars_corr_cache_path, 'rb') as handle:
            gc_df = pickle.load(handle)
            return gc_df

    pillar2mask = get_last_img_mask_for_each_pillar()
    frame_to_alive_pillars = get_frame_to_alive_pillars_by_same_mask(pillar2mask)
    alive_pillars_to_frame = {}
    for frame, alive_pillars_in_frame in frame_to_alive_pillars.items():
        for alive_pillar in alive_pillars_in_frame:
            if alive_pillar not in alive_pillars_to_frame:
                alive_pillars_to_frame[alive_pillar] = frame

    alive_pillars_intens = get_alive_pillars_to_intensities()
    alive_pillars = list(alive_pillars_intens.keys())
    alive_pillars_str = [str(p) for p in alive_pillars]
    pillars_corr = pd.DataFrame(0.0, index=alive_pillars_str, columns=alive_pillars_str)

    # epsilon = 0.0001
    # Symmetric correlation - calc correlation of 2 pillars start from the frame they are both alive: maxFrame(A, B)
    for p1 in alive_pillars:
        p1_living_frame = alive_pillars_to_frame[p1]
        for p2 in alive_pillars:
            p2_living_frame = alive_pillars_to_frame[p2]
            both_alive_frame = max(p1_living_frame, p2_living_frame)
            p1_relevant_intens = alive_pillars_intens[p1][both_alive_frame - 1:]
            p2_relevant_intens = alive_pillars_intens[p2][both_alive_frame - 1:]
            if len(p1_relevant_intens) > 1 and len(p2_relevant_intens) > 1:
                pillars_corr.loc[str(p2), str(p1)] = pearsonr(p1_relevant_intens, p2_relevant_intens)[0]
            # if pillars_corr.loc[str(p2), str(p1)] == 1.0 or pillars_corr.loc[str(p2), str(p1)] == 1:
            #     pillars_corr.loc[str(p2), str(p1)] -= epsilon
            # if pillars_corr.loc[str(p2), str(p1)] == 0.0 or pillars_corr.loc[str(p2), str(p1)] == 0:
            #     pillars_corr.loc[str(p2), str(p1)] += epsilon

    if Consts.USE_CACHE:
        with open(Consts.alive_pillars_corr_cache_path, 'wb') as handle:
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


def get_number_of_inwards_outwards_gc_edges(gc_df):
    """
    Count the number of inward and outward gc edges
    :param gc_df:
    :return:
    """
    all_alive_centers = get_alive_centers()
    center = get_center_of_points(all_alive_centers)
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
                ang = get_angle(int_col, int_row, center)
                if math.dist(int_col, center) < math.dist(int_row, center) and ang >= 135:
                    outwards += 1
                    out_edges.append((col, row))
                elif math.dist(int_col, center) > math.dist(int_row, center) and ang <= 45:
                    inwards += 1
                    in_edges.append((col, row))
    if total_edges == 0:
        in_percentage, out_percentage = 0, 0
    else:
        in_percentage = (inwards / total_edges)
        out_percentage = (outwards / total_edges)
    print("Number of total edges: " + str(total_edges))
    print("Number of inwards gc edges: " + str(inwards) + " (" + str(in_percentage * 100) + "%)")
    print("Number of outwards gc edges: " + str(outwards) + " (" + str(out_percentage * 100) + "%)")
    return inwards, outwards, in_edges, out_edges, in_percentage, out_percentage


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
        reciprocity = two_sided_edge / total_edges

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

    print("Heterogeneity: " + str(g_heterogeneity))
    return g_heterogeneity


def get_output_df(output_path_type):
    output_path = get_output_path(output_path_type)

    output_df = pd.read_csv(output_path, index_col=0)
    output_df.drop('passed_stationary', axis=1, inplace=True)

    return output_df


def get_output_path(output_path_type):
    if Consts.inner_cell:
        output_path = './features output/output_inner_cell_' + output_path_type +'.csv'
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
    return stat, pval
