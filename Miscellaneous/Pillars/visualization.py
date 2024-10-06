import csv

from matplotlib.pyplot import axline

from Pillars.analyzer import *
import networkx as nx
import seaborn as sns
from Pillars.consts import *
import Pillars.consts as consts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib import animation
from matplotlib.pyplot import cm
import matplotlib.colors as mcolors
from matplotlib.animation import PillowWriter

from Pillars.pillar_neighbors import *


def show_last_image_masked(mask_path=None, pillars_mask=None, save_mask=True, frame=None):
    """
    show the mask on the video's last image
    :param mask_path:
    :return:
    """
    img = get_last_image()
    if frame is not None:
        img = frame

    # plt.imshow(img, cmap=plt.cm.gray)
    # if Consts.RESULT_FOLDER_PATH is not None:
    #     plt.savefig(Consts.RESULT_FOLDER_PATH + "/last_image.png")
    #     plt.close()  # close the figure window

    if Consts.SHOW_GRAPH:
        plt.show()

    if mask_path is not None:
        with open(mask_path, 'rb') as f:
            pillars_mask = np.load(f)
    pillars_mask = 255 - pillars_mask
    mx = ma.masked_array(img, pillars_mask)
    plt.imshow(mx, cmap=plt.cm.gray)
    plt.axis('off')
    if save_mask and Consts.RESULT_FOLDER_PATH is not None:
        plt.savefig(Consts.RESULT_FOLDER_PATH + "/mask.png")
        plt.close()  # close the figure window
        print("saved new mask.png")
    if Consts.SHOW_GRAPH:
        # add the centers location on the image
        # centers = find_centers()
        # for center in centers:

        #     s = '(' + str(center[0]) + ',' + str(center[1]) + ')'
        #     plt.text(center[VIDEO_06_LENGTH], center[0], s=s, fontsize=7, color='red')
        plt.show()
    return mx


def indirect_alive_neighbors_correlation_plot(pillar_location, only_alive=True):
    """
    Plotting the correlation of a pillar with its all indirected neighbors
    :param pillar_location:
    :param only_alive:
    :return:
    """

    my_G = nx.Graph()
    nodes_loc = get_all_center_generated_ids()
    node_loc2index = {}
    for i in range(len(nodes_loc)):
        node_loc2index[nodes_loc[i]] = i
        my_G.add_node(i)

    if only_alive:
        pillars = get_pillar_to_intensity_norm_by_inner_pillar_noise()
    else:
        # pillars = get_pillar_to_intensities(get_images_path())
        pillars = normalized_intensities_by_mean_background_intensity()

    pillar_loc = pillar_location
    indirect_neighbors_dict = get_pillar_indirect_neighbors_dict(pillar_location)
    # alive_pillars = get_alive_pillars_to_intensities()
    directed_neighbors = get_pillar_directed_neighbors(pillar_loc)
    indirect_alive_neighbors = {pillar: indirect_neighbors_dict[pillar] for pillar in pillars.keys() if
                                pillar not in directed_neighbors}
    pillars_corr = get_indirect_neighbors_correlation(pillar_loc, only_alive)
    for no_n1 in indirect_alive_neighbors.keys():
        my_G.add_edge(node_loc2index[pillar_loc], node_loc2index[no_n1])
        try:
            my_G[node_loc2index[pillar_loc]][node_loc2index[no_n1]]['weight'] = pillars_corr[str(pillar_loc)][
                str(no_n1)]
        except:
            x = -1

    edges, weights = zip(*nx.get_edge_attributes(my_G, 'weight').items())

    cmap = plt.cm.seismic
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    # pillar2mask = get_last_img_mask_for_each_pillar()
    frame2pillars = get_alive_center_ids_by_frame_v3()  # get_frame_to_alive_pillars_by_same_mask(pillar2mask)

    nodes_loc_y_inverse = [(loc[1], loc[0]) for loc in nodes_loc]
    nx.draw(my_G, nodes_loc_y_inverse, with_labels=True, node_color='gray', edgelist=edges, edge_color=weights,
            width=3.0,
            edge_cmap=cmap)
    plt.colorbar(sm)
    if Consts.SHOW_GRAPH:
        plt.show()


def correlation_plot(only_alive=True,
                     neighbors_str='all',
                     alive_correlation_type='symmetric',
                     pillars_corr_df=None,
                     frame_to_show=None,
                     save_fig=False):
    """
    Plotting graph of correlation between neighboring pillars
    Each point represent pillar itop_5_neighboring_corr_animationn its exact position in the image, and the size of each point represent how many
    time frames the pillar was living (the larger the pillar, the sooner he started to live)
    :param only_alive:
    :param neighbors_str:
    :param alive_correlation_type:
    :return:
    """
    my_G = nx.Graph()
    # last_img = get_last_image()
    alive_centers = get_seen_centers_for_mask()
    # nodes_loc = generate_centers_from_alive_centers(alive_centers, Consts.IMAGE_SIZE_ROWS, Consts.IMAGE_SIZE_COLS)
    nodes_loc = get_alive_pillar_ids_overall_v3()
    nodes_loc = list(nodes_loc)
    if neighbors_str == 'alive2back':
        neighbors = get_alive_pillars_in_edges_to_l1_neighbors()[0]
    elif neighbors_str == 'back2back':
        neighbors = get_background_level_1_to_level_2()
    elif neighbors_str == 'random':
        neighbors = get_random_neighbors()
    else:
        neighbors = get_alive_pillars_to_alive_neighbors()

    node_loc2index = {}
    for i in range(len(nodes_loc)):
        node_loc2index[nodes_loc[i]] = i
        my_G.add_node(i)

    if alive_correlation_type == 'all':
        alive_pillars_correlation = get_alive_pillars_correlation()
    elif alive_correlation_type == 'symmetric':
        alive_pillars_correlation = get_alive_pillars_symmetric_correlation()
    elif alive_correlation_type == 'custom':
        alive_pillars_correlation = pillars_corr_df
    all_pillars_corr = get_all_pillars_correlations()

    if only_alive:
        correlation = alive_pillars_correlation
    else:
        correlation = all_pillars_corr

    for n1 in neighbors.keys():
        for n2 in neighbors[n1]:
            my_G.add_edge(node_loc2index[n1], node_loc2index[n2])
            try:
                my_G[node_loc2index[n1]][node_loc2index[n2]]['weight'] = correlation[str(n1)][str(n2)]
            except:
                x = 1

    edges, weights = zip(*nx.get_edge_attributes(my_G, 'weight').items())
    cmap = plt.cm.viridis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    # pillar2mask = get_last_img_mask_for_each_pillar()
    frame2pillars = get_alive_center_ids_by_frame_v3()  # get_frame_to_alive_pillars_by_same_mask(pillar2mask)
    nodes_loc_y_inverse = [(loc[1], loc[0]) for loc in nodes_loc]
    if frame_to_show:
        img_to_show = get_images(get_images_path())[frame_to_show]
    else:
        img_to_show = get_images(get_images_path())[-1]
    # plt.imshow(img_to_show, cmap=plt.cm.gray)
    plt.imshow(get_last_image_whiten(Consts.build_image), cmap=plt.cm.gray)

    nx.draw(my_G, nodes_loc_y_inverse, with_labels=False, node_color='black', edgelist=edges, edge_color=weights,
            width=1,
            edge_cmap=cmap, node_size=1,
            vmin=-1, vmax=1, edge_vmin=-1, edge_vmax=1)
    # nx.draw_networkx_edges(my_G, nodes_loc_y_inverse, alpha=0.5)
    plt.colorbar(sm)
    if save_fig:
        plt.savefig('../corrs_graph_fig_1d.svg', format="svg")
    if Consts.SHOW_GRAPH:
        plt.show()
    x = 1


def build_gc_directed_graph(gc_df, non_stationary_pillars=None, inwards=None, outwards=None, random_neighbors=False,
                            draw=True):
    """
    Plotting a directed graph where an arrow represent that the pillar was "granger cause" the other pillar
    :param gc_df: dataframe with granger causality significance values
    :param only_alive:
    :return:
    """
    #
    # if Consts.USE_CACHE and os.path.isfile(Consts.gc_graph_cache_path):
    #     with open(Consts.gc_graph_cache_path, 'rb') as handle:
    #         pillar_to_neighbors = pickle.load(handle)
    #         return pillar_to_neighbors

    my_G = nx.Graph().to_directed()
    nodes_loc = get_all_center_generated_ids()
    # neighbors1, neighbors2 = get_pillar_to_neighbors()
    node_loc2index = {}
    for i in range(len(nodes_loc)):
        node_loc2index[str(nodes_loc[i])] = i
        my_G.add_node(i)
    # alive_pillars_correlation = get_alive_pillars_correlation()
    if random_neighbors:
        neighbors = get_random_neighbors()
    else:
        neighbors = get_alive_pillars_to_alive_neighbors()

    if Consts.only_alive:
        correlation = get_alive_pillars_symmetric_correlation()
    else:
        correlation = get_all_pillars_correlations()

    p_vals_lst = []
    for col in gc_df.keys():
        for row, _ in gc_df.iterrows():
            if eval(row) in neighbors[eval(col)]:
                p_vals_lst.append(gc_df[col][row])
            if gc_df[col][row] < Consts.gc_pvalue_threshold and eval(row) in neighbors[eval(col)]:
                # if edges_direction_lst:
                #     if (col, row) in edges_direction_lst:
                #         my_G.add_edge(node_loc2index[col], node_loc2index[row])
                #         try:
                #             my_G[node_loc2index[col]][node_loc2index[row]]['weight'] = correlation[col][row]
                #         except:
                #             x = 1
                # else:
                my_G.add_edge(node_loc2index[col], node_loc2index[row])
                try:
                    my_G[node_loc2index[col]][node_loc2index[row]]['weight'] = correlation[col][row]
                except:
                    x = 1

    # return my_G, p_vals_lst
    if draw:
        nodes_loc_y_inverse = [(loc[1], loc[0]) for loc in nodes_loc]

        if nx.get_edge_attributes(my_G, 'weight') == {}:
            return

        edges, weights = zip(*nx.get_edge_attributes(my_G, 'weight').items())
        # edges = list(filter(lambda x: x[0] == 52, edges))

        img = get_last_image_whiten(build_image=Consts.build_image)
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')

        if not non_stationary_pillars:
            non_stationary_pillars = []
        if not inwards:
            inwards = []
        if not outwards:
            outwards = []

        node_idx2loc = {v: k for k, v in node_loc2index.items()}

        node_color = []
        for node in my_G.nodes():
            if node_idx2loc[node] in non_stationary_pillars:
                node_color.append('red')
            else:
                node_color.append('black')

        edge_color = []
        for edge in my_G.edges():
            e = (node_idx2loc[edge[0]], node_idx2loc[edge[1]])
            if e in inwards:
                edge_color.append('green')
            elif e in outwards:
                edge_color.append('blue')
            else:
                edge_color.append('gray')

        node_size = [20 if c == 'red' else 1 for c in node_color]

        nx.draw(my_G, nodes_loc_y_inverse, node_color=node_color, edgelist=edges, edge_color=edge_color,
                width=3.0,
                node_size=node_size)
        nx.draw_networkx_labels(my_G, nodes_loc_y_inverse, font_color="whitesmoke", font_size=8)

        # plt.scatter(get_image_size()[0]/2, get_image_size()[1]/2, s=250, c="red")

        # ax.plot()
        if Consts.RESULT_FOLDER_PATH is not None:
            plt.savefig(Consts.RESULT_FOLDER_PATH + "/gc.png")
            plt.close()  # close the figure window
            print("saved gc.png")
        if Consts.SHOW_GRAPH:
            plt.show()
        x = 1


# TODO: delete
def build_gc_directed_graph_test(gc_df, non_stationary_pillars=None, inwards=None, outwards=None, only_alive=True,
                                 draw=True):
    """
    Plotting a directed graph where an arrow represent that the pillar was "granger cause" the other pillar
    :param gc_df: dataframe with granger causality significance values
    :param only_alive:
    :return:
    """
    my_G = nx.Graph().to_directed()
    nodes_loc = get_all_center_generated_ids()
    # neighbors1, neighbors2 = get_pillar_to_neighbors()
    node_loc2index = {}
    for frame in range(len(nodes_loc)):
        node_loc2index[str(nodes_loc[frame])] = frame
        my_G.add_node(frame)
    # alive_pillars_correlation = get_alive_pillars_correlation()
    alive_pillars_correlation = get_alive_pillars_symmetric_correlation()
    all_pillars_corr = get_all_pillars_correlations()
    neighbors = get_alive_pillars_to_alive_neighbors()

    if only_alive:
        correlation = alive_pillars_correlation
    else:
        correlation = all_pillars_corr

    for col in gc_df.keys():
        for row, _ in gc_df.iterrows():
            if gc_df[col][row] < Consts.gc_pvalue_threshold and eval(row) in neighbors[eval(col)]:
                my_G.add_edge(node_loc2index[col], node_loc2index[row])
                try:
                    my_G[node_loc2index[col]][node_loc2index[row]]['weight'] = correlation[col][row]
                except:
                    x = 1

    if draw:
        cmap = plt.cm.seismic
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
        # plt.colorbar(sm)

        # pillar2mask = get_last_img_mask_for_each_pillar()
        frame2alive_pillars = get_alive_center_ids_by_frame_v3()  # get_frame_to_alive_pillars_by_same_mask(pillar2mask)
        nodes_loc_y_inverse = [(loc[1], loc[0]) for loc in nodes_loc]

        edges, weights = zip(*nx.get_edge_attributes(my_G, 'weight').items())
        # edges = list(filter(lambda x: x[0] == 52, edges))

        img = get_last_image_whiten(build_image=Consts.build_image)
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')

        if not non_stationary_pillars:
            non_stationary_pillars = []
        if not inwards:
            inwards = []
        if not outwards:
            outwards = []

        node_idx2loc = {v: k for k, v in node_loc2index.items()}

        node_color = []
        for node in my_G.nodes():
            if node_idx2loc[node] in non_stationary_pillars:
                node_color.append('red')
            else:
                node_color.append('black')

        edge_color = []
        for edge in my_G.edges():
            e = (node_idx2loc[edge[0]], node_idx2loc[edge[1]])
            if e in inwards:
                edge_color.append('green')
            elif e in outwards:
                edge_color.append('blue')
            else:
                edge_color.append('gray')

        node_size = [20 if c == 'red' else 1 for c in node_color]

        nx.draw(my_G, nodes_loc_y_inverse, with_labels=True, node_color=node_color, edgelist=edges,
                edge_color=edge_color,
                width=3.0,
                node_size=node_size)
        nx.draw_networkx_labels(my_G, nodes_loc_y_inverse, font_color="whitesmoke")

        # ax.plot()
        if Consts.SHOW_GRAPH:
            plt.show()
        x = 1
    return my_G


def correlation_histogram(correlations_df):
    """
    Plotting a histogram of the pillars correlations
    :param correlations_df:
    :return:
    """
    corr = set()
    correlations = correlations_df
    for i in correlations:
        for j in correlations:
            if i != j:
                corr.add(correlations[i][j])
    corr_array = np.array(list(corr))
    mean_corr = np.mean(corr_array)
    sns.histplot(data=corr_array, kde=True)
    plt.xlabel("Correlation")
    if Consts.SHOW_GRAPH:
        plt.show()
    mean_corr = format(mean_corr, ".3f")
    print("mean correlations: " + str(mean_corr))
    return mean_corr


def neighbors_correlation_histogram(correlations_lst):
    """
    Display histogram plot of the correlations between the neighbors
    :param correlations_df:
    :param neighbors_dict:
    :param symmetric_corr:
    :return:
    """
    sns.histplot(data=correlations_lst, kde=True)
    plt.xlim(-1, 1)
    plt.title("Correlation of Neighbors Pillars")
    plt.xlabel("Correlation")
    if Consts.RESULT_FOLDER_PATH is not None:
        plt.savefig(Consts.RESULT_FOLDER_PATH + "/neighbors_corr_histogram.png")
        plt.close()  # close the figure window
    print("neighbors mean correlation: " + str(np.nanmean(correlations_lst)))
    if Consts.SHOW_GRAPH:
        plt.show()


def non_neighbors_correlation_histogram(correlations_lst):
    sns.histplot(data=correlations_lst, kde=True)
    plt.xlim(-1, 1)
    plt.title("Correlation of Non-Neighbors Pillars")
    plt.xlabel("Correlation")
    if Consts.RESULT_FOLDER_PATH is not None:
        plt.savefig(Consts.RESULT_FOLDER_PATH + "/non_neighbors_corr_histogram.png")
        plt.close()  # close the figure window
    print("non-neighbors mean correlation: " + str(np.nanmean(correlations_lst)))
    if Consts.SHOW_GRAPH:
        plt.show()


def nbrs_and_non_nbrs_corrs_histogram(correlations_dict, neighbors_dict):
    nbrs_mean_corr, nbrs_corrs_list = get_neighbors_avg_correlation(correlations_dict, neighbors_dict)
    non_nbrs_mean_corr, non_nbrs_corrs_list = get_non_neighbors_mean_correlation(correlations_dict, neighbors_dict)

    sns.histplot(data=nbrs_corrs_list, label="nbrs corrs", kde=True, alpha=0.3, stat='density')
    sns.histplot(data=non_nbrs_corrs_list, label="non-neighbors corrs", kde=True, alpha=0.3, stat='density')
    plt.xlabel("Correlations")
    plt.title("Neighbors & Non-Neighbors Correlation Histogram")
    print("avg nbrs corrs:", nbrs_mean_corr)
    print("avg non-nbrs corrs:", non_nbrs_mean_corr)
    plt.legend()
    plt.show()


def plot_pillar_time_series(pillars_loc=None, temporal_res=30, img_res=0.1565932, inner_pillar=False, ring_vs_inner=False,
                            inner_pillar_norm=False, series_to_plot=None, custom_p_to_intens=None, color=None, save_fig=False):
    """
    Plotting a time series graph of the pillar intensity over time
    :return:
    """
    if color is None:
        color = 'tab:blue'
    if Consts.normalized:
        pillar2intens = normalized_intensities_by_mean_background_intensity()
    elif inner_pillar:
        pillar2intens = pillar_to_inner_intensity()
        color = 'tab:orange'
    elif ring_vs_inner:
        inner = pillar_to_inner_intensity()
        ring = get_pillar_to_intensity_norm_by_inner_pillar_noise()
    elif inner_pillar_norm:
        p2intens_norm = pillar_to_inner_intensity_norm_by_noise()
    elif custom_p_to_intens is not None:
        pillar2intens = custom_p_to_intens
    else:
        pillar2intens = get_pillar_to_intensity_norm_by_inner_pillar_noise()
        # pillar2intens = get_pillar_to_intensities(get_images_path())

    plt.rcParams.update({'font.size': 10})
    plt.rcParams['font.family'] = 'Arial'

    if ring_vs_inner:
        intensities_inner = inner[pillars_loc]
        intensities_ring = ring[pillars_loc]
        x = [i * temporal_res for i in range(len(intensities_ring))]
        intensities_inner = [i * img_res for i in intensities_inner]
        intensities_ring = [i * img_res for i in intensities_ring]
        plt.plot(x, intensities_ring, label='ring intensity')
        plt.plot(x, intensities_inner, label='pillar intensity')
    elif inner_pillar and type(pillars_loc) is list:
        intensities_1 = pillar2intens[pillars_loc[0]]
        intensities_2 = pillar2intens[pillars_loc[1]]
        x = [i * temporal_res for i in range(len(intensities_1))]
        intensities_1 = [i * img_res for i in intensities_1]
        intensities_2 = [i * img_res for i in intensities_2]
        plt.plot(x, intensities_1, label=str(pillars_loc[0]), color='tab:blue')
        plt.plot(x, intensities_2, label=str(pillars_loc[1]), color='tab:orange')
    elif inner_pillar_norm:
        intensities_1 = p2intens_norm[pillars_loc[0]]
        intensities_2 = p2intens_norm[pillars_loc[1]]
        x = [i * temporal_res for i in range(len(intensities_1))]
        intensities_1 = [i * img_res for i in intensities_1]
        intensities_2 = [i * img_res for i in intensities_2]
        plt.plot(x, intensities_1, label=str(pillars_loc[0]), color='tab:blue')
        plt.plot(x, intensities_2, label=str(pillars_loc[1]), color='tab:orange')
    elif series_to_plot is not None:
        intensities = series_to_plot.tolist()
        x = [i * temporal_res for i in range(len(intensities))]
        intensities_1 = [i * img_res for i in intensities]
        plt.plot(x, intensities_1, label="mean background ts", color='tab:red')
    else:
        intensities_1 = pillar2intens[pillars_loc]
        x = [i * temporal_res for i in range(len(intensities_1))]
        len1 = len(x)
        x = [(i / 60) for i in x if i >= 400]
        len2 = len(x)
        intensities_1 = [(intens * img_res) for i,intens in enumerate(intensities_1) if i >= (len1 - len2)]
        plt.plot(x, intensities_1, label=str(pillars_loc), color=color)

    plt.xlabel('Time (sec)')
    plt.ylabel('Intensity (micron)')
    plt.yticks(rotation=90)
    # plt.legend(title='Pillar Location')
    if save_fig:
        plt.savefig('../ts_fig_1c.svg', format="svg")
    if Consts.SHOW_GRAPH:
        plt.show()


def compare_neighbors_corr_histogram_random_vs_real(random_amount):
    """
    Show on same plot the mean correlation of the real neighbors and
    :param random_amount:
    :return:
    """
    mean_original_nbrs, _ = get_neighbors_avg_correlation(get_alive_pillars_symmetric_correlation(),
                                                          get_alive_pillars_to_alive_neighbors())
    means = []
    rand = []
    for i in range(random_amount):
        mean_random_nbrs = get_neighbors_avg_correlation(get_alive_pillars_symmetric_correlation(),
                                                         get_random_neighbors())
        means.append(mean_random_nbrs)
        rand.append('random' + str(i + 1))
    print("Random nbrs mean correlation: " + str(np.mean(means)))
    means.append(mean_original_nbrs)
    rand.append('original')
    fig, ax = plt.subplots()
    ax.scatter(rand, means)
    plt.ylabel('Average Correlation')
    plt.xticks(rotation=45)
    if Consts.RESULT_FOLDER_PATH is not None:
        plt.savefig(Consts.RESULT_FOLDER_PATH + "/neighbors_corr_histogram_random_vs_real.png")
        plt.close()  # close the figure window
        print("saved neighbors_corr_histogram_random_vs_real.png")
    if Consts.SHOW_GRAPH:
        plt.show()


def edges_distribution_plots(gc_df, pillar_intensity_dict=None):
    alive_pillars_correlation = get_alive_pillars_symmetric_correlation()
    neighbors = get_alive_pillars_to_alive_neighbors()
    no_edge = []
    one_sided_edge = []
    two_sided_edge = []

    rows = list(gc_df.keys())
    for i, col in enumerate(gc_df.keys()):
        for j in range(i + 1, len(rows)):
            row = rows[j]
            if eval(row) in neighbors[eval(col)]:
                if pillar_intensity_dict:
                    corr = pearsonr(pillar_intensity_dict[col], pillar_intensity_dict[row])[0]
                else:
                    corr = alive_pillars_correlation[col][row]
                if gc_df[col][row] < 0.05 and gc_df[row][col] < 0.05:
                    two_sided_edge.append(corr)
                elif (gc_df[col][row] < 0.05 and gc_df[row][col] > 0.05) or (
                        gc_df[col][row] > 0.05 and gc_df[row][col] < 0.05):
                    one_sided_edge.append(corr)
                else:
                    no_edge.append(corr)

    # for col in gc_df.keys():
    #     for row, _ in gc_df.iterrows():
    #         if eval(row) in neighbors[eval(col)]:
    #             if pillar_intensity_dict:
    #                 corr = pearsonr(pillar_intensity_dict[col], pillar_intensity_dict[row])[0]
    #             else:
    #                 corr = alive_pillars_correlation[col][row]
    #             if gc_df[col][row] < 0.05 and gc_df[row][col] < 0.05:
    #                 two_sided_edge.append(corr)
    #             elif (gc_df[col][row] < 0.05 and gc_df[row][col] > 0.05) or (
    #                     gc_df[col][row] > 0.05 and gc_df[row][col] < 0.05):
    #                 one_sided_edge.append(corr)
    #             else:
    #                 no_edge.append(corr)
    sns.histplot(data=no_edge, kde=True)
    plt.xlabel("Correlations")
    plt.title('Correlation of no edges between neighbors')
    if Consts.SHOW_GRAPH:
        plt.show()
    print("number of neighbors with no edges: " + str(len(no_edge)))
    print("average of neighbors with no edges: " + str(np.mean(no_edge)))
    sns.histplot(data=one_sided_edge, kde=True)
    plt.xlabel("Correlations")
    plt.title('Correlation of 1 sided edges between neighbors')
    if Consts.SHOW_GRAPH:
        plt.show()
    print("number of neighbors with 1 edge: " + str(len(one_sided_edge)))
    print("average of neighbors with 1 edge: " + str(np.mean(one_sided_edge)))
    sns.histplot(data=two_sided_edge, kde=True)
    plt.xlabel("Correlations")
    plt.title('Correlation of 2 sided edges between neighbors')
    if Consts.SHOW_GRAPH:
        plt.show()
    print("number of neighbors with 2 edges: " + str(len(two_sided_edge)))
    print("average of neighbors with 2 edges: " + str(np.mean(two_sided_edge)))


def in_out_degree_distribution(in_degree_list, out_degree_list):
    sns.histplot(data=in_degree_list, kde=True)
    plt.xlabel("In Degree")
    plt.title('Pillars In Degree Distribution')
    if Consts.SHOW_GRAPH:
        plt.show()
    print("In degree average: " + str(np.mean(in_degree_list)))
    sns.histplot(data=out_degree_list, kde=True)
    plt.xlabel("Out Degree")
    plt.title('Pillars Out Degree Distribution')
    if Consts.SHOW_GRAPH:
        plt.show()
    print("Out degree average: " + str(np.mean(out_degree_list)))


def features_correlations_heatmap(output_path_type, custom_df=None):
    if custom_df is None:
        output_df = get_output_df(output_path_type)
    else:
        output_df = custom_df
    f, ax = plt.subplots(figsize=(10, 8))
    corr = output_df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), annot=True,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=45)
    if Consts.SHOW_GRAPH:
        plt.show()


def pca_number_of_components(output_path_type, custom_df=None):
    # get the number of components to pca - a rule of thumb is to preserve around 80 % of the variance
    if custom_df is None:
        output_df = get_output_df(output_path_type)
    else:
        output_df = custom_df
    x = StandardScaler().fit_transform(output_df)
    pca = PCA()
    pca.fit(x)
    plt.figure(figsize=(10, 8))
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.plot(range(1, output_df.shape[1] + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    if Consts.SHOW_GRAPH:
        plt.show()


def plot_2d_pca_components(targets_list, output_path_type, n_components, custom_df=None):
    # plot 2D pca of all components in one plot
    pca, principal_components = get_pca(output_path_type, n_components=n_components, custom_df=custom_df)
    labels = {
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
    fig = px.scatter_matrix(
        principal_components,
        labels=labels,
        dimensions=range(principal_components.shape[1]),
        color=targets_list
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()


def components_feature_weight(pca):
    # Components Feature Weight
    if len(pca.components_) == 2:
        subplot_titles = ("PC1", "PC2")
        cols = 2
    if len(pca.components_) == 3:
        subplot_titles = ("PC1", "PC2", "PC3")
        cols = 3
    if len(pca.components_) == 4:
        subplot_titles = ("PC1", "PC2", "PC3", "PC4")
        cols = 4
    fig = make_subplots(rows=1, cols=cols, subplot_titles=subplot_titles)
    fig.add_trace(
        go.Scatter(y=pca.components_[0], mode='markers'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=pca.components_[1], mode='markers'),
        row=1, col=2
    )
    if len(pca.components_) == 3:
        fig.add_trace(
            go.Scatter(y=pca.components_[2], mode='markers'),
            row=1, col=3
        )
    if len(pca.components_) == 4:
        fig.add_trace(
            go.Scatter(y=pca.components_[2], mode='markers'),
            row=1, col=3
        )
        fig.add_trace(
            go.Scatter(y=pca.components_[3], mode='markers'),
            row=1, col=4
        )
    fig.update_xaxes(title_text="Feature", row=1, col=2)
    fig.update_yaxes(title_text="Weight", row=1, col=1)
    fig.show()


def features_coefficient_heatmap(pca, output_path_type, custom_df=None):
    if custom_df is None:
        output_df = get_output_df(output_path_type)
    else:
        output_df = custom_df
    for i in range(len(pca.components_)):
        ax = sns.heatmap(pca.components_[i].reshape(1, output_df.shape[1]),
                         cmap='seismic',
                         yticklabels=["PC" + str(i + 1)],
                         xticklabels=list(output_df.columns),
                         cbar_kws={"orientation": "horizontal"},
                         annot=True, vmin=-1, vmax=1)
        ax.set_aspect("equal")
        ax.tick_params(axis='x', rotation=20)
        sns.set(rc={"figure.figsize": (3, 3)})
        if Consts.SHOW_GRAPH:
            plt.show()


def gc_edge_probability_original_vs_random(gc_df, gc_edge_prob_lst):
    fig, ax = plt.subplots()
    original = probability_for_gc_edge(gc_df, random_neighbors=False)
    ax.scatter(0, np.mean(gc_edge_prob_lst), label="random")
    ax.scatter(1, original, label="original")
    plt.ylabel("Edge Probability")
    plt.title("GC Edge Probability - Original vs. Random Neighbors")
    ax.legend()
    if Consts.RESULT_FOLDER_PATH is not None:
        plt.savefig(Consts.RESULT_FOLDER_PATH + "/gc_probability_original_vs_random.png")
        plt.close()  # close the figure window
        print("saved gc_probability_original_vs_random.png")
    if Consts.SHOW_GRAPH:
        plt.show()


# decide on the number of clustering to k-means. wcss = Within Cluster Sum of Squares
def number_clusters_kmeans(principalComponents):
    global kmeans_pca
    wcss = []
    for i in range(1, 9):
        kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans_pca.fit(principalComponents)
        wcss.append(kmeans_pca.inertia_)
    plt.figure(figsize=(10, 8))
    plt.title('K-means with PCA Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.plot(range(1, 9), wcss, marker='o', linestyle='--')
    if Consts.SHOW_GRAPH:
        plt.show()


# implement k-means with pca
def k_means(principalComponents, output_path_type, n_clusters=2, custom_df=None):
    global kmeans_pca
    kmeans_pca = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans_pca.fit(principalComponents)
    if custom_df is None:
        output_df = get_output_df(output_path_type)
    else:
        output_df = custom_df
    df_segm_pca_kmeans = pd.concat([output_df.reset_index(drop=True), pd.DataFrame(principalComponents)], axis=1)
    n_components = principalComponents.shape[1]
    df_segm_pca_kmeans.columns.values[-n_components:] = ['Component ' + str(i + 1) for i in range(n_components)]
    df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_
    df_segm_pca_kmeans['Segment'] = df_segm_pca_kmeans['Segment K-means PCA'].map({0: 'first', 1: 'second'})
    for i in range(n_components):
        x_axis = df_segm_pca_kmeans['Component ' + str(i + 1)]
        for j in range(i + 1, n_components):
            y_axis = df_segm_pca_kmeans['Component ' + str(j + 1)]
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x_axis, y_axis, hue=df_segm_pca_kmeans['Segment'], palette=['g', 'r'])
            plt.title('Clusters by PCA Components')
            if Consts.SHOW_GRAPH:
                plt.show()


def plot_average_correlation_neighbors_vs_non_neighbors(lst1, lst2, labels=None, title=None, xlabel=None,
                                                        ylabel=None, colors=None, special_marker=None, cells_lst=None, arg1=None, arg2=None, save_fig=False, fig_name=None):
    f, ax = plt.subplots(figsize=(4, 4))
    color = iter(cm.rainbow(np.linspace(0, 1, len(labels))))

    cmap = None
    # if cells_lst:
    #     cmap = {}
    #     cell_color = mcolors.TABLEAU_COLORS.values()
    #     cell_color = list(cell_color)
    #     cell_color.extend(cell_color)
    #
    # for i in range(len(lst1)):
    #     c = next(color)
    #
    #     if cells_lst:
    #         k = cells_lst[i]
    #         if k in cmap.keys():
    #             c = cmap[k]
    #         else:
    #             c = list(cell_color)[int(float(k))-1]
    #             cmap[k] = c
    #
    #     marker = 'bo'
    #     if special_marker:
    #         marker = "*" if special_marker[i] == 'special' else '.'
    #
    #     if cmap:
    #         c = cmap[k]
    #
    #     plt.plot(float(lst1[i]), float(lst2[i]), marker, label=labels[i], c=c, alpha=0.5)
    #     # plt.plot(float(lst1[i]), float(lst2[i]), marker, label=k, c=c)

    ## TODO: this code is for the comparison of 5.3 vs 13.2
    if colors is None:
        colors = list(mcolors.TABLEAU_COLORS.values())[2:4]
    label1 = 'before inhibitor'
    label2 = 'after inhibitor'
    # label1 = '5.3'
    # label2 = '13.2'
    lst_x_1 = []
    lst_y_1 = []
    lst_x_2 = []
    lst_y_2 = []

    for label_index in range(len(labels)):
        if labels[label_index] == label1:
            lst_x_1.append(float(lst1[label_index]))
            lst_y_1.append(float(lst2[label_index]))
        else:
            lst_x_2.append(float(lst1[label_index]))
            lst_y_2.append(float(lst2[label_index]))

    # plt.rcParams.update({'font.size': 10})
    # plt.rcParams['font.family'] = 'Arial'

    plt.plot(lst_x_1, lst_y_1, 'bo', label=label1, c=colors[0], alpha=0.2)
    plt.plot(lst_x_2, lst_y_2, 'bo', label=label2, c=colors[1], alpha=0.2)

    plt.axvline(x=0, color="gainsboro", linestyle="--")
    plt.axhline(y=0, color="gainsboro", linestyle="--")

    plt.axis('square')
    # plt.setp(ax, xlim=(-1, 1), ylim=(-1, 1))
    plt.setp(ax, xlim=(-0.5, 1), ylim=(-0.5, 1))
    axline([ax.get_xlim()[0], ax.get_ylim()[0]], [ax.get_xlim()[1], ax.get_ylim()[1]], ls='--')

    # plt.axvline(x=arg1, color="gainsboro", linestyle="--")
    # plt.axhline(y=arg2, color="gainsboro", linestyle="--")
    plt.yticks(rotation=90)
    if title:
        plt.title(title)
    else:
        plt.title('Average Correlation', fontsize=15)
    if xlabel:
        plt.xlabel(xlabel, fontsize=12)
    else:
        plt.xlabel('Non-Adjacent pair correlation')
    if ylabel:
        plt.ylabel(ylabel, fontsize=12)
    else:
        plt.ylabel('Adjacent pair correlation')
    if labels is not None:
        # plt.legend(labels, bbox_to_anchor=(1, 1), title='cell')
        # cells_int_lst = [int(c) for c in set(cells_lst)]
        # cells_int_lst.sort()
        # plt.legend(cells_int_lst)
        # plt.legend(title='Video-cell')
        plt.legend()
    if save_fig:
        plt.savefig('../' + fig_name + '.svg', format="svg")
    if Consts.SHOW_GRAPH:
        plt.show()


def plot_pillar_pairs_correlation_frames_window(pairs_corr_dict, neighbor_pairs=True):
    pairs = []
    n = len(pairs_corr_dict)
    plt.clf()
    for pair, corrs in pairs_corr_dict.items():
        window_num = [n for n in range(1, len(corrs) + 1)]
        plt.plot(window_num, corrs)
        pairs.append(pair)
    plt.xlabel('Window')
    plt.ylabel('Correlation')
    if neighbor_pairs:
        plt.title('Window Correlation - neighbor pairs')
    else:
        plt.title('Window Correlation')
    plt.legend(pairs)

    if Consts.RESULT_FOLDER_PATH is not None:
        if neighbor_pairs:
            plt.savefig(Consts.RESULT_FOLDER_PATH + "/pillar_neighboring_top_" + str(
                n) + "_pairs_correlation_frames_window.png")
            plt.close()  # close the figure window
        else:
            plt.savefig(Consts.RESULT_FOLDER_PATH + "/pillar_top_" + str(n) + "_pairs_correlation_frames_window.png")
            plt.close()  # close the figure window

    if Consts.SHOW_GRAPH:
        plt.show()


def show_correlated_pairs_in_video(n=5, neighbor_pairs=True):
    pairs_corr_dict = get_top_pairs_corr_in_each_frames_window(n=n, neighbor_pairs=neighbor_pairs)
    all_images = get_images(get_images_path())

    fig = plt.figure()
    ax = fig.add_subplot()

    coords = []

    for pair in pairs_corr_dict.keys():
        coordinate_1 = eval(pair[0])
        coordinate_2 = eval(pair[1])
        coords.append([coordinate_1, coordinate_2])

    def animate(i):
        ax.clear()

        color = iter(cm.rainbow(np.linspace(0, 1, len(pairs_corr_dict.keys()))))

        for pair in pairs_corr_dict.keys():
            coordinate_1 = eval(pair[0])
            coordinate_2 = eval(pair[1])
            c = next(color)
            ax.plot([coordinate_1[1], coordinate_2[1]], [coordinate_1[0], coordinate_2[0]], c=c, linewidth=1)
        ax.legend(coords)

        ax.imshow(all_images[i % len(all_images)], cmap=plt.cm.gray)

    ani = animation.FuncAnimation(fig, animate, frames=len(all_images), interval=50)

    if Consts.RESULT_FOLDER_PATH is not None:
        writergif = animation.PillowWriter(fps=30)
        if neighbor_pairs:
            ani.save(Consts.RESULT_FOLDER_PATH + "/top_" + str(n) + "_neighboring_corr_animation.gif", dpi=300,
                     writer=writergif)
            plt.close()  # close the figure window
        else:
            ani.save(Consts.RESULT_FOLDER_PATH + "/top_" + str(n) + "_corr_animation.gif", dpi=300, writer=writergif)
            plt.close()  # close the figure window

    if Consts.SHOW_GRAPH:
        plt.show()


def show_correlated_pairs_in_last_image(n=5, neighbor_pairs=True):
    plt.clf()
    pairs_corr_dict = get_top_pairs_corr_in_each_frames_window(n=n, neighbor_pairs=neighbor_pairs)
    last_img = get_last_image()
    coords = []

    color = iter(cm.rainbow(np.linspace(0, 1, len(pairs_corr_dict.keys()))))

    for pair in pairs_corr_dict.keys():
        coordinate_1 = eval(pair[0])
        coordinate_2 = eval(pair[1])
        coords.append([coordinate_1, coordinate_2])
        c = next(color)
        plt.plot([coordinate_1[1], coordinate_2[1]], [coordinate_1[0], coordinate_2[0]], c=c, linewidth=1)
    plt.legend(coords)
    plt.imshow(last_img, cmap=plt.cm.gray)

    if Consts.RESULT_FOLDER_PATH is not None:
        if neighbor_pairs:
            plt.savefig(Consts.RESULT_FOLDER_PATH + "/top_" + str(n) + "_correlated_neighboring_pairs_last_image.png")
            plt.close()  # close the figure window
        else:
            plt.savefig(Consts.RESULT_FOLDER_PATH + "/top_" + str(n) + "_correlated_pairs_last_image.png")
            plt.close()  # close the figure window

    if Consts.SHOW_GRAPH:
        plt.show()


def plot_pillar_intensity_with_movement():
    centers_movements = get_alive_centers_movements_v2()
    pillars_intens = get_pillar_to_intensity_norm_by_inner_pillar_noise()
    for pillar, moves in centers_movements.items():
        pillar_id = min(pillars_intens.keys(), key=lambda point: math.hypot(point[1] - pillar[1], point[0] - pillar[0]))
        pillar_movment = []
        for move in moves:
            pillar_movment.append(move['distance'])

        pillar_intens = pillars_intens[pillar_id]
        norm_intens = (pillar_intens - np.min(pillar_intens)) / (np.max(pillar_intens) - np.min(pillar_intens))
        plt.plot(norm_intens, label='intensity')
        plt.plot(pillar_movment, label='movement')
        plt.title('pillar ' + str(pillar_id))
        plt.legend()

        # Consts.RESULT_FOLDER_PATH = "../multi_config_runner_results/13.2/06/movement_intensity_plots"
        # Path(Consts.RESULT_FOLDER_PATH).mkdir(parents=True, exist_ok=True)
        # plt.savefig(Consts.RESULT_FOLDER_PATH + "/movement_intensity_pillar_" + str(pillar_id) + ".png")
        # plt.close()  # close the figure window
        if Consts.SHOW_GRAPH:
            plt.show()


def plot_pillars_intensity_movement_correlations(pillars_intensity_movement_correlations_lst):
    plt.scatter()


def plot_pair_of_pillars_movement_corr_and_intensity_corr(movement_corr_df, intensity_corr_df, neighbors_only=False):
    pillars = movement_corr_df.columns
    neighbors_dict = get_alive_pillars_to_alive_neighbors()

    for p1 in pillars:
        for p2 in pillars:
            if p1 == p2:
                continue
            if neighbors_only:
                if eval(p1) not in neighbors_dict[eval(p2)]:
                    continue
            move_corr = movement_corr_df[p1][p2]
            intens_corr = intensity_corr_df[p1][p2]
            if eval(p1) in neighbors_dict[eval(p2)]:
                plt.scatter(intens_corr, move_corr, c='red', alpha=0.2)
            else:
                plt.scatter(intens_corr, move_corr, c='blue', alpha=0.2)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("Intensity correlation")
    plt.ylabel("Movement correlation")
    plt.title("Intensity and Movement Correlation of Pillar Pairs")

    if Consts.RESULT_FOLDER_PATH is not None:
        plt.savefig(Consts.RESULT_FOLDER_PATH + "/intensity_movement_pairs_correlation.png")
        plt.close()  # close the figure window
        print("saved intensity_movement_pairs_correlation.png")

    if Consts.SHOW_GRAPH:
        plt.show()


def plot_pillars_average_intensity_by_movement():
    avg_intens_by_distance = get_average_intensity_by_distance()
    f, ax = plt.subplots(figsize=(6, 6))
    zero = []
    not_zero = []
    for p, intens in avg_intens_by_distance.items():
        intens_dist_zero = intens['avg_intens_when_dist_zero']
        intens_dist_non_zero = intens['avg_intens_when_dist_non_zero']
        # plt.scatter(intens_dist_zero, intens_dist_non_zero)
        zero.append(intens_dist_zero)
        not_zero.append(intens_dist_non_zero)
    plt.scatter(zero, not_zero)

    zero_intens_lst = [d['avg_intens_when_dist_zero'] for d in list(avg_intens_by_distance.values())]
    non_zero_intens_lst = [d['avg_intens_when_dist_non_zero'] for d in list(avg_intens_by_distance.values())]
    min_axis_val = min(min(zero_intens_lst), min(non_zero_intens_lst))
    max_axis_val = max(max(zero_intens_lst), max(non_zero_intens_lst))
    plt.xlabel("Avg intensity when distance == 0")
    plt.ylabel("Avg intensity when distance != 0")
    plt.title("Pillars intensity by movement")
    plt.xlim(min_axis_val, max_axis_val)
    plt.ylim(min_axis_val, max_axis_val)
    axline([ax.get_xlim()[0], ax.get_ylim()[0]], [ax.get_xlim()[1], ax.get_ylim()[1]], ls='--')

    if Consts.RESULT_FOLDER_PATH is not None:
        plt.savefig(Consts.RESULT_FOLDER_PATH + "/Pillars intensity by movement.png")
        plt.close()  # close the figure window

    if Consts.SHOW_GRAPH:
        plt.show()


def plot_average_intensity_by_distance():
    avg_intens_to_dict = get_average_intensity_by_distance()
    f, ax = plt.subplots(figsize=(6, 6))
    all_distances = []
    [all_distances.extend(list(p_dists.keys())) for p_dists in list(avg_intens_to_dict.values())]
    unique_distances = sorted(list(set(all_distances)))

    dist_to_overall_avg_intens = {}
    for dist in unique_distances:
        avg_intens_of_dist = []
        for val in avg_intens_to_dict.values():
            if dist in val:
                avg_intens_of_dist.append(val[dist])
        dist_to_overall_avg_intens[dist] = np.mean(avg_intens_of_dist)

    plt.scatter(dist_to_overall_avg_intens.keys(), dist_to_overall_avg_intens.values())
    plt.xlabel("Distance")
    plt.ylabel("Avg intensity")
    plt.title("Average Intensity By Distance")

    if Consts.RESULT_FOLDER_PATH is not None:
        plt.savefig(Consts.RESULT_FOLDER_PATH + "/Average intensity by distance.png")
        plt.close()  # close the figure window

    if Consts.SHOW_GRAPH:
        plt.show()


def show_peripheral_pillars_in_video(frame_to_peripheral_center_dict):
    frame_to_peripheral_center_dict = frame_to_peripheral_center_dict
    all_images = get_images(get_images_path())

    fig = plt.figure()
    ax = fig.add_subplot()

    def animate(i):
        ax.clear()

        curr_frame = list(frame_to_peripheral_center_dict.keys())[i]
        periph_pillars = frame_to_peripheral_center_dict[curr_frame]["peripherals"]

        y = [p[0] for p in periph_pillars]
        x = [p[1] for p in periph_pillars]
        scatter_size = [3 for center in periph_pillars]

        ax.scatter(x, y, s=scatter_size, color='blue')

        central_pillars = frame_to_peripheral_center_dict[curr_frame]["centrals"]

        y = [p[0] for p in central_pillars]
        x = [p[1] for p in central_pillars]
        scatter_size = [3 for center in central_pillars]

        ax.scatter(x, y, s=scatter_size, color='red')

        ax.imshow(all_images[i % len(all_images)], cmap=plt.cm.gray)

    ani = animation.FuncAnimation(fig, animate, frames=len(all_images), interval=100)
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    # writer = animation.FFMpegWriter(fps=10)

    writer = animation.PillowWriter(fps=10000)

    # if Consts.RESULT_FOLDER_PATH is not None:
    #     ani.save(Consts.RESULT_FOLDER_PATH + "/peripheral_pillars.gif", dpi=300, writer=writer)
    #     plt.close()  # close the figure window

    if Consts.SHOW_GRAPH:
        plt.show()


def plot_average_movement_signal_sync_peripheral_vs_central(central_lst, peripheral_lst, labels=None, title=None):
    f, ax = plt.subplots(figsize=(6, 6))
    color = iter(cm.rainbow(np.linspace(0, 1, len(labels))))
    for i in range(len(peripheral_lst)):
        c = next(color)
        plt.plot(central_lst[i], peripheral_lst[i], 'bo', label=labels[i], c=c)

    # plt.axis('square')
    plt.setp(ax, xlim=(-1, 1), ylim=(-1, 1))
    axline([ax.get_xlim()[0], ax.get_ylim()[0]], [ax.get_xlim()[1], ax.get_ylim()[1]], ls='--')
    if title:
        plt.title('Average Movement-Signal Correlation' + ' ' + title)
    else:
        plt.title('Average Correlation')
    plt.ylabel('Peripheral pillars correlation')
    plt.xlabel('Central pillars correlation')
    if labels is not None:
        plt.legend(labels)
    if Consts.SHOW_GRAPH:
        plt.show()


def plot_avg_correlation_spreading_level(exp_type_name, avg_corr, spreading_level):
    f, ax = plt.subplots()
    colors = []
    labels = []
    for i in range(len(exp_type_name)):
        c = 'red' if spreading_level[i] == 'high' else 'blue'
        colors.append(c)
        labels.append(spreading_level[i])
        ax.scatter(exp_type_name[i], float(avg_corr[i][0]), c=c, label=spreading_level[i])

    ax.legend(['high', 'low'])
    plt.xlabel("Experiment")
    plt.ylabel("Avg Correlation")
    plt.title("Experiment Avg Correlation in Spreading Level")

    plt.show()


def plot_experiment_features_heatmap(exp_lst, features_lst, matrix_values):
    matrix_df = pd.DataFrame(matrix_values, columns=features_lst, index=exp_lst)
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)

    sns.heatmap(matrix_df,
                annot=True,
                # vmin=-1,
                # vmax=1,
                annot_kws={"fontsize": 9})

    plt.xticks(rotation=15, fontsize=8)
    plt.yticks(rotation=45)

    plt.xlabel('Feature', fontsize=18)
    plt.ylabel('Experiment', fontsize=18)

    plt.show()


def plot_nbrs_correlations_heatmap(correlations_df, neighbors_dict):
    # matrix_df = pd.DataFrame(None, columns=correlations_df.columns, index=correlations_df.columns, dtype='float64')
    labels = correlations_df.applymap(lambda v: '')

    for p, nbrs in neighbors_dict.items():
        for n in nbrs:
            labels.loc[str(p), str(n)] = round(correlations_df.loc[str(p), str(n)], 2)

    sns.heatmap(correlations_df,
                annot=labels,
                mask=correlations_df.isnull(),
                vmin=-1,
                vmax=1,
                annot_kws={"fontsize": 6}, fmt='')

    plt.xticks(rotation=25)
    plt.xlabel('Pillar', fontsize=15)
    plt.ylabel('Pillar', fontsize=15)

    plt.show()


def show_nbrs_distance_graph(nbrs_dict, pillar2middle_img_steps):
    my_G = nx.Graph()

    nodes_loc = list(nbrs_dict)

    node_loc2index = {}
    for i in range(len(nodes_loc)):
        node_loc2index[nodes_loc[i]] = i
        my_G.add_node(i)

    for n1, nbrs in nbrs_dict.items():
        for n2 in nbrs:
            my_G.add_edge(node_loc2index[n1], node_loc2index[n2])
            try:
                my_G[node_loc2index[n1]][node_loc2index[n2]]['weight'] = find_vertex_distance_from_center(n1, n2,
                                                                                                          pillar2middle_img_steps)
            except:
                x = 1

    edges, weights = zip(*nx.get_edge_attributes(my_G, 'weight').items())
    cmap = plt.cm.hot

    max_value = max(pillar2middle_img_steps.values())
    min_value = min(pillar2middle_img_steps.values())

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_value, vmax=max_value))
    nodes_loc_y_inverse = [(loc[1], loc[0]) for loc in nodes_loc]
    plt.imshow(get_last_image(), cmap=plt.cm.gray)

    nx.draw(my_G, nodes_loc_y_inverse, with_labels=False, node_color='black', edgelist=edges, edge_color=weights,
            width=3.0,
            edge_cmap=cmap, node_size=15,
            vmin=min_value, vmax=max_value, edge_vmin=min_value, edge_vmax=max_value)
    plt.colorbar(sm)
    plt.show()


def plot_correlation_by_distance_from_center_cell(list_of_dist_to_corr_dict, labels):
    f, ax = plt.subplots(figsize=(6, 6))
    color = iter(cm.rainbow(np.linspace(0, 1, len(labels))))
    for i in range(len(labels)):
        c = next(color)
        dist_to_corr = list_of_dist_to_corr_dict[i]
        plt.plot(list(dist_to_corr.keys()), list(dist_to_corr.values()), label=labels[i], color=c, linestyle='--',
                 marker='o')

    plt.title('Local Correlation by Distance Level From the Cell Center', fontsize=14)
    plt.ylabel('Correlation', fontsize=12)
    plt.xlabel('Distance from cell center', fontsize=12)
    if labels is not None:
        plt.legend(labels)
    if Consts.SHOW_GRAPH:
        plt.show()


def print_tagged_centers():
    plt.imshow(get_images(get_images_path())[-1], cmap=plt.cm.gray)
    y = [center[0] for center in Consts.tagged_centers]
    x = [center[1] for center in Consts.tagged_centers]
    scatter_size = [3 for center in Consts.tagged_centers]

    plt.scatter(x, y, s=scatter_size)
    plt.show()


def plot_node_strengths_distribution(node_strengths):
    # Plot the node strength distribution
    plt.hist(list(node_strengths.values()), bins=10, edgecolor='black')
    plt.xlabel('Node Strength')
    plt.ylabel('Frequency')
    plt.title('Node Strength Distribution')

    # Show the plot
    plt.show()


def plot_avg_similarity_by_nbrhood_degree(level_to_similarities_dict):
    levels = list(level_to_similarities_dict.keys())
    level_avg_similarity = [np.mean(v) for v in level_to_similarities_dict.values()]
    plt.plot(levels, level_avg_similarity, linestyle='dashed', marker='o')
    plt.xlabel('Neighborhood Degree')
    plt.ylabel('Avg Similarity')
    plt.title('Average Similarity Of Each Neighborhood Degree')
    plt.show()


def plot_avg_correlation_by_nbrhood_degree(level_to_corrs_dict, save_fig=False):
    levels = list(level_to_corrs_dict.keys())
    level_avg_similarity = [np.mean(v) for v in level_to_corrs_dict.values()]

    plt.rcParams.update({'font.size': 10})
    plt.rcParams['font.family'] = 'Arial'

    plt.plot(levels, level_avg_similarity, linestyle='dashed', marker='o', color="tab:orange")
    plt.bar(levels, level_avg_similarity)
    # sns.kdeplot(level_avg_similarity, color="red", linestyle="--")
    plt.xlabel('Topological Distance')
    plt.ylabel('Avg Correlation')
    plt.yticks(rotation=90)
    # plt.title('Average Correlation by Topological Distance')
    if save_fig:
        plt.savefig('../top_dist_fig_1f.svg', format="svg")
    plt.show()


def plot_correlation_by_topological_distance_histogram(level_to_corrs_dict):
    distances = list(level_to_corrs_dict.keys())
    level_avg_similarity = [np.mean(v) for v in level_to_corrs_dict.values()]

    # for l, corrs in level_to_corrs_dict.items():
    #     sns.histplot(data=corrs, label=l, kde=True, alpha=0.3, stat='density')
    # plt.xlabel("Correlations")
    # plt.title('Correlation by Topological Distance Histogram')
    # plt.legend(title="Topological Distance")
    # plt.show()


def plot_avg_cluster_time_series(segments, matrix_3d, pillars_id_matrix_2d, save_clusters_fig=None):
    p_to_intens = get_pillar_to_intensity_norm_by_inner_pillar_noise()
    for i in range(matrix_3d.shape[0]):
        plt.imshow(matrix_3d[i], cmap='gray')  # Use 'cmap' appropriate for your data
        plt.title(f"Frame {i}")
        plt.pause(0.5)  # Pause time between frames in seconds
        plt.clf()
    plt.show()
    unique_labels = np.unique(segments)
    unique_labels.sort()
    avg_segment_intnes = []
    for l in unique_labels:
        if l != 0:
            intens = []
            pillars_in_labels = pillars_id_matrix_2d[segments == l]
            for p in pillars_in_labels:
                intens.append(p_to_intens[p])
            average_intensity_in_segment = [sum(values) / len(values) for values in zip(*intens)]
            avg_segment_intnes.append(average_intensity_in_segment)
    for index, intensity in enumerate(avg_segment_intnes):
        plt.plot(intensity, label=f'Segment {index + 1}')
    plt.xlabel('Index')
    plt.ylabel('Avg Intensity')
    plt.legend()
    if save_clusters_fig is not None:
        file_name = "/Superpixel_Segmentation_avg_intensity_frames_" + str(save_clusters_fig) + ".png"
        plt.savefig(Consts.RESULT_FOLDER_PATH + file_name)
        plt.close()
    # plt.show()


def show_superpixel_intensity_gif(n_segments):
    segments, matrix_3d, pillars_id_matrix_2d = superpixel_segmentation(n_segments=n_segments, shuffle_ts=False)
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots()
    def update(i):
        ax.imshow(matrix_3d[i], cmap='gray')  # Update with frame i
        ax.set_title(f"Frame {i}")
    ani = FuncAnimation(fig, update, frames=matrix_3d.shape[0], interval=300)
    video_filename = Consts.RESULT_FOLDER_PATH + "/pillars_as_pixel_intensity_vid.gif"
    ani.save(video_filename, writer='ffmpeg')


def similarities_above_avg_graph(G_sims, pillars_pair_to_sim_above_avg):
    pos = nx.get_node_attributes(G_sims, 'pos')
    nodes_loc_y_inverse = {k: (v[1], v[0]) for k, v in pos.items()}

    edges_list = []
    for pair, sim in pillars_pair_to_sim_above_avg.items():
        node1_idx = next((i for i, p in G_sims.nodes(data=True) if p.get('pos') == pair[0]), None)
        node2_idx = next((i for i, p in G_sims.nodes(data=True) if p.get('pos') == pair[1]), None)
        edges_list.append((node1_idx, node2_idx))

    img = get_last_image_whiten(build_image=Consts.build_image)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    nx.draw(G_sims, nodes_loc_y_inverse, edge_color='black', width=3.0, node_size=50)
    nx.draw_networkx_edges(G_sims, nodes_loc_y_inverse, edgelist=edges_list, edge_color='red', width=3.0)

    plt.show()


def plot_core_periphery_pillars_in_graph(G, core_list, periphery_list):
    core_pos_to_idx = [i for n in core_list for i, p in G.nodes(data=True) if p.get('pos') == n]
    periphery_pos_to_idx = [i for n in periphery_list for i, p in G.nodes(data=True) if p.get('pos') == n]
    # for n in core_list:
    #     for i, p in G.nodes(data=True):
    #         if p.get('pos')==n:
    #             core_pos_to_idx.append(i)

    pos = nx.get_node_attributes(G, 'pos')
    nodes_loc_y_inverse = {k: (v[1], v[0]) for k, v in pos.items()}
    img = get_last_image_whiten(build_image=Consts.build_image)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    nx.draw_networkx_nodes(G, nodes_loc_y_inverse, nodelist=core_pos_to_idx, node_color='g', label='C')
    nx.draw_networkx_nodes(G, nodes_loc_y_inverse, nodelist=periphery_pos_to_idx, node_color='r', label='P')
    nx.draw_networkx_edges(G, nodes_loc_y_inverse)
    plt.show()


def plot_distribution_similarity_core_vs_periphery(core_sim_lst, periphery_sim_lst, total_avg_similarity):
    sns.histplot(core_sim_lst, label="core similarity", kde=True, alpha=0.3, stat='density')
    sns.histplot(periphery_sim_lst, label="periphery similarity", kde=True, alpha=0.3, stat='density')
    plt.axvline(total_avg_similarity, linestyle='--', color='grey', label='total_avg_similarity')
    plt.axvline(np.mean(core_sim_lst), linestyle='--', color='blue', label='core_avg_similarity')
    plt.axvline(np.mean(periphery_sim_lst), linestyle='--', color='orange', label='periphery_avg_similarity')
    plt.title("Similarity Distribution Core vs. Periphery")
    plt.xlabel('Similarity')
    print("Total average similarity:", "%.3f" % total_avg_similarity)
    print("Average core similarity:", "%.3f" % np.mean(core_sim_lst))
    print("Average periphery similarity:", "%.3f" % np.mean(periphery_sim_lst))
    t_stat, p_value = ttest_ind(core_sim_lst, periphery_sim_lst)
    print("P-Value: ", "%.5f" % p_value)
    plt.legend()
    plt.show()


def plot_distribution_similarity_of_exp_nbrs_vs_non_nbrs(nbrs_sim_lst, non_nbrs_sim_lst):
    if Consts.SHOW_GRAPH:
        sns.histplot(nbrs_sim_lst, label="Neighbors similarity", kde=True, alpha=0.3, stat='density')
        sns.histplot(non_nbrs_sim_lst, label="Non-neighbors similarity", kde=True, alpha=0.3, stat='density')
        plt.title("Distribution of Similarity Strength of Neighbors vs. Non-Neighbors")
        plt.xlabel('Similarity')
        plt.legend()
        plt.show()
    print("Average neighbors similarity:", "%.3f" % np.mean(nbrs_sim_lst))
    print("Average non-neighbors similarity:", "%.3f" % np.mean(non_nbrs_sim_lst))
    t_stat, p_value = ttest_ind(nbrs_sim_lst, non_nbrs_sim_lst)
    print("###### t-test #####")
    print("T-statistic value: ", "%.2f" % t_stat)
    print("P-Value: ", p_value)
    return p_value


def plot_significance_bar(type_1_significant, type_1_total, type_2_significant, type_2_total, type_labels_lst):
    type_1_insignificant = type_1_total - type_1_significant
    type_2_insignificant = type_2_total - type_2_significant
    significant_counts = [type_1_significant, type_2_significant]
    insignificant_counts = [type_1_insignificant, type_2_insignificant]
    type_1_percentage = (type_1_significant / type_1_total) * 100
    type_2_percentage = (type_2_significant / type_2_total) * 100

    x = range(len(type_labels_lst))
    fig, ax = plt.subplots()

    ax.bar(x[0], significant_counts[0], width=0.4, label='5.3 Significant', color='tab:green')
    # ax.bar(x[0], insignificant_counts[0], width=0.4, bottom=significant_counts[0], label='5.3 Insignificant',
    #        color='lightgreen')
    ax.bar(x[1], significant_counts[1], width=0.4, label='13.2 Significant', color='tab:red')
    # ax.bar(x[1], insignificant_counts[1], width=0.4, bottom=significant_counts[1], label='13.2 Insignificant', color='pink')

    # ax.bar(x, significant_counts, width=0.4, label='Significant', color='tab:blue')b
    ax.bar(x, insignificant_counts, width=0.4, label='Insignificant', bottom=significant_counts, color='lightgrey')
    ax.set_ylabel('Number of Experiments')
    ax.set_title('Significant Similarity Neighbors vs. Non-Neighbors')
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels_lst)
    ax.text(x[0], significant_counts[0], f'{type_1_percentage:.1f}%', ha='center', va='bottom', color='black')
    ax.text(x[1], significant_counts[1], f'{type_2_percentage:.1f}%', ha='center', va='bottom', color='black')
    # ax.legend()
    plt.show()


def histogram_for_53_exps_time_distribution():
    map_exp_to_minuts = {'20230320-02': 60.6,
                         '20230320-03': 60.3,
                         '20230320-04': 60.6,
                         '20230320-05': 60.3,
                         '20230320-06': 60.6,
                         '20230323-01': 21.3,
                         '20230323-03': 48.4,
                         '20230323-04': 60.9,
                         '20230323-05': 60,
                         '20230323-06': 60.6,
                         '20230323-07': 120.6,
                         '20230323-08': 60.4,
                         '20230323-09': 60.6,
                         '20230323-10': 60.6,
                         }
    exp_minuts = [60.6,60.6,60.6,60.6,60.6,
                  60.3,60.3,60.3,60.3,60.3,
                  60.6,60.6,60.6,60.6,60.6,
                  60.3,60.3,60.3,60.3,
                  60.6,60.6,60.6,60.6,
                  21.3,21.3,21.3,21.3,21.3,21.3,21.3,21.3,
                  48.4,48.4,48.4,48.4,48.4,48.4,48.4,48.4,
                  60.9,60.9,60.9,60.9,60.9,60.9,60.9,60.9,
                  60,60,60,60,60,60,60,60,
                  60.6,60.6,60.6,60.6,60.6,60.6,
                  120.6,120.6,120.6,120.6,120.6,
                  60.4,60.4,60.4,60.4,60.4,60.4,
                  60.6,60.6,60.6,60.6,60.6,
                  60.6,60.6,60.6,60.6]
    sns.distplot(exp_minuts, kde=True, bins=3)
    plt.xlabel("Time (minutes)")
    plt.show()




