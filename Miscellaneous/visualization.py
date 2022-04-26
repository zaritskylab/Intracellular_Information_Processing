import numpy as np
from Miscellaneous.pillars_utils import *


_last_image_path = LAST_IMG_VIDEO_06
_normalized = False

def show_last_image_masked(mask_path=PATH_MASKS_VIDEO_06_15_35):
    """
    show the mask on the video's last image
    :param mask_path:
    :return:
    """
    last_img = get_last_image(_last_image_path)
    plt.imshow(last_img, cmap=plt.cm.gray)
    plt.show()

    with open(mask_path, 'rb') as f:
        pillars_mask = np.load(f)
        pillars_mask = 255 - pillars_mask
        mx = ma.masked_array(last_img, pillars_mask)
        plt.imshow(mx, cmap=plt.cm.gray)
        # add the centers location on the image
        # centers = find_centers()
        # for center in centers:
        #     s = '(' + str(center[0]) + ',' + str(center[1]) + ')'
        #     plt.text(center[VIDEO_06_LENGTH], center[0], s=s, fontsize=7, color='red')

        plt.show()


def neighbors_correlation_histogram(correlations_df, neighbors_dict, symmetric_corr=False):
    """
    Display histogram plot of the correlations between the neighbors
    :param correlations_df:
    :param neighbors_dict:
    :param symmetric_corr:
    :return:
    """
    sym_corr = set()
    asym_corr = []
    for pillar, nbrs in neighbors_dict.items():
        for nbr in nbrs:
            if symmetric_corr:
                sym_corr.add(correlations_df[str(pillar)][str(nbr)])
            else:
                asym_corr.append(correlations_df[str(pillar)][str(nbr)])
    if symmetric_corr:
        corr = np.array(list(sym_corr))
    else:
        corr = asym_corr
    mean_corr = np.mean(corr)
    ax = sns.histplot(data=corr, kde=True)
    plt.xlabel("Correlation")
    plt.show()
    # return mean_corr


def indirect_alive_neighbors_correlation_plot(pillar_location, only_alive=True):
    """
    Plotting the correlation of a pillar with its all indirected neighbors
    :param pillar_location:
    :param only_alive:
    :return:
    """
    my_G = nx.Graph()
    nodes_loc = find_centers_with_logic()
    node_loc2index = {}
    for i in range(len(nodes_loc)):
        node_loc2index[nodes_loc[i]] = i
        my_G.add_node(i)

    if only_alive:
        pillars = get_alive_pillars_to_intensities()
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
    frame2pillars = get_frame_to_alive_pillars()
    nodes_index2size = [10] * len(nodes_loc)
    for node in nodes_loc:
        for i in range(len(frame2pillars)):
            if node in frame2pillars[i + 1]:
                nodes_index2size[node_loc2index[node]] = len(frame2pillars) - ((i // 13) * 13)
                break
    nodes_loc_y_inverse = [(loc[1], 1000 - loc[0]) for loc in nodes_loc]
    nx.draw(my_G, nodes_loc_y_inverse, with_labels=True, node_color='gray', edgelist=edges, edge_color=weights,
            width=3.0,
            edge_cmap=cmap,
            node_size=nodes_index2size)
    plt.colorbar(sm)
    plt.show()


def correlation_plot(only_alive=True, neighbors_str='all', alive_correlation_type='all'):
    """
    Plotting graph of correlation between neighboring pillars
    Each point represent pillar in its exact position in the image, and the size of each point represent how many
    time frames the pillar was living (the larger the pillar, the sooner he started to live)
    :param only_alive:
    :param neighbors_str:
    :param alive_correlation_type:
    :return:
    """
    my_G = nx.Graph()
    last_img = get_last_image(_last_image_path)
    alive_centers = get_alive_centers(last_img)
    nodes_loc = generate_centers_from_alive_centers(alive_centers, len(last_img))
    if neighbors_str == 'alive2back':
        neighbors = get_alive_pillars_in_edges_to_l1_neighbors()[0]
    elif neighbors_str == 'back2back':
        neighbors = get_background_level_1_to_level_2()
    elif neighbors_str == 'random':
        neighbors = get_random_neighbors()
    else:
        neighbors = get_pillar_to_neighbors()

    node_loc2index = {}
    for i in range(len(nodes_loc)):
        node_loc2index[nodes_loc[i]] = i
        my_G.add_node(i)

    if alive_correlation_type == 'all':
        alive_pillars_correlation = get_alive_pillars_correlation()
    elif alive_correlation_type == 'symmetric':
        alive_pillars_correlation = alive_pillars_symmetric_correlation()
    elif alive_correlation_type == 'asymmetric':
        alive_pillars_correlation = alive_pillars_asymmetric_correlation()
        my_G = my_G.to_directed()
    all_pillars_corr = get_all_pillars_correlation()

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
    cmap = plt.cm.seismic
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    frame2pillars = get_frame_to_alive_pillars()
    nodes_index2size = [10] * len(nodes_loc)
    for node in nodes_loc:
        for i in range(len(frame2pillars)):
            if node in frame2pillars[i + 1]:
                nodes_index2size[node_loc2index[node]] = len(frame2pillars) - ((i // 13) * 13)
                break
    nodes_loc_y_inverse = [(loc[1], 1000 - loc[0]) for loc in nodes_loc]

    nx.draw(my_G, nodes_loc_y_inverse, with_labels=True, node_color='gray', edgelist=edges, edge_color=weights,
            width=3.0,
            edge_cmap=cmap,
            node_size=nodes_index2size)
    plt.colorbar(sm)
    plt.show()
    x = 1


def build_gc_directed_graph(gc_df, only_alive=True):
    """
    Plotting a directed graph where an arrow represent that the pillar was "granger cause" the other pillar
    :param gc_df: dataframe with granger causality significance values
    :param only_alive:
    :return:
    """
    my_G = nx.Graph().to_directed()
    nodes_loc = find_centers_with_logic()
    # neighbors1, neighbors2 = get_pillar_to_neighbors()
    node_loc2index = {}
    for i in range(len(nodes_loc)):
        node_loc2index[str(nodes_loc[i])] = i
        my_G.add_node(i)
    # alive_pillars_correlation = get_alive_pillars_correlation()
    alive_pillars_correlation = alive_pillars_symmetric_correlation()
    all_pillars_corr = get_all_pillars_correlation()
    neighbors = get_pillar_to_neighbors()

    if only_alive:
        correlation = alive_pillars_correlation
    else:
        correlation = all_pillars_corr

    for col in gc_df.keys():
        for row, _ in gc_df.iterrows():
            if gc_df[col][row] < 0.05 and eval(row) in neighbors[eval(col)]:
                my_G.add_edge(node_loc2index[col], node_loc2index[row])
                try:
                    my_G[node_loc2index[col]][node_loc2index[row]]['weight'] = correlation[col][row]
                except:
                    x = 1

    cmap = plt.cm.seismic
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    frame2pillars = get_frame_to_alive_pillars()
    nodes_index2size = [10] * len(nodes_loc)
    for node in nodes_loc:
        for i in range(len(frame2pillars)):
            if node in frame2pillars[i + 1]:
                nodes_index2size[node_loc2index[str(node)]] = len(frame2pillars) - ((i // 13) * 13)
                break
    nodes_loc_y_inverse = [(loc[1], 1000 - loc[0]) for loc in nodes_loc]

    edges, weights = zip(*nx.get_edge_attributes(my_G, 'weight').items())
    # edges = list(filter(lambda x: x[0] == 52, edges))
    nx.draw(my_G, nodes_loc_y_inverse, with_labels=True, node_color='gray', edgelist=edges, edge_color="tab:red",
            width=3.0,
            node_size=nodes_index2size)
    # plt.colorbar(sm)
    plt.show()
    x = 1


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
    ax = sns.histplot(data=corr_array, kde=True)
    plt.xlabel("Correlation")
    plt.show()
    # return mean_corr


def plot_pillar_time_series():
    """
    Plotting a time series graph of the pillar intensity over time
    :return:
    """
    if _normalized:
        pillar2intens = normalized_intensities_by_mean_background_intensity()
    else:
        pillar2intens = get_pillar_to_intensities(get_images_path())

    intensities_1 = pillar2intens[(588, 669)]
    intensities_2 = pillar2intens[(472, 603)]
    # intensities_3 = pillar2intens[(94, 172)]
    x = [i * 19.87 for i in range(len(intensities_1))]
    intensities_1 = [i * 0.0519938 for i in intensities_1]
    intensities_2 = [i * 0.0519938 for i in intensities_2]
    # intensities_3 = [i * 0.0519938 for i in intensities_3]
    plt.plot(x, intensities_1, label='(588, 669)')
    plt.plot(x, intensities_2, label='(472, 603)')
    # plt.plot(x, intensities_3, label='(94, 172)')

    # plt.plot(x, intensities)
    plt.xlabel('Time (sec)')
    plt.ylabel('Intensity (micron)')
    # plt.title('Pillar ' + str(pillar_loc))
    plt.legend()
    plt.show()
