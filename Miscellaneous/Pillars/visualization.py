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

from Pillars.pillar_neighbors import *


def show_last_image_masked(mask_path=None, pillars_mask=None):
    """
    show the mask on the video's last image
    :param mask_path:
    :return:
    """
    last_img = get_last_image()
    # # TODO: delete
    # imgs = get_images(get_images_path())
    # last_img = imgs[0]
    # if len(last_img.shape) == 3:
    #     last_img = last_img[-1]

    plt.imshow(last_img, cmap=plt.cm.gray)
    plt.show()

    if mask_path is not None:
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


def indirect_alive_neighbors_correlation_plot(pillar_location, only_alive=True):
    """
    Plotting the correlation of a pillar with its all indirected neighbors
    :param pillar_location:
    :param only_alive:
    :return:
    """

    my_G = nx.Graph()
    nodes_loc = find_all_centers_with_logic()
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
    pillar2mask = get_last_img_mask_for_each_pillar()
    frame2pillars = get_frame_to_alive_pillars_by_same_mask(pillar2mask)
    nodes_index2size = [10] * len(nodes_loc)
    for node in nodes_loc:
        for i in range(len(frame2pillars)):
            if node in frame2pillars[i + 1]:
                nodes_index2size[node_loc2index[node]] = len(frame2pillars) - ((i // 13) * 13)
                break
    nodes_loc_y_inverse = [(loc[1], loc[0]) for loc in nodes_loc]
    nx.draw(my_G, nodes_loc_y_inverse, with_labels=True, node_color='gray', edgelist=edges, edge_color=weights,
            width=3.0,
            edge_cmap=cmap,
            node_size=nodes_index2size)
    plt.colorbar(sm)
    plt.show()


def correlation_plot(only_alive=True,
                     neighbors_str='all',
                     alive_correlation_type='all'):
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
    last_img = get_last_image()
    alive_centers = get_alive_centers()
    nodes_loc = generate_centers_from_alive_centers(alive_centers, len(last_img))
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
    cmap = plt.cm.seismic
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    pillar2mask = get_last_img_mask_for_each_pillar()
    frame2pillars = get_frame_to_alive_pillars_by_same_mask(pillar2mask)
    nodes_index2size = [10] * len(nodes_loc)
    for node in nodes_loc:
        for i in range(len(frame2pillars)):
            if node in frame2pillars[i + 1]:
                nodes_index2size[node_loc2index[node]] = len(frame2pillars) - ((i // 13) * 13)
                break
    nodes_loc_y_inverse = [(loc[1], loc[0]) for loc in nodes_loc]

    nx.draw(my_G, nodes_loc_y_inverse, with_labels=True, node_color='gray', edgelist=edges, edge_color=weights,
            width=3.0,
            edge_cmap=cmap,
            node_size=nodes_index2size,
            vmin=-1, vmax=1, edge_vmin=-1, edge_vmax=1)
    plt.colorbar(sm)
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
    my_G = nx.Graph().to_directed()
    nodes_loc = find_all_centers_with_logic()
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

    if draw:
        cmap = plt.cm.seismic
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
        # plt.colorbar(sm)

        pillar2mask = get_last_img_mask_for_each_pillar()
        frame2pillars = get_frame_to_alive_pillars_by_same_mask(pillar2mask)
        nodes_index2size = [10] * len(nodes_loc)
        for node in nodes_loc:
            for i in range(len(frame2pillars)):
                if node in frame2pillars[i + 1]:
                    nodes_index2size[node_loc2index[str(node)]] = len(frame2pillars) - ((i // 13) * 13)
                    break
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
        # nx.draw(my_G, nodes_loc_y_inverse, with_labels=True, node_color='gray', edgelist=edges, edge_color="tab:red",
        #         width=3.0,
        #         node_size=nodes_index2size)
        nx.draw_networkx_labels(my_G, nodes_loc_y_inverse, font_color="whitesmoke", font_size=8)

        # plt.scatter(get_image_size()[0]/2, get_image_size()[1]/2, s=250, c="red")

        # ax.plot()
        plt.show()
        x = 1
    return my_G, p_vals_lst


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
    nodes_loc = find_all_centers_with_logic()
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

        pillar2mask = get_last_img_mask_for_each_pillar()
        frame2alive_pillars = get_frame_to_alive_pillars_by_same_mask(pillar2mask)
        nodes_index2size = [10] * len(nodes_loc)
        for node in nodes_loc:
            for frame in range(len(frame2alive_pillars)):
                if node in frame2alive_pillars[frame + 1]:
                    nodes_index2size[node_loc2index[str(node)]] = len(frame2alive_pillars) - ((frame // 13) * 13)
                    break
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
        # nx.draw(my_G, nodes_loc_y_inverse, with_labels=True, node_color='gray', edgelist=edges, edge_color="tab:red",
        #         width=3.0,
        #         node_size=nodes_index2size)
        nx.draw_networkx_labels(my_G, nodes_loc_y_inverse, font_color="whitesmoke")

        # ax.plot()
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
    plt.show()
    print("mean correlations: " + str(mean_corr))
    return mean_corr


def neighbors_correlation_histogram(correlations_df, neighbors_dict, original_neighbors=True):
    """
    Display histogram plot of the correlations between the neighbors
    :param correlations_df:
    :param neighbors_dict:
    :param symmetric_corr:
    :return:
    """
    sym_corr = set()
    for pillar, nbrs in neighbors_dict.items():
        for nbr in nbrs:
            sym_corr.add(correlations_df[str(pillar)][str(nbr)])
    corr = np.array(list(sym_corr))
    mean_corr = np.mean(corr)
    sns.histplot(data=corr, kde=True)
    if original_neighbors:
        plt.title("Correlation of Original Neighbors")
    else:
        plt.title("Correlation of Randomize Neighbors")
    plt.xlabel("Correlation")
    plt.show()
    print("mean correlation between neighbors: " + str(mean_corr))
    return mean_corr


def plot_pillar_time_series():
    """
    Plotting a time series graph of the pillar intensity over time
    :return:
    """
    if Consts.normalized:
        pillar2intens = normalized_intensities_by_mean_background_intensity()
    else:
        pillar2intens = get_pillar_to_intensities(get_images_path())

    # for p in pillar2intens.keys():

    intensities_1 = pillar2intens[(524, 523)]
    intensities_2 = pillar2intens[(454, 493)]
    intensities_3 = pillar2intens[(463, 569)]
    x = [i * 19.94 for i in range(len(intensities_1))]
    intensities_1 = [i * 0.0519938 for i in intensities_1]
    intensities_2 = [i * 0.0519938 for i in intensities_2]
    intensities_3 = [i * 0.0519938 for i in intensities_3]
    plt.plot(x, intensities_1, label='(524, 523)')
    plt.plot(x, intensities_2, label='(454, 493)')
    plt.plot(x, intensities_3, label='(463, 569)')

    # plt.plot(x, intensities)
    plt.xlabel('Time (sec)')
    plt.ylabel('Intensity (micron)')
    # plt.title('Pillar ' + str(pillar_loc))
    plt.legend()
    plt.show()


def compare_neighbors_corr_histogram_random_vs_real(random_amount):
    """
    Show on same plot the mean correlation of the real neighbors and
    :param random_amount:
    :return:
    """
    mean_original_nbrs = neighbors_correlation_histogram(get_alive_pillars_symmetric_correlation(),
                                                         get_alive_pillars_to_alive_neighbors(),
                                                         original_neighbors=True)
    means = []
    rand = []
    for i in range(random_amount):
        mean_random_nbrs = neighbors_correlation_histogram(get_alive_pillars_symmetric_correlation(),
                                                           get_random_neighbors(), original_neighbors=False)
        means.append(mean_random_nbrs)
        rand.append('random' + str(i + 1))
    print("Random nbrs mean correlation: " + str(np.mean(means)))
    means.append(mean_original_nbrs)
    rand.append('original')
    fig, ax = plt.subplots()
    ax.scatter(rand, means)
    plt.ylabel('Average Correlation')
    plt.xticks(rotation=45)
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
    plt.show()
    print("number of neighbors with no edges: " + str(len(no_edge)))
    print("average of neighbors with no edges: " + str(np.mean(no_edge)))
    sns.histplot(data=one_sided_edge, kde=True)
    plt.xlabel("Correlations")
    plt.title('Correlation of 1 sided edges between neighbors')
    plt.show()
    print("number of neighbors with 1 edge: " + str(len(one_sided_edge)))
    print("average of neighbors with 1 edge: " + str(np.mean(one_sided_edge)))
    sns.histplot(data=two_sided_edge, kde=True)
    plt.xlabel("Correlations")
    plt.title('Correlation of 2 sided edges between neighbors')
    plt.show()
    print("number of neighbors with 2 edges: " + str(len(two_sided_edge)))
    print("average of neighbors with 2 edges: " + str(np.mean(two_sided_edge)))


def in_out_degree_distribution(in_degree_list, out_degree_list):
    sns.histplot(data=in_degree_list, kde=True)
    plt.xlabel("In Degree")
    plt.title('Pillars In Degree Distribution')
    plt.show()
    print("In degree average: " + str(np.mean(in_degree_list)))
    sns.histplot(data=out_degree_list, kde=True)
    plt.xlabel("Out Degree")
    plt.title('Pillars Out Degree Distribution')
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
    plt.plot(range(1, output_df.shape[1]+1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
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
        plt.show()


def gc_edge_probability_original_vs_random(gc_df):
    gc_edge_prob = []
    idx = []
    for i in range(10):
        prob = probability_for_gc_edge(gc_df, random_neighbors=True)
        gc_edge_prob.append(prob)
        idx.append(i)
    print("avg gc edge probability for random " + str(np.mean(gc_edge_prob)))
    print("std: " + str(np.std(gc_edge_prob)))
    fig, ax = plt.subplots()
    original = probability_for_gc_edge(gc_df, random_neighbors=False)
    ax.scatter(0, np.mean(gc_edge_prob), label="random")
    ax.scatter(1, original, label="original")
    plt.ylabel("Edge Probability")
    plt.title("GC Edge Probability - Original vs. Random Neighbors")
    ax.legend()
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
    df_segm_pca_kmeans.columns.values[-n_components:] = ['Component ' + str(i+1) for i in range(n_components)]
    df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_
    df_segm_pca_kmeans['Segment'] = df_segm_pca_kmeans['Segment K-means PCA'].map({0: 'first', 1: 'second'})
    for i in range(n_components):
        x_axis = df_segm_pca_kmeans['Component ' + str(i+1)]
        for j in range(i+1, n_components):
            y_axis = df_segm_pca_kmeans['Component ' + str(j+1)]
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x_axis, y_axis, hue=df_segm_pca_kmeans['Segment'], palette=['g', 'r'])
            plt.title('Clusters by PCA Components')
            plt.show()
