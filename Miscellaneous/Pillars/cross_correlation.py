from Pillars.analyzer import *
import seaborn as sns
import matplotlib.patches as mpatches


def cross_correlations(pillar_to_intensities, pillars_neighbors, max_lag=3):
    # pillars_time_series_stationary, non_stationary_pillars = adf_stationary_test(pillar_to_intensities)

    def normalize_time_series(data):
        normalized_data = {}
        for key, series in data.items():
            mean = np.mean(series)
            std = np.std(series)
            normalized_series = (series - mean) / std if std != 0 else series - mean
            normalized_data[key] = normalized_series
        return normalized_data

    normalized_ts = normalize_time_series(pillar_to_intensities)

    def calculate_pearson_correlation(series1, series2, max_lag=3):
        corr_values = []
        lags = np.arange(-max_lag, max_lag + 1)
        for lag in lags:
            if lag < 0:
                series1_lagged, series2_lagged = series1[:lag], series2[-lag:]
            elif lag > 0:
                series1_lagged, series2_lagged = series1[lag:], series2[:-lag]
            else:
                series1_lagged, series2_lagged = series1, series2

            if len(series1_lagged) < 2 or len(series2_lagged) < 2:
                continue  # Not enough data points to calculate correlation
            else:
                corr, _ = pearsonr(series1_lagged, series2_lagged)

            corr_values.append(corr)
        return lags, corr_values

    correlations = {}
    for pillar, neighbors in pillars_neighbors.items():
        for neighbor in neighbors:
            if (neighbor, pillar) not in correlations and (pillar, neighbor) not in correlations:
                lags, corr = calculate_pearson_correlation(normalized_ts[pillar], normalized_ts[neighbor], max_lag)
                correlations[(pillar, neighbor)] = (lags, corr)

    return correlations


def identify_peaks(correlations):
    peak_lags = {}
    for pair, corr in correlations.items():
        # lag = np.argmax(np.abs(corr[1])) - (len(corr[1]) // 2)
        lag = corr[0][np.argmax(np.abs(corr[1]))]
        cor = corr[1][np.argmax(np.abs(corr[1]))]
        peak_lags[pair] = (lag, cor)
    return peak_lags


def categorize_patterns(peak_lags):
    leading_nodes = {}
    lagging_nodes = {}
    for pair, lag in peak_lags.items():
        if lag[0] > 0:
            leading_nodes[pair] = lag
        elif lag[0] < 0:
            lagging_nodes[pair] = lag
    return leading_nodes, lagging_nodes


def cross_correlation_avg_each_lag(correlations):
    corrs_in_lag_dict = {}
    for lags, corrs in correlations.values():
        lags_lst = list(lags)
        for i, lag in enumerate(lags_lst):
            if lag in corrs_in_lag_dict:
                corrs_in_lag_dict[lag].append(corrs[i])
            else:
                corrs_in_lag_dict[lag] = []
                corrs_in_lag_dict[lag].append(corrs[i])

    avg_corrs_in_lag_dict = {k: np.mean(v) for k, v in corrs_in_lag_dict.items()}
    return avg_corrs_in_lag_dict


def in_out_degree(DG):
    in_degrees = DG.in_degree()
    in_degrees_dict = dict(in_degrees)

    out_degrees = DG.out_degree()
    out_degrees_dict = dict(out_degrees)

    return in_degrees_dict, out_degrees_dict


def plot_cross_correlation_directed_graph(peak_correlations):
    # Create a directed graph
    DG = nx.DiGraph()

    # Transfer nodes to the directed graph and create position dictionary
    nodes_loc = get_alive_pillar_ids_overall_v3()
    nodes_loc = list(nodes_loc)
    node_loc2index = {}
    pos = {}
    for i, loc in enumerate(nodes_loc):
        node_loc2index[str(loc)] = i
        DG.add_node(i)
        pos[i] = (loc[1], loc[0])

    # Iterate through each edge and determine direction based on peak_correlations
    for pair in peak_correlations:
        lag, corr = peak_correlations[pair]
        # Convert pair to indices
        node1_idx = node_loc2index[str(pair[0])]
        node2_idx = node_loc2index[str(pair[1])]

        if lag < 0:
            DG.add_edge(node2_idx, node1_idx, weight=corr)
        elif lag > 0:
            DG.add_edge(node1_idx, node2_idx, weight=corr)

    # Draw the directed graph
    edges, weights = zip(*nx.get_edge_attributes(DG, 'weight').items())
    img = get_last_image_whiten(build_image=Consts.build_image)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    nx.draw(DG, pos, edgelist=edges, edge_color=weights, width=1.5, node_size=100, arrows=True)

    # Create a ScalarMappable and normalize its color scale to [-1, 1]
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    sm.set_clim(-1, 1)
    plt.colorbar(sm)

    plt.show()

    return DG


def plot_total_avg_correlation_in_lag(lag_to_avg_corr_dict):
    lags = lag_to_avg_corr_dict.keys()
    avg_corrs = lag_to_avg_corr_dict.values()

    plt.plot(lags, avg_corrs)
    plt.xlabel('lag')
    plt.ylabel('avg correlation')
    plt.title('Avg Correlation in Lag')
    plt.show()


def lag_correlations_heatmap(correlations):
    columns = list(correlations.keys())
    index = list(list(correlations.values())[0][0])
    values = {}
    nodes_idx = 0
    for nodes, (lag, correlation) in correlations.items():
        lag_lst = list(lag)
        for i, l in enumerate(lag_lst):
            values[((nodes_idx, nodes), l)] = correlation[i]
        nodes_idx += 1
    cols = []
    for i, tup in enumerate(columns):
        cols.append(i)
    correlation_matrix_df = pd.DataFrame(index=cols, columns=sorted(index))
    for ((i, nodes), lag), correlation in values.items():
        correlation_matrix_df.at[i, lag] = correlation
    correlation_matrix_df = correlation_matrix_df.apply(pd.to_numeric, errors='coerce')

    plt.figure(figsize=(10, 10))

    sns.heatmap(correlation_matrix_df, annot=False, annot_kws={"size": 6}, vmin=-1, vmax=1)

    num_ticks = 10
    tick_positions = np.linspace(start=0, stop=len(correlation_matrix_df) - 1, num=num_ticks, dtype=int)
    tick_labels = [str(correlation_matrix_df.index[i]) for i in tick_positions]
    plt.yticks(tick_positions + 0.5, tick_labels)

    plt.xticks(rotation=90)
    plt.title('Correlation Heatmap')
    plt.ylabel('Node Pairs Index')
    plt.xlabel('Lag')
    plt.show()


def lag_distribution_plot(peak_lags):
    lags = [lag[0] for _, lag in peak_lags.items()]

    sns.histplot(lags, kde=True, alpha=0.3)
    plt.title('Distribution of Peak Correlation Lags')
    plt.xlabel('Lag')
    plt.show()


def plot_peak_correlation_vs_lag(peak_correlations):
    lags, corrs = zip(*peak_correlations.values())

    plt.scatter(lags, corrs)
    plt.title('Peak Correlation vs. Lag')
    plt.xlabel('Lag')
    plt.ylabel('Peak Correlation')
    plt.show()


def plot_nodes_correlations_through_lags(G, nodes, correlations):
    pillars_coor = []
    for node_idx in nodes:
        coor = G.nodes[node_idx]['pos']
        pillars_coor.append(coor)

    for pair in correlations.keys():
        if pair[0] in pillars_coor and pair[1] in pillars_coor:
            lags, corr_values = correlations[pair]
            plt.plot(lags, corr_values, label=pair)

    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_only_in_or_only_out_degree_in_graph(DG):
    nodes_loc = get_alive_pillar_ids_overall_v3()
    nodes_loc = list(nodes_loc)
    node_loc2index = {}
    pos = {}
    for i, loc in enumerate(nodes_loc):
        node_loc2index[str(loc)] = i
        DG.add_node(i)
        pos[i] = (loc[1], loc[0])

    img = get_last_image_whiten(build_image=Consts.build_image)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    node_color_map = {}
    node_color = []
    for node in DG.nodes():
        in_degree = DG.in_degree(node)
        out_degree = DG.out_degree(node)

        if in_degree > 0 and out_degree == 0:
            node_color.append('tab:red')
            node_color_map[node] = 'red'
        elif in_degree == 0 and out_degree > 0:
            node_color.append('tab:green')
            node_color_map[node] = 'green'
        else:
            node_color.append('tab:blue')
            node_color_map[node] = 'blue'

    edges, weights = zip(*nx.get_edge_attributes(DG, 'weight').items())
    nx.draw(DG, pos, edgelist=edges, edge_color=weights, width=1.5, node_size=100, arrows=True)
    nx.draw_networkx_nodes(DG, pos, node_size=100, node_color=node_color)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    sm.set_clim(-1, 1)
    plt.colorbar(sm)

    unique_colors = set(node_color_map.values())
    patches = [mpatches.Patch(color=color, label='only in' if color == 'red' else 'only out' if color == 'green' else 'in and out') for color in unique_colors]
    # node_colors = [node_color_map.get(node) for node in DG.nodes()]
    # nx.draw(DG, node_color=node_colors, with_labels=True)
    plt.legend(handles=patches, title='Node Degree')

    plt.show()

