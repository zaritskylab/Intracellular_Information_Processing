from itertools import permutations

from statsmodels.tsa.stattools import kpss
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from Pillars.analyzer import *
from Pillars.consts import *
import pandas as pd
import statsmodels.api as sm


def differencing_time_series(pillar_to_intens_dict):
    pillar_to_differenced_intens = {node: np.diff(series) for node, series in pillar_to_intens_dict.items()}

    return pillar_to_differenced_intens


def get_stationary_and_non_stationary_pillars(pillar_to_intens_dict):
    def check_stationarity(time_series):
        result = sm.tsa.adfuller(time_series, autolag='AIC')
        return result[1] <= 0.05

    stationary_p_to_intens = {}
    non_stationary_pillars = []
    for pillar, ts in pillar_to_intens_dict.items():
        if check_stationarity(ts):
            stationary_p_to_intens[pillar] = ts
        else:
            non_stationary_pillars.append(pillar)

    print("passed stationary", len(stationary_p_to_intens), "not passed stationary", len(non_stationary_pillars))
    return stationary_p_to_intens, non_stationary_pillars


def find_optimal_lag(stationary_pillar_to_intensity, max_lag=3):
    df = pd.DataFrame(stationary_pillar_to_intensity)
    model = sm.tsa.VAR(df)
    results = model.select_order(maxlags=max_lag)
    results.summary()


def perform_granger_test(pillar_pairs, stationary_pillar_to_intensity, maxlag=3):
    granger_results = {}
    for pair in pillar_pairs:
        p1 = pair[0]
        p2 = pair[1]
        if p1 in stationary_pillar_to_intensity.keys() and p2 in stationary_pillar_to_intensity.keys():
            test_result = grangercausalitytests(
                list(zip(stationary_pillar_to_intensity[p1], stationary_pillar_to_intensity[p2])), maxlag=maxlag, verbose=False )
            p_values = [test_result[i + 1][0]['ssr_chi2test'][1] for i in range(maxlag)]
            granger_results[(p1, p2)] = min(p_values)
    return granger_results


def perform_granger_test_for_adjacent(stationary_pillar_to_intensity, maxlag=3):
    neighbors = get_alive_pillars_to_alive_neighbors()
    nbrs_granger_results = {}
    for pillar1 in stationary_pillar_to_intensity:
        for pillar2 in stationary_pillar_to_intensity:
            if pillar1 != pillar2 and pillar1 in neighbors[pillar2]:
                test_result = grangercausalitytests(
                    list(zip(stationary_pillar_to_intensity[pillar1], stationary_pillar_to_intensity[pillar2])),
                    maxlag=maxlag, verbose=False
                )
                p_values = [test_result[i + 1][0]['ssr_chi2test'][1] for i in range(maxlag)]
                nbrs_granger_results[(pillar1, pillar2)] = min(p_values)
    return nbrs_granger_results


def perform_granger_test_for_non_adjacent(stationary_pillar_to_intensity, maxlag=3):
    neighbors = get_alive_pillars_to_alive_neighbors()
    non_nbrs_granger_results = {}
    for pillar1 in stationary_pillar_to_intensity:
        for pillar2 in stationary_pillar_to_intensity:
            if pillar1 != pillar2 and pillar1 not in neighbors[pillar2]:
                test_result = grangercausalitytests(
                    list(zip(stationary_pillar_to_intensity[pillar1], stationary_pillar_to_intensity[pillar2])),
                    maxlag=maxlag, verbose=False
                )
                p_values = [test_result[i + 1][0]['ssr_chi2test'][1] for i in range(maxlag)]
                non_nbrs_granger_results[(pillar1, pillar2)] = min(p_values)
    return non_nbrs_granger_results


def perform_statistical_test(adjacent_results, non_adjacent_results):
    # Extract p-values from results dictionaries
    adjacent_p_values = list(adjacent_results.values())
    non_adjacent_p_values = list(non_adjacent_results.values())

    # Perform Mann-Whitney U Test
    stat, p_value = stats.mannwhitneyu(adjacent_p_values, non_adjacent_p_values, alternative='two-sided')

    print(f"Mann-Whitney U test stat: {stat}, p-value: {p_value}")

    # Determine significance
    if p_value < 0.05:
        print("Significant difference between adjacent and non-adjacent neighbors' Granger causality.")
    else:
        print("No significant difference between adjacent and non-adjacent neighbors' Granger causality.")


def build_gc_graph(gc_dict, threshold=0.05):
    G = nx.DiGraph()

    nodes_loc = get_alive_pillar_ids_overall_v3()
    nodes_loc = list(nodes_loc)
    node_loc2index = {}
    for i in range(len(nodes_loc)):
        node_loc2index[str(nodes_loc[i])] = i
        G.add_node(i, pos=nodes_loc[i])

    for (p1, p2), p_value in gc_dict.items():
        if p_value < threshold:
            G.add_edge(node_loc2index[str(p1)], node_loc2index[str(p2)])

    return G


def plot_graph(G):
    img = get_last_image_whiten(build_image=Consts.build_image)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    pos = nx.get_node_attributes(G, 'pos')
    nodes_loc_y_inverse = {k: (v[1], v[0]) for k, v in pos.items()}

    nx.draw(G, nodes_loc_y_inverse, edge_color='tab:red', width=1.5, node_size=100, arrows=True)

    plt.show()


def in_degree_centrality(G):
    neighbors = get_alive_pillars_to_alive_neighbors()
    custom_in_degree_centrality = {}
    max_neighbors = max(len(neighbors[n]) for n in neighbors)

    for i, p in G.nodes(data=True):
        node_pos = p.get('pos')
        num_neighbors = len(neighbors[node_pos])
        raw_in_degree = G.in_degree(i)
        centrality = raw_in_degree / num_neighbors if num_neighbors > 0 else 0
        weight_factor = num_neighbors / max_neighbors
        custom_in_degree_centrality[i] = centrality * weight_factor

    return custom_in_degree_centrality


def plot_graph_in_degree_label(G, in_degree_nodes, non_stationary_pillars):
    img = get_last_image_whiten(build_image=Consts.build_image)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    pos = nx.get_node_attributes(G, 'pos')
    nodes_loc_y_inverse = {k: (v[1], v[0]) for k, v in pos.items()}

    non_stationary_pillars_idx = [i for i, n in G.nodes(data=True) if n.get('pos') in non_stationary_pillars]
    only_stationary_pillars_data = {k: round(v, 2) for k, v in in_degree_nodes.items() if
                                    k not in non_stationary_pillars_idx}
    # avg_weighted_degree = np.sum(list(only_stationary_pillars_data.values())) / len(only_stationary_pillars_data.values())
    top_weighted_degree = np.percentile(list(only_stationary_pillars_data.values()), 85)

    in_degree_nodes = {k: round(v, 2) for k, v in in_degree_nodes.items()}
    top_weighted_nodes = {n: d for n, d in in_degree_nodes.items() if d > top_weighted_degree}
    nodes_color = ['tab:red' if d > top_weighted_degree else 'tab:blue' for n, d in in_degree_nodes.items()]
    nx.draw(G, nodes_loc_y_inverse, node_size=250, arrows=True, width=1.5, node_color=nodes_color)
    nx.draw_networkx_labels(G, nodes_loc_y_inverse, labels=in_degree_nodes, font_size=8)

    plt.show()

    return top_weighted_nodes


def out_degree_centrality(G):
    neighbors = get_alive_pillars_to_alive_neighbors()
    custom_out_degree_centrality = {}
    max_neighbors = max(len(neighbors[n]) for n in neighbors)
    for i, p in G.nodes(data=True):
        node_pos = p.get('pos')
        num_neighbors = len(neighbors[node_pos])
        raw_out_degree = G.out_degree(i)
        centrality = raw_out_degree / num_neighbors if num_neighbors > 0 else 0
        weight_factor = num_neighbors / max_neighbors
        custom_out_degree_centrality[i] = centrality * weight_factor

    return custom_out_degree_centrality


def plot_graph_out_degree_label(G, out_degree_nodes, non_stationary_pillars):
    img = get_last_image_whiten(build_image=Consts.build_image)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    pos = nx.get_node_attributes(G, 'pos')
    nodes_loc_y_inverse = {k: (v[1], v[0]) for k, v in pos.items()}

    non_stationary_pillars_idx = [i for i, n in G.nodes(data=True) if n.get('pos') in non_stationary_pillars]
    only_stationary_pillars_data = {k: round(v, 2) for k, v in out_degree_nodes.items() if
                                    k not in non_stationary_pillars_idx}
    # avg_weighted_degree = np.sum(list(only_stationary_pillars_data.values())) / len(only_stationary_pillars_data.values())
    top_weighted_degree = np.percentile(list(only_stationary_pillars_data.values()), 85)

    out_degree_nodes = {k: round(v, 2) for k, v in out_degree_nodes.items()}
    top_weighted_nodes = {n: d for n, d in out_degree_nodes.items() if d > top_weighted_degree}
    nodes_color = ['tab:green' if d > top_weighted_degree else 'tab:blue' for n, d in out_degree_nodes.items()]
    nx.draw(G, nodes_loc_y_inverse, node_size=250, arrows=True, width=1.5, node_color=nodes_color)
    nx.draw_networkx_labels(G, nodes_loc_y_inverse, labels=out_degree_nodes, font_size=8)

    plt.show()

    return top_weighted_nodes


def plot_main_in_out_centrality_pillars(G, non_stationary_pillars, in_degree_nodes, out_degree_nodes):
    non_stationary_pillars_idx = [i for i, n in G.nodes(data=True) if n.get('pos') in non_stationary_pillars]

    stationary_pillars = [k for k in in_degree_nodes.keys() if k not in non_stationary_pillars_idx]
    only_stationary_pillars_data_in = {k: round(v, 2) for k, v in in_degree_nodes.items() if
                                       k not in non_stationary_pillars_idx}
    top_weighted_in_degree = np.percentile(list(only_stationary_pillars_data_in.values()), 85)

    only_stationary_pillars_data_out = {k: round(v, 2) for k, v in out_degree_nodes.items() if
                                        k not in non_stationary_pillars_idx}
    top_weighted_out_degree = np.percentile(list(only_stationary_pillars_data_out.values()), 85)

    img = get_last_image_whiten(build_image=Consts.build_image)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    pos = nx.get_node_attributes(G, 'pos')
    nodes_loc_y_inverse = {k: (v[1], v[0]) for k, v in pos.items()}
    nodes_color = [
        'tab:orange' if p in stationary_pillars and only_stationary_pillars_data_out[p] > top_weighted_out_degree and
                        only_stationary_pillars_data_in[p] > top_weighted_in_degree else 'tab:blue' for p in
        in_degree_nodes.keys()]
    nx.draw(G, nodes_loc_y_inverse, node_size=200, arrows=True, width=1.5, node_color=nodes_color)
    plt.show()


def eigenvector_centrality(G):
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

    return eigenvector_centrality


def connected_component(G):
    scc = list(nx.strongly_connected_components(G))
    comp = [c for c in scc if len(c) > 1]
    color_list = plt.cm.tab10(np.linspace(0, 1, len(comp)))
    component_map = {node: color_list[cid] for cid, component in enumerate(comp) for node in component}
    colors = [component_map.get(node, 'tab:blue') for node in G.nodes()]

    # component_map = {node: cid for cid, component in enumerate(comp) for node in component}
    # colors = [component_map[node] if node in component_map.keys() else 'tab:blue' for node in G.nodes()]

    img = get_last_image_whiten(build_image=Consts.build_image)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    pos = nx.get_node_attributes(G, 'pos')
    nodes_loc_y_inverse = {k: (v[1], v[0]) for k, v in pos.items()}

    nx.draw(G, nodes_loc_y_inverse, node_color=colors, node_size=100, arrows=True)
    plt.show()
