from Pillars.pillars_utils import *
import random
from Pillars.consts import *
import copy

def get_alive_pillars_in_edges_to_l1_neighbors():
    """
    Mapping the alive pillars in the edges to their level 1 background neighbors
    :return: dictionary mapping the alive pillars to their background neighbors, list of the alive pillars in the edges,
            list of the level one background pillars
    """
    pillar2mask = get_last_img_mask_for_each_pillar(get_all_center_generated_ids())
    alive_pillars = get_alive_pillar_ids_in_last_frame_v3()
    all_pillars = pillar2mask.keys()
    background_pillars = [pillar for pillar in all_pillars if
                          pillar not in alive_pillars]
    pillar_to_neighbors = get_pillar_to_neighbors()
    edge_pillars = set()
    back_pillars_level_1 = set()
    edge_pillar_to_back_nbrs_level_1 = {}
    for pillar in all_pillars:
        nbrs = pillar_to_neighbors[pillar]
        if pillar in alive_pillars:
            back_neighbors = []
            for n in nbrs:
                if n in background_pillars:
                    edge_pillars.add(pillar)
                    back_neighbors.append(n)
                    back_pillars_level_1.add(n)
            if len(back_neighbors) > 0:
                edge_pillar_to_back_nbrs_level_1[pillar] = back_neighbors

    return edge_pillar_to_back_nbrs_level_1, list(edge_pillars), list(back_pillars_level_1)


def get_background_level_1_to_level_2():
    """
    Mapping pillar in background from level 1 (neighbors of alive pillars) to their neighbors background pillars in level 2
    :return:
    """
    _, _, back_pillars_level_1 = get_alive_pillars_in_edges_to_l1_neighbors()
    pillar_to_neighbors = get_pillar_to_neighbors()
    pillar2mask = get_last_img_mask_for_each_pillar(get_all_center_generated_ids())
    alive_pillars = get_alive_pillar_ids_in_last_frame_v3()
    all_pillars = pillar2mask.keys()
    background_pillars = [pillar for pillar in all_pillars if
                          pillar not in alive_pillars]
    back_pillars_l1_to_l2 = {}
    for pillar_l1 in back_pillars_level_1:
        back_pillars_level_2 = []
        for n in pillar_to_neighbors[pillar_l1]:
            if n in background_pillars and n not in back_pillars_level_1:
                back_pillars_level_2.append(n)
        if len(back_pillars_level_2) > 0:
            back_pillars_l1_to_l2[pillar_l1] = back_pillars_level_2

    return back_pillars_l1_to_l2


def get_pillar_to_neighbors():
    """
    Mapping each pillar to its neighbors
    :return:
    """
    if Consts.USE_CACHE and os.path.isfile(Consts.pillar_to_neighbors_cache_path):
        with open(Consts.pillar_to_neighbors_cache_path, 'rb') as handle:
            pillar_to_neighbors = pickle.load(handle)
            return pillar_to_neighbors

    alive_centers = get_seen_centers_for_mask()

    pillar_ids, rule_jump_1, rule_jump_2, generated_location2real_pillar_loc = \
        generate_centers_and_rules_from_alive_centers(alive_centers, Consts.IMAGE_SIZE_ROWS, Consts.IMAGE_SIZE_COLS)
    real_pillar_loc2generated_location = dict((v, k) for k, v in generated_location2real_pillar_loc.items())
    pillar_to_neighbors = {}
    for pillar_id in pillar_ids:
        # If pillar_actual_location is the original location (moved), we should treat it as the generated location for the rules
        # Still, the generated location is the ID for the pillar.
        if pillar_id in real_pillar_loc2generated_location:
            pillar_center_to_activate_rule = real_pillar_loc2generated_location[pillar_id]
        else:
            pillar_center_to_activate_rule = pillar_id

        n1 = (pillar_center_to_activate_rule[0] - rule_jump_1[0], pillar_center_to_activate_rule[1] - rule_jump_1[1])
        n2 = (pillar_center_to_activate_rule[0] + rule_jump_1[0], pillar_center_to_activate_rule[1] + rule_jump_1[1])
        n3 = (pillar_center_to_activate_rule[0] - rule_jump_2[0], pillar_center_to_activate_rule[1] - rule_jump_2[1])
        n4 = (pillar_center_to_activate_rule[0] + rule_jump_2[0], pillar_center_to_activate_rule[1] + rule_jump_2[1])
        n_minus1_minus2 = (n1[0] - rule_jump_2[0], n1[1] - rule_jump_2[1])
        n_minus1_plus2 = (n1[0] + rule_jump_2[0], n1[1] + rule_jump_2[1])
        n_plus1_minus2 = (n2[0] - rule_jump_2[0], n2[1] - rule_jump_2[1])
        n_plus1_plus2 = (n2[0] + rule_jump_2[0], n2[1] + rule_jump_2[1])

        potential_neighbors = {n1, n2, n3, n4, n_minus1_plus2, n_minus1_minus2, n_plus1_minus2, n_plus1_plus2}
        neighbors_lst = list(potential_neighbors.intersection(pillar_ids))

        # In case we switched pillar location from rule based to actual live pillar base, we take the actual live pillar location
        neighbors_lst.extend(
            [generated_location2real_pillar_loc[potential_nbr] for potential_nbr in potential_neighbors if potential_nbr in generated_location2real_pillar_loc]
        )

        pillar_to_neighbors[pillar_id] = list(set(neighbors_lst))

    if Consts.USE_CACHE:
        with open(Consts.pillar_to_neighbors_cache_path, 'wb') as handle:
            pickle.dump(pillar_to_neighbors, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pillar_to_neighbors


def dfs(graph, node, visited, component):
    # Mark the current node as visited
    visited[node] = True
    # Add it to the current connected component
    component.append(node)
    # Recur for all the adjacent nodes
    for neighbor in graph[node]:
        if not visited.get(neighbor, False):
            dfs(graph, neighbor, visited, component)


def find_connected_components(graph):
    visited = {}  # Dictionary to keep track of visited nodes
    components = []  # List to store connected components

    for node in graph:
        if not visited.get(node, False):
            component = []  # Initialize an empty connected component
            dfs(graph, node, visited, component)
            components.append(component)

    return components



def get_pillar_indirect_neighbors_dict(pillar_location):
    """
    Mapping pillar to its indirect neighbors (start from level 2 neighbors)
    :param pillar_location:
    :return:
    """
    pillar_directed_neighbors = get_pillar_directed_neighbors(pillar_location)
    neighbors1, neighbors2 = get_pillar_to_neighbors()
    indirect_neighbors_dict = {}
    for n in neighbors1.keys():
        if n not in pillar_directed_neighbors:
            indirect_neighbors_dict[n] = neighbors1[n]
    for n in neighbors2.keys():
        if n not in pillar_directed_neighbors:
            indirect_neighbors_dict[n] = neighbors2[n]

    return indirect_neighbors_dict


def get_pillar_directed_neighbors(pillar_location):
    """
    Creating a list of a pillar's directed neighbors
    :param pillar_location:
    :return:
    """
    neighbors1, neighbors2 = get_pillar_to_neighbors()
    pillar_directed_neighbors = []
    pillar_directed_neighbors.extend(neighbors1[pillar_location])
    pillar_directed_neighbors.extend(neighbors2[pillar_location])
    pillar_directed_neighbors.append(pillar_location)

    return pillar_directed_neighbors


def get_alive_pillars_to_alive_neighbors():
    """
    Mapping each alive pillar to its level 1 alive neighbors
    :return:
    """
    if Consts.USE_CACHE and os.path.isfile(Consts.alive_pillars_to_alive_neighbors_cache_path):
        with open(Consts.alive_pillars_to_alive_neighbors_cache_path, 'rb') as handle:
            alive_pillars_to_alive_neighbors = pickle.load(handle)
            return alive_pillars_to_alive_neighbors

    Consts.MULTI_COMPONENT = False

    pillar_to_neighbors = get_pillar_to_neighbors()
    alive_pillars = get_alive_pillar_ids_in_last_frame_v3()
    alive_pillars_to_alive_neighbors = {}
    for p, nbrs in pillar_to_neighbors.items():
        if p in alive_pillars:
            alive_nbrs = []
            for nbr in nbrs:
                if nbr in alive_pillars:
                    alive_nbrs.append(nbr)
            alive_pillars_to_alive_neighbors[p] = alive_nbrs

    # Remove pillars neighours that are not connected to main graph
    components = find_connected_components(alive_pillars_to_alive_neighbors)
    print("number of components:", len(components))
    if len(components) > 1:
        Consts.MULTI_COMPONENT = True
    # Remove biggest component
    longest_list = max(components, key=len)
    components.remove(longest_list)

    for component in components:
        for pillar in component:
            alive_pillars_to_alive_neighbors[pillar] = []

    if Consts.USE_CACHE:
        with open(Consts.alive_pillars_to_alive_neighbors_cache_path, 'wb') as handle:
            pickle.dump(alive_pillars_to_alive_neighbors, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return alive_pillars_to_alive_neighbors


def get_random_neighbors():
    """
    Mapping pillar to new random neighbors (fake neighbors)
    :return:
    """
    pillar_to_nbrs = get_alive_pillars_to_alive_neighbors()
    alive_pillars = list(pillar_to_nbrs.keys())
    new_neighbors_dict = {}

    for pillar, nbrs in pillar_to_nbrs.items():
        num_of_nbrs = len(nbrs)
        if pillar in new_neighbors_dict.keys():
            num_of_nbrs = num_of_nbrs - len(new_neighbors_dict[pillar])
        relevant_pillars = alive_pillars
        relevant_pillars = [p for p in relevant_pillars if p not in nbrs and p != pillar]
        new_nbrs = []
        if len(relevant_pillars) < num_of_nbrs:
            num_of_nbrs = len(relevant_pillars)
        for i in range(num_of_nbrs):
            new_nbr = random.choice(relevant_pillars)
            new_nbrs.append(new_nbr)
            if new_nbr in new_neighbors_dict.keys():
                new_neighbors_dict[new_nbr].append(pillar)
            else:
                new_neighbors_dict[new_nbr] = [pillar]
            relevant_pillars.remove(new_nbr)
        new_neighbors_dict[pillar] = new_nbrs

    return new_neighbors_dict
