from Pillars.pillar_intensities import *
from Pillars.pillar_neighbors import *
from Pillars.pillars_graph import *


def get_frame_to_graph():
    path = get_images_path()
    pillar_to_neighbors, pillar_to_cross_neighbors = get_pillar_to_neighbors()
    pillar_frame_intensity_dict = get_pillar_to_intensities(path)
    images = get_images(path)
    frame_to_graph_dict = {}
    for i in range(len(images)):
        pillars_graph = PillarsGraph()
        # fill the graph with pillar nodes
        for pillar in pillar_frame_intensity_dict.items():
            pillar_id = pillar[0]
            pillar_intensity = pillar[1][i]
            pillar_neighbors = pillar_to_neighbors[pillar_id]
            pillar_node = PillarNode(pillar_id, pillar_intensity, i)
            pillars_graph.add_pillar_node(pillar_id, pillar_node)
        # fill each node with his neighbors nodes
        for pillar_item in pillars_graph.pillar_id_to_node.items():
            pillar_id = pillar_item[0]
            pillar_node = pillar_item[1]
            pillar_to_node = pillars_graph.pillar_id_to_node
            for neighbor in pillar_to_neighbors[pillar_id]:
                pillar_node.add_neighbor(pillar_to_node[neighbor])
            # TODO: also on cross neighbors?

        frame_to_graph_dict[i] = pillars_graph

    return frame_to_graph_dict
