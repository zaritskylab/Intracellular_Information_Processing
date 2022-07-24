class PillarsGraph:
    def __init__(self):
        self.pillar_id_to_node = {}

    def add_pillar_node(self, pillar_id, node):
        self.pillar_id_to_node[pillar_id] = node


class PillarNode:
    def __init__(self, pillar_id, intensity, frame_number):
        self.id = pillar_id
        self.curr_frame = frame_number
        self.intensity = intensity
        self.neighbors_node_list = []
        #TODO: self.prev_frame : PillarNode

    def add_neighbors(self, neighbors: list):
        self.neighbors_node_list.extend(neighbors)

    def add_neighbor(self, neighbor):
        self.neighbors_node_list.append(neighbor)
