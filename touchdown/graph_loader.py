class Node:
    def __init__(self, panoid, pano_yaw_angle, lat, lng):
        self.panoid = panoid
        self.pano_yaw_angle = pano_yaw_angle
        self.neighbors = {}
        self.coordinate = (lat, lng)


class Graph:
    def __init__(self):
        self.nodes = {}
        
    def add_node(self, panoid, pano_yaw_angle, lat, lng):
        self.nodes[panoid] = Node(panoid, pano_yaw_angle, lat, lng)

    def add_edge(self, start_panoid, end_panoid, heading):
        start_node = self.nodes[start_panoid]
        end_node = self.nodes[end_panoid]
        start_node.neighbors[heading] = end_node


class GraphLoader:
    def __init__(self, opts):
        self.opts = opts
        self.graph = Graph()
        self.node_file = "%s/graph/nodes.txt" % opts.dataset
        self.link_file = "%s/graph/links.txt" % opts.dataset

    def construct_graph(self):
        with open(self.node_file) as f:
            for line in f:
                panoid, pano_yaw_angle, lat, lng = line.strip().split(',')
                self.graph.add_node(panoid, int(pano_yaw_angle), float(lat), float(lng))

        with open(self.link_file) as f:
            for line in f:
                start_panoid, heading, end_panoid = line.strip().split(',')
                self.graph.add_edge(start_panoid, end_panoid, float(heading))

        num_edges = 0
        for panoid in self.graph.nodes.keys():
            num_edges += len(self.graph.nodes[panoid].neighbors)

        return self.graph
