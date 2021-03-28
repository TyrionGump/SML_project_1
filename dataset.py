import pandas as pd
import numpy as np
import math
import copy
import itertools
import os


class Graph:
    def __init__(self, file_path, search_range=2):
        self.search_range = search_range
        self.txt_data = None
        self.connections = {}
        self.common_nodes = {}
        self.near_nodes = {}
        self.adjacency_matrix = None
        self.node_to_idx = {}
        self.idx_to_node = {}

        self.degrees = {}
        self.CN = {}
        self.Salton = {}
        self.Jaccard = {}
        self.local_path = {}
        self.Katz = {}

        self.features = {'CN': self.CN, 'Salton': self.Salton, 'Jaccard': self.Jaccard,
                         'LP': self.local_path, 'Katz': self.Katz}
        self.dataset = None

        self.__load_file__(file_path)
        self.__scan_connection__()
        self.__generate_adjacency_matrix__()
        self.__near_nodes__()
        self.__common_nodes_and_degrees__()

    def __load_file__(self, file_path):
        with open(file_path) as f:
            self.txt_data = f.readlines()

    def __scan_connection__(self):
        for line in self.txt_data:
            line = line.strip('\n')
            line = line.split(' ')
            connections = list(itertools.permutations(line, 2))
            for connection in connections:
                if int(connection[0]) == int(connection[1]):
                    continue
                if int(connection[0]) not in self.connections.keys():
                    self.connections[int(connection[0])] = []
                self.connections[int(connection[0])].append(int(connection[1]))

        for key, value in self.connections.items():
            self.connections[key] = list(set(value))

    def __generate_adjacency_matrix__(self):
        author_num = len(self.connections.keys())
        self.adjacency_matrix = np.zeros((author_num, author_num))
        self.node_to_idx = dict(zip(self.connections.keys(), range(author_num)))
        self.idx_to_node = {i: node for node, i in self.node_to_idx.items()}

        for node_1, node_1_connections in self.connections.items():
            for node_2 in node_1_connections:
                self.adjacency_matrix[self.node_to_idx[node_1], self.node_to_idx[node_2]] = 1

    def __common_nodes_and_degrees__(self):
        for node_1, node_1_connections in self.connections.items():
            if node_1 not in self.common_nodes.keys():
                self.common_nodes[node_1] = {}
            for node_2 in node_1_connections:
                node_2_connection = self.connections[node_2]
                self.common_nodes[node_1][node_2] = list(set(node_1_connections) & set(node_2_connection))
            self.degrees[node_1] = len(node_1_connections)

    def __near_nodes__(self):
        for node in self.connections.keys():
            self.near_nodes[node] = self.__dfs_search__(node)

    def __dfs_search__(self, node):
        queue, visit_node = [], []
        queue.append(node)
        search_dist = self.search_range
        while queue:
            n = queue.pop()
            visit_node.append(n)

            if not search_dist:
                continue
            for next_node in self.connections[n]:
                if next_node not in queue and next_node not in visit_node:
                    queue.append(next_node)
            search_dist -= 1

        visit_node.remove(node)
        return visit_node

    def cal_CN(self):
        for node_1 in self.common_nodes.keys():
            if node_1 not in self.CN.keys():
                self.CN[node_1] = {}
            for node_2, common_nodes in self.common_nodes[node_1].items():
                self.CN[node_1][node_2] = len(common_nodes)

    def cal_Salton(self):
        for node_1 in self.common_nodes.keys():
            if node_1 not in self.Salton.keys():
                self.Salton[node_1] = {}
            for node_2, common_nodes in self.common_nodes[node_1].items():
                self.Salton[node_1][node_2] = len(common_nodes) / math.sqrt(self.degrees[node_1] * self.degrees[node_2])

    def cal_Jaccard(self):
        for node_1 in self.common_nodes.keys():
            if node_1 not in self.Jaccard.keys():
                self.Jaccard[node_1] = {}
            for node_2, common_nodes in self.common_nodes[node_1].items():
                self.Jaccard[node_1][node_2] = len(common_nodes) / len(set(self.connections[node_1]) |
                                                                       set(self.connections[node_2]))

    def cal_local_path(self, alpha=0.01):
        CN_matrix = np.dot(self.adjacency_matrix, self.adjacency_matrix)
        LP_matrix = CN_matrix + alpha * np.dot(self.adjacency_matrix, CN_matrix)

        for i in range(LP_matrix.shape[0]):
            for j in range(LP_matrix.shape[1]):
                if self.idx_to_node[i] not in self.local_path.keys():
                    self.local_path[self.idx_to_node[i]] = {}
                self.local_path[self.idx_to_node[i]][self.idx_to_node[j]] = LP_matrix[i][j]

    def cal_Katz(self, beta=0.01):
        identity = np.eye(self.adjacency_matrix.shape[0])
        Katz_matrix = np.linalg.inv(identity - beta * self.adjacency_matrix) - identity
        for i in range(Katz_matrix.shape[0]):
            for j in range(Katz_matrix.shape[1]):
                if self.idx_to_node[i] not in self.Katz.keys():
                    self.Katz[self.idx_to_node[i]] = {}
                self.Katz[self.idx_to_node[i]][self.idx_to_node[j]] = Katz_matrix[i][j]

    def generate_dataset(self):
        x = []
        for line in self.txt_data:
            line = line.strip('\n')
            line = line.split(' ')
            connections = list(itertools.permutations(line, 2))
            for connection in connections:
                node_1 = int(connection[0])
                node_2 = int(connection[1])
                row = []
                for feature_name, feature in self.features.items():
                    if feature is not {}:
                        row.append(feature[node_1].setdefault(node_2, 0))
                row.append(self.degrees[node_1])
                row.append(self.degrees[node_2])

                row.append(1)
                x.append(row)

        for node_1, connections in self.connections.items():
            unconnected_nodes = list(set(self.near_nodes[node_1]) ^ set(connections))

            for node_2 in unconnected_nodes:
                row = []
                for feature_name, feature in self.features.items():
                    if feature is not {}:
                        row.append(feature[node_1].setdefault(node_2, 0))

                row.append(self.degrees[node_1])
                row.append(self.degrees[node_2])
                row.append(0)
                x.append(row)

        self.dataset = pd.DataFrame(x, columns=['CN', 'Salton', 'Jaccard', 'LP', 'Katz', 'node_1_degree', 'node_2_degree', 'y'])

    def get_dataset(self):
        return self.dataset.iloc[:, :-1], self.dataset.iloc[:, -1]


g = Graph(os.path.join('../..', 'data/train.txt'))

g.cal_CN()
g.cal_Salton()
g.cal_Jaccard()
g.cal_local_path()
g.cal_Katz()
g.generate_dataset()
X, Y = g.get_dataset()

print(X)
print(Y)


