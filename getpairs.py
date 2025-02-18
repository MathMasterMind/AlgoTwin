# Description: This script creates pairs of graphs and their corresponding labels for training the model.

import torch
from torch_geometric.data import Data
import random

def create_graph_data(graph):
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    x = torch.arange(graph.number_of_nodes()).view(-1, 1).float()  # Dummy features
    return Data(x=x, edge_index=edge_index)

graph_pairs = []
graph_labels = []

functions = list(graphs.keys())

for i in range(len(functions)):
    for j in range(i + 1, len(functions)):
        graph1 = create_graph_data(graphs[functions[i]])
        graph2 = create_graph_data(graphs[functions[j]])

        # Assume they are the same if function names match (ignoring suffixes)
        label = 1 if functions[i].split('_')[0] == functions[j].split('_')[0] else 0

        graph_pairs.append((graph1, graph2))
        graph_labels.append(label)
