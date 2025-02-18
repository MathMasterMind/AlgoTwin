# Description: This script reads the pcode.json file and converts it into a graph representation using the networkx library

import json
import networkx as nx

def parse_pcode_to_graph(pcode_file):
    with open(pcode_file, "r") as f:
        functions_pcode = json.load(f)

    function_graphs = {}

    for func_name, pcode_ops in functions_pcode.items():
        G = nx.DiGraph()
        for idx, op in enumerate(pcode_ops):
            G.add_node(idx, operation=op)
            if idx > 0:
                G.add_edge(idx - 1, idx)  # Sequential dependency
        function_graphs[func_name] = G

    return function_graphs

graphs = parse_pcode_to_graph("/path/to/save/pcode.json")
