import pandas as pd
import os
import magic
import pyghidra
from LocalConfig import GHIDRA_INSTALL_PATH
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import pyghidra

pyghidra.start(verbose=True, install_dir=GHIDRA_INSTALL_PATH)
from ghidra.util.task import ConsoleTaskMonitor
from ghidra.app.decompiler import DecompInterface, DecompileOptions
from ghidra.program.model.pcode import PcodeOp

import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader

# Define the GNN model
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        # Compute pairwise distance
        distance = torch.norm(z1 - z2, p=2, dim=1)
        loss = label * distance**2 + (1 - label) * torch.clamp(self.margin - distance, min=0)**2
        return loss.mean()

# Prepare the dataset
def prepare_dataset(positive_graphs, negative_graphs):
    dataset = []
    labels = []

    # Add positive graphs with label 1
    for graph in positive_graphs:
        dataset.append(graph)
        labels.append(1)

    # Add negative graphs with label 0
    for graph in negative_graphs:
        dataset.append(graph)
        labels.append(0)

    return dataset, torch.tensor(labels, dtype=torch.float)

# Training loop
def train_gnn(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        z = model(data)

        # Contrastive loss
        # Assume we have pairs of graphs (z1, z2) and their labels
        z1, z2 = z[:len(z)//2], z[len(z)//2:]
        labels = data.y[:len(z)//2]
        loss = criterion(z1, z2, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def start_training(positive_graphs, negative_graphs):
    # Prepare dataset
    dataset = []
    dataset.extend(positive_graphs)
    dataset.extend(negative_graphs)

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define model, optimizer, and loss function
    input_dim = dataset[0].x.size(1)  # Feature dimension
    hidden_dim = 64
    output_dim = 32
    model = GNNModel(input_dim, hidden_dim, output_dim).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = ContrastiveLoss(margin=1.0)

    # Train the model
    epochs = 20
    for epoch in range(epochs):
        loss = train_gnn(model, loader, optimizer, criterion, 'cuda')
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    print("Training complete!")

def extract_graph_from_high_pcode(high_function):
    """
    Extract a graph from the P-code level of a function.
    The graph includes:
    - Nodes for each P-code operation
    - Control flow edges between sequential P-code operations
    - Dataflow edges based on Varnode outputs and their descendants
    - Global variable inputs and outputs tracked using a dummy node (node 0)
    """
    nodes = []  # List of nodes
    edges = []  # List of edges (source, target)
    edge_features = []  # List of edge features
    global_node_id = 0  # Dummy node for global variables
    node_map = {}  # Map to track unique nodes for Varnodes

    try:
        # Add the global dummy node
        nodes.append({"id": global_node_id, "type": -1})

        pcode_ops = list(high_function.getPcodeOps())

        # Add P-code operation to nodes
        for i, op in enumerate(pcode_ops):
            op_mnem = op.getMnemonic()
            op_type = op.getOpcode()

            node_id = len(nodes)
            nodes.append({"id": node_id, "type": op_type})
            node_map[op] = node_id

        # Add edges between nodes based on P-code operations
        for i, op in enumerate(pcode_ops):
            op_mnem = op.getMnemonic()
            op_type = op.getOpcode()
            inputs = op.getInputs()
            output = op.getOutput()

            node_id = node_map[op]

            # Add control flow edges (branching operations)
            if op_type in [PcodeOp.BRANCH, PcodeOp.CBRANCH]:
                block = op.getParent()
                for successor_index in range(block.getOutSize()):
                    successor = block.getOut(successor_index)
                    edges.append([node_id, node_map[successor.getFirstOp()]])
                    edge_features.append([1, 0, 0])  # Control flow edge feature

            # Add control flow edges (sequential P-code operations)
            if i < len(pcode_ops) - 1 and op_type not in [PcodeOp.BRANCH, PcodeOp.CBRANCH]:
                edges.append([node_id, node_id + 1])
                edge_features.append([1, 0, 0])  # Control flow edge feature

            # Add dataflow edges for floating-point operations
            if "FLOAT" in op_mnem and output is not None:
                descendants = output.getDescendants()
                for descendant in descendants:
                    edges.append([node_id, node_map[descendant]])
                    edge_features.append([0, 1, 0])  # Dataflow edge feature

            # Track global variable inputs and outputs
            if output is not None and output.isAddress():
                edges.append([node_id, global_node_id])
                edge_features.append([0, 0, 1])  # Global output edge feature
            for inp in inputs:
                if inp.isAddress():
                    edges.append([global_node_id, node_id])
                    edge_features.append([0, 0, 1])  # Global input edge feature

    except Exception as e:
        print(f"[!] Error extracting graph: {e}")

    return nodes, edges, edge_features


def format_graph_for_gnn(nodes, edges, edge_features, label):
    """
    Format the extracted graph into a PyTorch Geometric Data object.
    """
    # Create node features (e.g., one-hot encoding of operation types)
    node_features = torch.tensor([[node["type"] for node in nodes]], dtype=torch.float).t().contiguous()

    # Create edge indices
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Create edge feature matrix
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # Combine everything into a PyTorch Geometric Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.float) 
    )
    
    return data

# MIME types considered executable
EXECUTABLE_MIME_TYPES = {
    'application/x-executable',
    'application/x-dosexec',         # Windows .exe
    'application/x-elf',             # Linux ELF
    'application/x-mach-binary',     # macOS Mach-O
    'application/octet-stream'       # Generic binary blob
}

def is_executable_file(filepath):
    try:
        mime_type = magic_checker.from_file(filepath)
        return mime_type in EXECUTABLE_MIME_TYPES
    except Exception as e:
        print(f"[!] MIME check failed for {filepath}: {e}")
        return False

def flatten_path(path):
    """Flatten a file path by replacing slashes with underscores."""
    return path.replace(os.sep, "_").replace("/", "_")

def import_and_process_binaries(binary_dir, functions_to_find):
    """
    Process all binaries in the directory and find the closest function names using LCS similarity.
    """
    positive_graphs = []
    negative_graphs = []

    for root, _, files in os.walk(binary_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if not is_executable_file(file_path):
                continue
            
            try:
                project_name = flatten_path(file_path)
                print(f"[*] Importing {file_path} with project name {project_name}")
                with pyghidra.open_program(file_path,
                                           project_location="ghidra_files",
                                           project_name=project_name) as flat_api:
                    program = flat_api.getCurrentProgram()
                    function_manager = program.getFunctionManager()
                    all_functions = function_manager.getFunctions(True)
                    decomp = DecompInterface()
                    decomp.openProgram(program)
                    decomp.setOptions(DecompileOptions())
                    monitor = ConsoleTaskMonitor()

                    for function in tqdm(all_functions, desc="Processing Functions", leave=True, colour="blue", position=1):
                        positive = False
                        for target_function in functions_to_find:
                            if target_function in function.getName():
                                positive = True
                                break
                    
                        results = decomp.decompileFunction(function, 60, monitor)
                        high_function = results.getHighFunction()
                        nodes, edges, edge_features = extract_graph_from_high_pcode(high_function)
                        graph = format_graph_for_gnn(nodes, edges, edge_features, label=1 if positive else 0)
                        
                        if positive:
                            positive_graphs.append(graph)
                        else:
                            negative_graphs.append(graph)
                        

                    print(f"[+] Processed binary: {file_path}")
            except Exception as e:
                print(f"[!] Failed to import {file_path}: {e}")

    return positive_graphs, negative_graphs

if __name__ == "__main__":

    csv_file = "dataset.csv"
    df = pd.read_csv(csv_file)
    magic_checker = magic.Magic(mime=True)
    positive_graphs, negative_graphs = [], []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Binaries", leave=True, colour="green", position=0):
        BINARY_DIR = f"Binaries/{row['Name']}"
        
        name = row['Name']        
        # find elements in row that are not 'Name' or None
        functions_to_find = [row[col] for col in df.columns if col != 'Name' and pd.notna(row[col])]

        print("[*] Importing %s" % BINARY_DIR)
        pos, neg = import_and_process_binaries(BINARY_DIR, functions_to_find)
        positive_graphs.extend(pos)
        negative_graphs.extend(neg)
        break

    start_training(positive_graphs, negative_graphs)
