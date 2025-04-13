import torch
from ProcessDataset import GNNModel
import json
import pyghidra
from tqdm import tqdm
from LocalConfig import GHIDRA_INSTALL_PATH

pyghidra.start(verbose=True, install_dir=GHIDRA_INSTALL_PATH)
from ghidra.util.task import ConsoleTaskMonitor
from ghidra.app.decompiler import DecompInterface, DecompileOptions

from ProcessDataset import extract_graph_from_high_pcode, format_graph_for_gnn

def load_model_and_dimensions(model_path, dimensions_path):
    # Load the model dimensions from the JSON file
    with open(dimensions_path, "r") as f:
        dimensions = json.load(f)

    input_dim = dimensions["input_dim"]
    hidden_dim = dimensions["hidden_dim"]
    output_dim = dimensions["output_dim"]

    # Define the model architecture
    model = GNNModel(input_dim, hidden_dim, output_dim).to('cuda')

    # Load the saved state dictionary
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    print(f"Model loaded from {model_path}")
    print(f"Dimensions loaded from {dimensions_path}: {dimensions}")

    return model

def compute_closest_positive_embeddings(model, functions, total, positive_embeddings_path):
    # Load positive embeddings
    with open(positive_embeddings_path, "r") as f:
        positive_embeddings = torch.tensor(json.load(f))  # Shape: [num_positive, output_dim]

    function_distances = []

    for function in tqdm(functions, total=total, desc="Processing Functions", leave=True, colour="blue", position=1):
        results = decomp.decompileFunction(function, 60, monitor)
        high_function = results.getHighFunction()

        # Extract graph from the function
        nodes, edges, edge_features = extract_graph_from_high_pcode(high_function)
        graph = format_graph_for_gnn(nodes, edges, edge_features, label=1)  # No label needed for inference

        # Perform detection
        with torch.no_grad():
            graph = graph.to('cuda')
            output = model(graph)  # Node embeddings of shape [num_nodes, output_dim]

            # Aggregate node embeddings into a graph-level embedding
            graph_embedding = torch.mean(output, dim=0, keepdim=True)  # Shape: [1, output_dim]

            # Compute distances to all positive embeddings
            distances = torch.norm(positive_embeddings - graph_embedding.cpu(), p=2, dim=1)  # Euclidean distance

            # Find the closest positive embedding
            min_distance = torch.min(distances).item()
            function_distances.append((function.getSignature(), min_distance))

    # Sort functions by distance
    function_distances.sort(key=lambda x: x[1])

    return function_distances

if __name__ == "__main__":
    # Load the GNN model and dimensions
    model = load_model_and_dimensions("trained_gnn_model.pth", "model_dimensions.json")

    # Load the binary file
    binary_file_path = "arducopter"
    with pyghidra.open_program(binary_file_path,
                                           project_location="ghidra_files",
                                           project_name="arducopter") as flat_api:
        
        program = flat_api.getCurrentProgram()
        function_manager = program.getFunctionManager()
        all_functions = function_manager.getFunctions(True)
        decomp = DecompInterface()
        decomp.openProgram(program)
        decomp.setOptions(DecompileOptions())
        monitor = ConsoleTaskMonitor()

        sorted_functions = compute_closest_positive_embeddings(model, all_functions, function_manager.getFunctionCount(), "positive_embeddings.json")

        # save results to a file
        with open("closest_functions.txt", "w") as f:
            for function_name, distance in sorted_functions:
                f.write(f"Function: {function_name}, Distance: {distance}\n")

        # print top 10 closest functions
        print("Top 10 closest functions:")
        for function_name, distance in sorted_functions[:10]:
            print(f"Function: {function_name}, Distance: {distance}")
