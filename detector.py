import torch
import torch_scatter
from ProcessDataset import GNNModel
import json
import pyghidra
from tqdm import tqdm
from LocalConfig import GHIDRA_INSTALL_PATH

pyghidra.start(verbose=True, install_dir=GHIDRA_INSTALL_PATH)
from ghidra.util.task import ConsoleTaskMonitor
from ghidra.app.decompiler import DecompInterface, DecompileOptions
from ghidra.program.model.pcode import PcodeOp

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

        for function in tqdm(all_functions, desc="Processing Functions", leave=True, colour="blue", position=1):
            results = decomp.decompileFunction(function, 60, monitor)
            high_function = results.getHighFunction()
            
            # Extract graph from the function
            nodes, edges, edge_features = extract_graph_from_high_pcode(high_function)
            graph = format_graph_for_gnn(nodes, edges, edge_features, label=None)  # No label needed for inference
            
            # Perform detection
            with torch.no_grad():
                graph = graph.to('cuda')
                output = model(graph)  # Node embeddings of shape [num_nodes, output_dim]
                
                # Aggregate node embeddings into a graph-level embedding
                graph_embedding = torch.mean(output, dim=0, keepdim=True)  # Shape: [1, output_dim]
                
                # Predict the label
                logits = model.predict(graph_embedding)  # Shape: [1, num_classes]
                predicted_label = torch.argmax(logits, dim=1).item()  # Get the predicted class
                print(f"Function: {function.getName()}, Predicted Label: {predicted_label}")

