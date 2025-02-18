def detect_pid(graph, gnn, threshold=0.5):
    emb = gnn(create_graph_data(graph))
    
    # Compare with known PID embeddings
    pid_embeddings = [gnn(create_graph_data(graphs[name])) for name in known_pid_functions]
    
    distances = [F.pairwise_distance(emb, pid_emb).item() for pid_emb in pid_embeddings]
    return min(distances) < threshold  # If close to known PID controllers, classify as PID

# Example usage:
is_pid = detect_pid(graphs["some_function"], gnn)
print("PID Controller Detected!" if is_pid else "Not a PID Controller.")
