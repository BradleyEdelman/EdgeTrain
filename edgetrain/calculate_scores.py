def compute_scores(system_resources, previous_accuracy, current_accuracy, previous_loss, current_loss, max_accuracy_range, max_loss_range):
    """
    Compute memory usage, accuracy stagnation, and loss stagnation scores, and normalize them.
    
    Parameters:
    - system_resources (dict): Dictionary containing system resource metrics (CPU, GPU memory usage).
    - previous_accuracy (float): Accuracy from the previous epoch.
    - current_accuracy (float): Current accuracy.
    - previous_loss (float): Loss from the previous epoch.
    - current_loss (float): Current loss.
    - max_accuracy_range (float): Maximum possible accuracy improvement.
    - max_loss_range (float): Maximum possible loss reduction.
    
    Returns:
    - normalized_scores (dict): Dictionary of normalized scores.
    """
    # Calculate memory usage score (average of CPU and GPU memory utilization)
    memory_usage_score = (system_resources["cpu_memory_percent"] + system_resources["gpu_memory_percent"]) / 2
    
    # Calculate accuracy stagnation score
    accuracy_stagnation_score = max(0, previous_accuracy - current_accuracy)
    
    # Calculate loss stagnation score
    loss_stagnation_score = max(0, current_loss - previous_loss)

    # Normalize the scores
    normalized_scores = normalize_scores(memory_usage_score, accuracy_stagnation_score, loss_stagnation_score, max_accuracy_range, max_loss_range)

    return normalized_scores
