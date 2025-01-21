def adjust_training_parameters(priority_scores, batch_size, pruning_ratio, lr, system_resources):
    """
    Adjust the training parameters (batch size, pruning ratio, learning rate) based on the highest priority score.
    
    Parameters:
    - priority_scores (dict): Dictionary containing priority scores for batch size, pruning, and learning rate.
    - batch_size (int): Current batch size.
    - pruning_ratio (float): Current pruning ratio.
    - lr (float): Current learning rate.
    - system_resources (dict): System resource usage data (CPU and GPU).
    
    Returns:
    - adjusted_batch_size (int): Adjusted batch size.
    - adjusted_pruning_ratio (float): Adjusted pruning ratio.
    - adjusted_lr (float): Adjusted learning rate.
    """
    # Determine which parameter has the highest priority score
    highest_priority = max(priority_scores, key=priority_scores.get)
    
    # Adjust the parameter based on system resources and highest priority score
    if highest_priority == "batch_size":
        # Only adjust batch size if memory usage is high
        if system_resources["cpu_memory_percent"] > 75 or system_resources["gpu_memory_percent"] > 75:
            adjusted_batch_size = max(16, batch_size // 2)  # Halve batch size
        else:
            adjusted_batch_size = batch_size
        adjusted_pruning_ratio = pruning_ratio
        adjusted_lr = lr
    
    elif highest_priority == "pruning":
        # Adjust pruning ratio if memory usage is high
        if system_resources["cpu_memory_percent"] > 75 or system_resources["gpu_memory_percent"] > 75:
            adjusted_pruning_ratio = min(0.8, pruning_ratio + 0.1)  # Increase pruning
        else:
            adjusted_pruning_ratio = pruning_ratio
        adjusted_batch_size = batch_size
        adjusted_lr = lr
    
    elif highest_priority == "learning_rate":
        # Adjust learning rate based on accuracy stagnation
        if accuracy_stagnation_score > 0.05:  # Example threshold for stagnation
            adjusted_lr = max(1e-5, lr * 0.5)  # Reduce learning rate if stagnation detected
        else:
            adjusted_lr = lr
        adjusted_batch_size = batch_size
        adjusted_pruning_ratio = pruning_ratio
    
    return adjusted_batch_size, adjusted_pruning_ratio, adjusted_lr
