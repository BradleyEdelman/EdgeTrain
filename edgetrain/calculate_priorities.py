def define_priorities(memory_usage_score, accuracy_stagnation_score, loss_stagnation_score, user_priorities=None):
    """
    Calculate priority scores for adjustments based on resource usage, accuracy, and loss.
    
    Parameters:
    - memory_usage_score (float): Score indicating memory usage pressure (0-1).
    - accuracy_stagnation_score (float): Score indicating stagnation in accuracy improvement (0-1).
    - loss_stagnation_score (float): Score indicating stagnation in loss reduction (0-1).
    - user_priorities (dict): Optional user-defined priorities for resource conservation, accuracy, and loss.
    
    Returns:
    - priority_scores (dict): A dictionary of priority scores for batch size, pruning, and learning rate.
    """
    # Default weights if user priorities are not provided
    default_priorities = {
        "resource_conservation": 0.4,
        "accuracy_improvement": 0.4,
        "loss_reduction": 0.2,
    }
    
    # Use user-defined priorities if available
    if user_priorities:
        priorities = user_priorities
    else:
        priorities = default_priorities

    # Calculate weighted priority scores
    priority_scores = {
        "batch_size": priorities["resource_conservation"] * memory_usage_score,
        "pruning": priorities["resource_conservation"] * memory_usage_score,
        "learning_rate": (
            priorities["accuracy_improvement"] * accuracy_stagnation_score +
            priorities["loss_reduction"] * loss_stagnation_score
        ),
    }
    
    return priority_scores
