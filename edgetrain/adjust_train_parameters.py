from edgetrain.resource_monitor import sys_resources


def adjust_training_parameters(
    priority_values, batch_size, lr, accuracy_score, resources=None
):
    """
    Adjust the training parameters (batch size, learning rate) based on the highest priority score,
    moving parameters in the opposite direction if resource usage or accuracy trends improve.

    Parameters:
    - priority_values (dict): Dictionary containing priority scores for batch size, pruning, and learning rate.
    - batch_size (int): Current batch size.
    - lr (float): Current learning rate.
    - accuracy_score (float): Current accuracy score from the latest epoch (0-1).

    Returns:
    - adjusted_batch_size (int): Adjusted batch size.
    - adjusted_lr (float): Adjusted learning rate.
    """

    # Get system resources
    if resources is None:
        resources = sys_resources()

    # Determine which parameter has the highest priority score
    highest_priority = max(priority_values, key=priority_values.get)

    # Adjust the parameter based on system resources and highest priority score
    if highest_priority == "batch_size":
        # Adjust batch size based on memory usage
        if resources["cpu_memory_percent"] > 75 or resources["gpu_memory_percent"] > 75:
            adjusted_batch_size = max(16, batch_size // 2)  # Halve batch size
        elif (
            resources["cpu_memory_percent"] < 50
            and resources["gpu_memory_percent"] < 50
        ):
            adjusted_batch_size = min(128, batch_size * 2)  # Double batch size
        else:
            adjusted_batch_size = batch_size
        adjusted_lr = lr

    elif highest_priority == "learning_rate":
        # Adjust learning rate based on accuracy score
        if accuracy_score < 0.05:  # Example threshold for low accuracy
            adjusted_lr = max(1e-5, lr * 0.5)  # Reduce learning rate
        elif accuracy_score > 0.95:  # Example threshold for high accuracy
            adjusted_lr = min(1e-2, lr * 1.2)  # Slightly increase learning rate
        else:
            adjusted_lr = lr
        adjusted_batch_size = batch_size

    return adjusted_batch_size, adjusted_lr
