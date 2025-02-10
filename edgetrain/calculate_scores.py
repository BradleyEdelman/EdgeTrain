from edgetrain import sys_resources

def compute_scores(previous_accuracy, current_accuracy, score_ranges=None, resources=None):
    """
    Compute memory and accuracy scores, and normalize them.
    
    Parameters:
    - previous_accuracy (float): Accuracy from the previous epoch.
    - current_accuracy (float): Current accuracy.
    - score_ranges (dict, optional): Dictionary of maximum possible improvements for each score.
    - resources (dict, optional): Dictionary containing system resource usage metrics. If None, system resources will be fetched.
    
    Returns:
    - normalized_scores (dict): Dictionary of normalized scores.
    """

    # Get system resources
    if resources is None:
        resources = sys_resources()
    
    # Default score ranges
    if score_ranges is None:
        score_ranges = {
            "memory_score_range": 100,  # Default 0-100 range for memory score
            "accuracy_score_range": 1,  # Default 0-1 range for accuracy score
        }
    
    # Calculate memory score
    # If there is a GPU, average GPU and CPU for memory score, otherwise, just use CPU
    if resources.get('num_gpus') > 0:
        memory_score = (resources.get('cpu_memory_percent') + resources.get('gpu_memory_percent')) / 2
    else:
        memory_score = resources.get('cpu_memory_percent')

    # Calculate accuracy score
    accuracy_score = 1 - max(0, current_accuracy - previous_accuracy)

    # Store all scores in a dictionary
    raw_scores = {
        "memory_score": memory_score,
        "accuracy_score": accuracy_score
    }

    # Normalize the scores
    normalized_scores = normalize_scores(raw_scores, score_ranges)

    return normalized_scores


def normalize_scores(raw_scores, score_ranges):
    """
    Normalize raw scores based on predefined score ranges.
    
    Parameters:
    - raw_scores (dict): Dictionary of raw scores.
    - score_ranges (dict): Dictionary of maximum possible improvements for each score.
    
    Returns:
    - normalized_scores (dict): Dictionary of normalized scores.
    """
    normalized_scores = {}
    
    for score_name, score_value in raw_scores.items():
        score_range = score_ranges.get(f'{score_name}_range', 1)  # Default range is 1 if not specified
        normalized_score = score_value / score_range
        normalized_scores[score_name] = normalized_score
    
    return normalized_scores