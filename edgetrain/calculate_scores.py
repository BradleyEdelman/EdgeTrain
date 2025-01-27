from edgetrain import sys_resources

def compute_scores(previous_accuracy, current_accuracy, score_ranges=None):
    """
    Compute memory, accuracy, and loss scores, and normalize them.
    
    Parameters:
    - previous_accuracy (float): Accuracy from the previous epoch.
    - current_accuracy (float): Current accuracy.
    - score_ranges (float): Maximum possible accuracy improvement.
    
    Returns:
    - normalized_scores (dict): Dictionary of normalized scores.
    """

    # Get system resources
    resources = sys_resources()
    
    # Default score ranges
    if score_ranges is None:
        score_ranges = {
            "memory_score_range": 100,  # Default 0-100 range for memory score
            "accuracy_score_range": 1,  # Default 0-1 range for accuracy score
        }
    
    # Calculate memory score
    # # If there is a gpu average gpu and cpu for memory score, otherwise, just use cpu
    if resources.get('num_gpus') > 0:
        memory_score = (resources.get('cpu_memory_percent') + resources.get('gpu_memory_percent')) / 2
    else:
        memory_score = resources.get('cpu_memory_percent')

    # Calculate accuracy score
    accuracy_score = max(0, previous_accuracy - current_accuracy)

    # store all three scores in a dictionary
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
        score_range = score_ranges.get(score_name, 1)  # Default range is 1 if not specified
        normalized_score = score_value / score_range
        normalized_scores[score_name] = normalized_score
    
    return normalized_scores