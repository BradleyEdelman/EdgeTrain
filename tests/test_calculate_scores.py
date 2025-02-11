import pytest

from edgetrain.calculate_scores import compute_scores, normalize_scores


def test_normalize_scores():
    # Test normalization of raw scores with specified ranges
    raw_scores = {"memory_score": 50, "accuracy_score": 0.5}
    score_ranges = {"memory_score_range": 100, "accuracy_score_range": 1}
    normalized = normalize_scores(raw_scores, score_ranges)
    assert normalized["memory_score"] == 0.5, "Memory score normalization failed."
    assert normalized["accuracy_score"] == 0.5, "Accuracy score normalization failed."


def test_compute_scores():
    # Mock system resource usage
    mock_resources = {"num_gpus": 1, "cpu_memory_percent": 60, "gpu_memory_percent": 40}

    # Test with a decrease in accuracy
    previous_accuracy = 0.8
    current_accuracy = 0.6
    scores = compute_scores(
        previous_accuracy, current_accuracy, score_ranges=None, resources=mock_resources
    )
    assert scores["memory_score"] == 0.5, "Memory score calculation failed with GPUs."
    assert scores["accuracy_score"] == pytest.approx(
        1, rel=1e-3
    ), "Accuracy score calculation failed."

    # Test with no GPUs
    mock_resources = {"num_gpus": 0, "cpu_memory_percent": 75}
    scores = compute_scores(
        previous_accuracy, current_accuracy, score_ranges=None, resources=mock_resources
    )
    assert (
        scores["memory_score"] == 0.75
    ), "Memory score calculation failed without GPUs."

    # Test edge case with accuracy stagnation
    previous_accuracy = 0.7
    current_accuracy = 0.7
    scores = compute_scores(
        previous_accuracy, current_accuracy, score_ranges=None, resources=mock_resources
    )
    assert scores["accuracy_score"] == pytest.approx(
        1, rel=1e-3
    ), "Accuracy score should be 1 for stagnation."

    # Test edge case where current accuracy is higher (clamped at 0)
    previous_accuracy = 0.6
    current_accuracy = 0.8
    scores = compute_scores(
        previous_accuracy, current_accuracy, score_ranges=None, resources=mock_resources
    )
    assert scores["accuracy_score"] == pytest.approx(
        0.8, rel=1e-3
    ), "Accuracy score should be 1 for decreasing accuracy."


def test_compute_scores_with_custom_ranges():
    # Mock resources
    mock_resources = {"num_gpus": 1, "cpu_memory_percent": 80, "gpu_memory_percent": 60}

    # Test custom ranges
    score_ranges = {"memory_score_range": 200, "accuracy_score_range": 0.5}

    previous_accuracy = 0.8
    current_accuracy = 0.6
    scores = compute_scores(
        previous_accuracy,
        current_accuracy,
        score_ranges=score_ranges,
        resources=mock_resources,
    )
    assert scores["memory_score"] == pytest.approx(
        0.35, rel=1e-3
    ), "Memory score normalization with custom range failed."
    assert scores["accuracy_score"] == pytest.approx(
        2.0, rel=1e-3
    ), "Accuracy score normalization with custom range failed."


def test_compute_scores_with_acc_improvement():
    # Mock resources
    mock_resources = {"num_gpus": 1, "cpu_memory_percent": 80, "gpu_memory_percent": 60}

    previous_accuracy = 0.6
    current_accuracy = 0.8
    scores = compute_scores(
        previous_accuracy, current_accuracy, score_ranges=None, resources=mock_resources
    )
    assert scores["accuracy_score"] == pytest.approx(
        0.80, rel=1e-3
    ), "Accuracy score normalization with custom range failed."
