import pytest

from edgetrain.calculate_priorities import define_priorities


def test_define_priorities_with_default_priorities():
    # Test default priorities with normalized scores
    normalized_scores = {"memory_score": 0.8, "accuracy_score": 0.4}

    priority_value = define_priorities(normalized_scores)

    # Default priorities: batch_size: 0.4, accuracy_improvement: 0.6
    assert priority_value["batch_size"] == pytest.approx(
        0.32, rel=1e-3
    ), "Batch size priority calculation failed."
    assert priority_value["learning_rate"] == pytest.approx(
        0.24, rel=1e-3
    ), "Learning rate priority calculation failed."


def test_define_priorities_with_custom_priorities():
    # Test user-defined priorities with normalized scores
    normalized_scores = {"memory_score": 0.5, "accuracy_score": 0.7}
    user_priorities = {"batch_size_adjustment": 0.8, "accuracy_improvement": 0.2}

    priority_value = define_priorities(normalized_scores, user_priorities)

    # Custom priorities: batch_size: 0.8, accuracy_improvement: 0.2
    assert priority_value["batch_size"] == pytest.approx(
        0.4, rel=1e-3
    ), "Batch size priority with custom priorities failed."
    assert priority_value["learning_rate"] == pytest.approx(
        0.14, rel=1e-3
    ), "Learning rate priority with custom priorities failed."


def test_define_priorities_with_zero_scores():
    # Test edge case where all normalized scores are zero
    normalized_scores = {"memory_score": 0.0, "accuracy_score": 0.0}

    priority_value = define_priorities(normalized_scores)

    assert (
        priority_value["batch_size"] == 0.0
    ), "Batch size priority with zero scores failed."
    assert (
        priority_value["learning_rate"] == 0.0
    ), "Learning rate priority with zero scores failed."


def test_define_priorities_with_extreme_scores():
    # Test edge case with extreme normalized scores
    normalized_scores = {"memory_score": 1.0, "accuracy_score": 1.0}

    priority_value = define_priorities(normalized_scores)

    assert (
        priority_value["batch_size"] == 0.40
    ), "Batch size priority with extreme scores failed."
    assert (
        priority_value["learning_rate"] == 0.60
    ), "Learning rate priority with extreme scores failed."
