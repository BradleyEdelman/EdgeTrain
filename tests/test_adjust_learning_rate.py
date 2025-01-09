from unittest.mock import patch
from edgetrain import adjust_learning_rate

def test_adjust_learning_rate_high_usage():
    # Mock system resources with high compute usage
    mock_resources = {'cpu_compute_percent': 85, 'gpu_compute_percent': 90}
    with patch('edgetrain.sys_resources', return_value=mock_resources):
        adjusted_lr = adjust_learning_rate(0.05)
        assert adjusted_lr < 0.05, "Learning rate should decrease under high resource usage."

def test_adjust_learning_rate_low_usage():
    # Mock system resources with low compute usage
    mock_resources = {'cpu_compute_percent': 10, 'gpu_compute_percent': 15}
    with patch('edgetrain.sys_resources', return_value=mock_resources):
        adjusted_lr = adjust_learning_rate(0.05)
        assert adjusted_lr > 0.05, "Learning rate should increase under low resource usage."

def test_adjust_learning_rate_normal_usage():
    # Mock system resources with normal compute usage
    mock_resources = {'cpu_compute_percent': 50, 'gpu_compute_percent': 50}
    with patch('edgetrain.sys_resources', return_value=mock_resources):
        adjusted_lr = adjust_learning_rate(0.05)
        assert adjusted_lr == 0.05, "Learning rate should remain unchanged under normal resource usage."

def test_adjust_learning_rate_min_limit():
    # Mock system resources with high compute usage and ensure learning rate doesn't drop below min
    mock_resources = {'cpu_compute_percent': 85, 'gpu_compute_percent': 90}
    with patch('edgetrain.sys_resources', return_value=mock_resources):
        adjusted_lr = adjust_learning_rate(1e-6)  # Already at minimum
        assert adjusted_lr == 1e-6, "Learning rate should not go below the minimum limit."

def test_adjust_learning_rate_max_limit():
    # Mock system resources with low compute usage and ensure learning rate doesn't exceed max
    mock_resources = {'cpu_compute_percent': 10, 'gpu_compute_percent': 15}
    with patch('edgetrain.sys_resources', return_value=mock_resources):
        adjusted_lr = adjust_learning_rate(0.1)  # Already at maximum
        assert adjusted_lr == 0.1, "Learning rate should not exceed the maximum limit."
