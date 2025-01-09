from unittest.mock import patch
from edgetrain import adjust_batch_size

def test_adjust_batch_size_high_usage():
    # Mock system resources with high memory usage
    mock_resources = {'cpu_memory_percent': 85, 'gpu_memory_percent': 90}
    with patch('edgetrain.sys_resources', return_value=mock_resources):
        batch_size = adjust_batch_size(32)
        assert batch_size < 32, "Batch size should decrease under high memory usage."

def test_adjust_batch_size_low_usage():
    # Mock system resources with low memory usage
    mock_resources = {'cpu_memory_percent': 15, 'gpu_memory_percent': 10}
    with patch('edgetrain.sys_resources', return_value=mock_resources):
        batch_size = adjust_batch_size(32)
        assert batch_size > 32, "Batch size should increase under low memory usage."

def test_adjust_batch_size_normal_usage():
    # Mock system resources with normal memory usage
    mock_resources = {'cpu_memory_percent': 50, 'gpu_memory_percent': 50}
    with patch('edgetrain.sys_resources', return_value=mock_resources):
        batch_size = adjust_batch_size(32)
        assert batch_size == 32, "Batch size should remain unchanged under normal memory usage."

def test_adjust_batch_size_min_limit():
    # Mock system resources with high memory usage, ensuring batch size doesn't go below the minimum
    mock_resources = {'cpu_memory_percent': 85, 'gpu_memory_percent': 90}
    with patch('edgetrain.sys_resources', return_value=mock_resources):
        batch_size = adjust_batch_size(8)  # Already at minimum
        assert batch_size == 8, "Batch size should not go below the minimum limit."

def test_adjust_batch_size_max_limit():
    # Mock system resources with low memory usage, ensuring batch size doesn't exceed the maximum
    mock_resources = {'cpu_memory_percent': 10, 'gpu_memory_percent': 10}
    with patch('edgetrain.sys_resources', return_value=mock_resources):
        batch_size = adjust_batch_size(128)  # Already at maximum
        assert batch_size == 128, "Batch size should not exceed the maximum limit."
