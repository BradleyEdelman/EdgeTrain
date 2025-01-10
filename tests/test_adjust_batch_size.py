from edgetrain import adjust_batch_size

def test_adjust_batch_size_high_usage():
    # Mock system resources with high memory usage
    mock_resources = {'cpu_memory_percent': 85, 'gpu_memory_percent': 90}
    batch_size = adjust_batch_size(32, resources=mock_resources)
    assert batch_size < 32, f"Expected batch size to decrease under high memory usage, but got {batch_size}."

def test_adjust_batch_size_low_usage():
    # Mock system resources with low memory usage
    mock_resources = {'cpu_memory_percent': 15, 'gpu_memory_percent': 10}
    batch_size = adjust_batch_size(32, resources=mock_resources)
    assert batch_size > 32, f"Expected batch size to increase under low memory usage, but got {batch_size}."

def test_adjust_batch_size_normal_usage():
    # Mock system resources with normal memory usage
    mock_resources = {'cpu_memory_percent': 50, 'gpu_memory_percent': 50}
    batch_size = adjust_batch_size(32, resources=mock_resources)
    assert batch_size == 32, f"Expected batch size to remain unchanged under normal memory usage, but got {batch_size}."

def test_adjust_batch_size_min_limit():
    # Mock system resources with high memory usage, ensuring batch size doesn't go below the minimum
    mock_resources = {'cpu_memory_percent': 85, 'gpu_memory_percent': 90}
    batch_size = adjust_batch_size(8, resources=mock_resources)
    assert batch_size == 8, f"Expected batch size to remain at 8 (min limit), but got {batch_size}."

def test_adjust_batch_size_max_limit():
    # Mock system resources with low memory usage, ensuring batch size doesn't exceed the maximum
    mock_resources = {'cpu_memory_percent': 10, 'gpu_memory_percent': 10}
    batch_size = adjust_batch_size(128, resources=mock_resources)
    assert batch_size == 128, f"Expected batch size to remain at 128 (max limit), but got {batch_size}."
