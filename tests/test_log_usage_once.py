import pytest
from unittest.mock import patch, mock_open
from your_module import log_usage_once  # Replace 'your_module' with the actual module name

# Test log_usage_once function
def test_log_usage_once(mocker):
    # Mock sys_resources function to return controlled values
    mocker.patch('your_module.sys_resources', return_value={
        "cpu_cores": 8,
        "cpu_compute_percent": 50,
        "cpu_memory_percent": 70,
        "gpu_compute_percent": 60,
        "gpu_memory_usage": 2000,
        "gpu_memory_total": 8000,
        "gpu_memory_percent": 40,
        "num_gpus": 1
    })

    # Mock file operations
    mock_file = mock_open()
    with patch('builtins.open', mock_file):
        log_usage_once('test_log.csv', batch_size=32, num_workers=4, num_epoch=1)

    # Assert file is written
    mock_file.assert_called_once_with('test_log.csv', 'a', newline='')

    # Extract the first write call
    handle = mock_file()
    handle.write.assert_any_call(','.join(map(str, [
        '2025-01-01 12:00:00',  # Example timestamp
        1,  # Epoch 1
        50,  # CPU compute percent
        70,  # CPU memory percent
        60,  # GPU compute percent
        40,  # GPU memory percent
        32,  # Batch size
        4   # Num workers
    ])) + '\n')
