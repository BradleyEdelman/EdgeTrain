import pytest
from unittest.mock import patch
from your_module import adjust_batch_size  # Replace 'your_module' with the actual module name

# Test adjust_batch_size function
def test_adjust_batch_size_increase(mocker):
    # Mock sys_resources function to return controlled values for low memory usage
    mocker.patch('your_module.sys_resources', return_value={
        "cpu_memory_percent": 10,
        "gpu_memory_percent": 10
    })
    
    # Call adjust_batch_size to increase batch size
    result = adjust_batch_size(16, cpu_threshold=[20, 80], gpu_threshold=[20, 80], increment=8)
    
    # Assertions
    assert result == 24  # 16 + increment (8)

def test_adjust_batch_size_decrease(mocker):
    # Mock sys_resources function to return controlled values for high memory usage
    mocker.patch('your_module.sys_resources', return_value={
        "cpu_memory_percent": 90,
        "gpu_memory_percent": 90
    })
    
    # Call adjust_batch_size to decrease batch size
    result = adjust_batch_size(32, cpu_threshold=[20, 80], gpu_threshold=[20, 80], increment=8)
    
    # Assertions
    assert result == 24  # 32 - increment (8)

def test_adjust_batch_size_no_change(mocker):
    # Mock sys_resources function to return controlled values for normal memory usage
    mocker.patch('your_module.sys_resources', return_value={
        "cpu_memory_percent": 50,
        "gpu_memory_percent": 50
    })
    
    # Call adjust_batch_size to leave batch size unchanged
    result = adjust_batch_size(32, cpu_threshold=[20, 80], gpu_threshold=[20, 80], increment=8)
    
    # Assertions
    assert result == 32  # No change
