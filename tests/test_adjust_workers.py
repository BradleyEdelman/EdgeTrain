import pytest
from unittest.mock import patch
from your_module import adjust_workers  # Replace 'your_module' with the actual module name

# Test adjust_workers function
def test_adjust_workers_increase(mocker):
    # Mock sys_resources function to return controlled values for low CPU and GPU usage
    mocker.patch('your_module.sys_resources', return_value={
        "cpu_compute_percent": 10,
        "gpu_compute_percent": 10,
        "cpu_cores": 8
    })
    
    # Call adjust_workers to increase workers
    result = adjust_workers(2, cpu_threshold=[20, 80], gpu_threshold=[20, 80], increment=1)
    
    # Assertions
    assert result == 3  # 2 + increment (1)

def test_adjust_workers_decrease(mocker):
    # Mock sys_resources function to return controlled values for high CPU and GPU usage
    mocker.patch('your_module.sys_resources', return_value={
        "cpu_compute_percent": 90,
        "gpu_compute_percent": 90,
        "cpu_cores": 8
    })
    
    # Call adjust_workers to decrease workers
    result = adjust_workers(4, cpu_threshold=[20, 80], gpu_threshold=[20, 80], increment=1)
    
    # Assertions
    assert result == 3  # 4 - increment (1)

def test_adjust_workers_no_change(mocker):
    # Mock sys_resources function to return controlled values for normal CPU and GPU usage
    mocker.patch('your_module.sys_resources', return_value={
        "cpu_compute_percent": 50,
        "gpu_compute_percent": 50,
        "cpu_cores": 8
    })
    
    # Call adjust_workers to leave workers unchanged
    result = adjust_workers(4, cpu_threshold=[20, 80], gpu_threshold=[20, 80], increment=1)
    
    # Assertions
    assert result == 4  # No change
