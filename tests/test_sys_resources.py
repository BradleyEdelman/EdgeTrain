import pytest
from unittest.mock import patch
from your_module import sys_resources  # Replace 'your_module' with the actual module name

# Test sys_resources function
def test_sys_resources(mocker):
    # Mock CPU usage and count
    mocker.patch('psutil.cpu_percent', return_value=50)
    mocker.patch('psutil.cpu_count', return_value=8)
    mocker.patch('psutil.virtual_memory', return_value=mocker.Mock(percent=70))

    # Mock GPU usage
    mock_gpu = mocker.Mock(memoryUsed=2000, memoryTotal=8000, memoryUtil=40)
    mocker.patch('GPUtil.getGPUs', return_value=[mock_gpu])

    # Mock NVML GPU compute utilization
    mocker.patch('pynvml.nvmlInit', return_value=None)
    mocker.patch('pynvml.nvmlShutdown', return_value=None)
    mocker.patch('pynvml.nvmlDeviceGetUtilizationRates', return_value=mocker.Mock(gpu=60))

    # Call sys_resources function
    result = sys_resources()

    # Assertions
    assert result["cpu_cores"] == 8
    assert result["cpu_compute_percent"] == 50
    assert result["cpu_memory_percent"] == 70
    assert result["gpu_compute_percent"] == 60
    assert result["gpu_memory_usage"] == 2000
    assert result["gpu_memory_total"] == 8000
    assert result["gpu_memory_percent"] == 40
    assert result["num_gpus"] == 1
