import pytest
from unittest import mock
from your_module import sys_resources  # Replace 'your_module' with the actual module name

# Mock the psutil and GPUtil modules
@pytest.fixture
def mock_psutil_and_gputil():
    with mock.patch("your_module.psutil.cpu_percent") as mock_cpu_percent, \
         mock.patch("your_module.psutil.cpu_count") as mock_cpu_count, \
         mock.patch("your_module.psutil.virtual_memory") as mock_virtual_memory, \
         mock.patch("your_module.GPUtil.getGPUs") as mock_get_gpus, \
         mock.patch("your_module.nvmlInit") as mock_nvml_init, \
         mock.patch("your_module.nvmlShutdown") as mock_nvml_shutdown, \
         mock.patch("your_module.nvmlDeviceGetUtilizationRates") as mock_nvml_device_utilization:

        # Setup mock return values for psutil
        mock_cpu_percent.return_value = 50.0
        mock_cpu_count.return_value = 8
        mock_virtual_memory.return_value.percent = 75.0
        
        # Setup mock return values for GPUtil
        mock_get_gpus.return_value = [
            mock.Mock(memoryUsed=1000, memoryTotal=8000, memoryUtil=12),
            mock.Mock(memoryUsed=1200, memoryTotal=8000, memoryUtil=15),
        ]
        
        # Setup mock return values for nvml functions
        mock_nvml_device_utilization.return_value.gpu = 60
        
        yield {
            "mock_cpu_percent": mock_cpu_percent,
            "mock_cpu_count": mock_cpu_count,
            "mock_virtual_memory": mock_virtual_memory,
            "mock_get_gpus": mock_get_gpus,
            "mock_nvml_init": mock_nvml_init,
            "mock_nvml_shutdown": mock_nvml_shutdown,
            "mock_nvml_device_utilization": mock_nvml_device_utilization
        }

def test_sys_resources(mock_psutil_and_gputil):
    # Call the sys_resources function
    result = sys_resources()

    # Check if the function returns a dictionary with the expected keys
    assert "cpu_cores" in result
    assert "cpu_compute_percent" in result
    assert "cpu_memory_percent" in result
    assert "gpu_compute_percent" in result
    assert "gpu_memory_usage" in result
    assert "gpu_memory_total" in result
    assert "gpu_memory_percent" in result
    assert "num_gpus" in result

    # Validate the values returned by the mock
    assert result["cpu_cores"] == 8
    assert result["cpu_compute_percent"] == 50.0
    assert result["cpu_memory_percent"] == 75.0
    assert result["gpu_compute_percent"] == 60.0
    assert result["gpu_memory_usage"] == 2200
    assert result["gpu_memory_total"] == 16000
    assert result["gpu_memory_percent"] == 13.5
    assert result["num_gpus"] == 2

