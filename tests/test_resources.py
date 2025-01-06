from scaleml import resources
import GPUtil, psutil

def test_cpu_cores():
    detected_resources = resources()
    logical_cores = psutil.cpu_count(logical=True)
    assert logical_cores > 0, "Logical CPU cores should be greater than 0"

def test_gpu_devices():
    detected_resources = resources()
    gpu_devices = GPUtil.getGPUs()
    detected_gpus = [dev for dev in detected_resources if dev.startswith('/gpu')]
    assert len(detected_gpus) == len(gpu_devices), "Mismatch in detected GPU devices"

def test_all_devices_listed():
    detected_resources = resources()
    logical_cores = psutil.cpu_count(logical=True)
    gpu_devices = GPUtil.getGPUs()
    total_devices = logical_cores + len(gpu_devices)
    assert len(detected_resources) == total_devices, "Not all devices are listed"