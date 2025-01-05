
import pytest
from scaleml.resources import resources
import tensorflow as tf
import psutil

def test_cpu_cores():
    devices = resources()
    logical_cores = psutil.cpu_count(logical=True)
    assert logical_cores > 0, "Logical CPU cores should be greater than 0"

def test_gpu_devices():
    devices = resources()
    gpu_devices = tf.config.list_physical_devices('GPU')
    detected_gpus = [dev for dev in devices if dev.startswith('/gpu')]
    assert len(detected_gpus) == len(gpu_devices), "Mismatch in detected GPU devices"

def test_all_devices_listed():
    devices = resources()
    logical_cores = psutil.cpu_count(logical=True)
    gpu_devices = tf.config.list_physical_devices('GPU')
    total_devices = logical_cores + len(gpu_devices)
    assert len(devices) == total_devices, "Not all devices are listed"
