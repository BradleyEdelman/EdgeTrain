import pytest
from unittest.mock import patch, MagicMock
import tensorflow as tf
from your_module import adjust_pruning  # Replace `your_module` with the actual module name.

# Mock sys_resources to simulate CPU and GPU memory usage
@pytest.fixture
def mock_sys_resources():
    with patch("your_module.sys_resources") as mock:
        yield mock

# Helper function to create a simple model
@pytest.fixture
def simple_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(10,)),
        tf.keras.layers.Dense(1)
    ])
    return model

def test_pruning_ratio_increase(mock_sys_resources, simple_model):
    mock_sys_resources.return_value = {
        'cpu_memory_percent': 85,  # Above the high threshold
        'gpu_memory_percent': 85
    }
    model = simple_model
    initial_pruning_ratio = 0.1
    _, new_pruning_ratio = adjust_pruning(
        model, pruning_ratio=initial_pruning_ratio,
        cpu_threshold=[20, 80], gpu_threshold=[20, 80]
    )
    assert new_pruning_ratio > initial_pruning_ratio  # Pruning ratio should increase

def test_pruning_ratio_decrease(mock_sys_resources, simple_model):
    mock_sys_resources.return_value = {
        'cpu_memory_percent': 15,  # Below the low threshold
        'gpu_memory_percent': 15
    }
    model = simple_model
    initial_pruning_ratio = 0.2
    _, new_pruning_ratio = adjust_pruning(
        model, pruning_ratio=initial_pruning_ratio,
        cpu_threshold=[20, 80], gpu_threshold=[20, 80]
    )
    assert new_pruning_ratio < initial_pruning_ratio  # Pruning ratio should decrease

def test_pruning_ratio_no_change(mock_sys_resources, simple_model):
    mock_sys_resources.return_value = {
        'cpu_memory_percent': 50,  # Within the threshold
        'gpu_memory_percent': 50
    }
    model = simple_model
    initial_pruning_ratio = 0.3
    _, new_pruning_ratio = adjust_pruning(
        model, pruning_ratio=initial_pruning_ratio,
        cpu_threshold=[20, 80], gpu_threshold=[20, 80]
    )
    assert new_pruning_ratio == initial_pruning_ratio  # Pruning ratio should not change

def test_pruning_ratio_min_boundary(mock_sys_resources, simple_model):
    mock_sys_resources.return_value = {
        'cpu_memory_percent': 15,  # Below the low threshold
        'gpu_memory_percent': 15
    }
    model = simple_model
    initial_pruning_ratio = 0.1  # At the minimum boundary
    _, new_pruning_ratio = adjust_pruning(
        model, pruning_ratio=initial_pruning_ratio,
        cpu_threshold=[20, 80], gpu_threshold=[20, 80]
    )
    assert new_pruning_ratio == 0.1  # Pruning ratio should not go below minimum

def test_pruning_ratio_max_boundary(mock_sys_resources, simple_model):
    mock_sys_resources.return_value = {
        'cpu_memory_percent': 85,  # Above the high threshold
        'gpu_memory_percent': 85
    }
    model = simple_model
    initial_pruning_ratio = 0.8  # At the maximum boundary
    _, new_pruning_ratio = adjust_pruning(
        model, pruning_ratio=initial_pruning_ratio,
        cpu_threshold=[20, 80], gpu_threshold=[20, 80]
    )
    assert new_pruning_ratio == 0.8  # Pruning ratio should not exceed maximum
