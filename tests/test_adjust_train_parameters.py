import pytest
from unittest.mock import patch
from edgetrain import adjust_training_parameters

@pytest.fixture
def default_parameters():
    return {
        "priority_values": {"batch_size": 0.5, "pruning": 0.3, "learning_rate": 0.2},
        "batch_size": 32,
        "pruning_ratio": 0.4,
        "lr": 0.001,
        "accuracy_score": 0.8
    }


def test_adjust_batch_size_high_memory(default_parameters):
    # Simulate high memory usage
    sys_resources = {"cpu_memory_percent": 80, "gpu_memory_percent": 85}
    params = default_parameters.copy()
    params["priority_values"]["batch_size"] = 0.8  # Highest priority
    params["resources"] = sys_resources

    adjusted_batch_size, adjusted_pruning_ratio, adjusted_lr = adjust_training_parameters(**params)

    assert adjusted_batch_size == 16, "Batch size adjustment for high memory usage failed."
    assert adjusted_pruning_ratio == params["pruning_ratio"], "Pruning ratio should remain unchanged."
    assert adjusted_lr == params["lr"], "Learning rate should remain unchanged."


def test_adjust_batch_size_low_memory(default_parameters):
    # Simulate low memory usage
    sys_resources = {"cpu_memory_percent": 40, "gpu_memory_percent": 35}
    params = default_parameters.copy()
    params["priority_values"]["batch_size"] = 0.8  # Highest priority
    params["resources"] = sys_resources

    adjusted_batch_size, adjusted_pruning_ratio, adjusted_lr = adjust_training_parameters(**params)
    
    assert adjusted_batch_size == 64, "Batch size adjustment for low memory usage failed."
    assert adjusted_pruning_ratio == params["pruning_ratio"], "Pruning ratio should remain unchanged."
    assert adjusted_lr == params["lr"], "Learning rate should remain unchanged."


def test_adjust_pruning_high_memory(default_parameters):
    # Simulate high memory usage
    sys_resources = {"cpu_memory_percent": 85, "gpu_memory_percent": 90}
    params = default_parameters.copy()
    params["priority_values"]["pruning"] = 0.8  # Highest priority
    params["resources"] = sys_resources

    adjusted_batch_size, adjusted_pruning_ratio, adjusted_lr = adjust_training_parameters(**params)
    print(adjusted_pruning_ratio)
    assert adjusted_pruning_ratio == pytest.approx(0.5, rel=1e-3), "Pruning ratio adjustment for high memory usage failed."
    assert adjusted_batch_size == params["batch_size"], "Batch size should remain unchanged."
    assert adjusted_lr == params["lr"], "Learning rate should remain unchanged."


def test_adjust_pruning_low_memory(default_parameters):
    # Simulate low memory usage
    sys_resources = {"cpu_memory_percent": 40, "gpu_memory_percent": 30}
    params = default_parameters.copy()
    params["priority_values"]["pruning"] = 0.8  # Highest priority
    params["resources"] = sys_resources
    
    adjusted_batch_size, adjusted_pruning_ratio, adjusted_lr = adjust_training_parameters(**params)
    print(adjusted_pruning_ratio)
    assert adjusted_pruning_ratio == pytest.approx(0.3, rel=1e-3), "Pruning ratio adjustment for low memory usage failed."
    assert adjusted_batch_size == params["batch_size"], "Batch size should remain unchanged."
    assert adjusted_lr == params["lr"], "Learning rate should remain unchanged."


def test_adjust_learning_rate_low_accuracy(default_parameters):
    # Simulate low accuracy
    sys_resources = {"cpu_memory_percent": 60, "gpu_memory_percent": 60}
    params = default_parameters.copy()
    params["priority_values"]["learning_rate"] = 0.8  # Highest priority
    params["accuracy_score"] = 0.03
    params["resources"] = sys_resources

    adjusted_batch_size, adjusted_pruning_ratio, adjusted_lr = adjust_training_parameters(**params)

    assert adjusted_lr == pytest.approx(0.0005, rel=1e-2), "Learning rate adjustment for low accuracy failed."
    assert adjusted_batch_size == params["batch_size"], "Batch size should remain unchanged."
    assert adjusted_pruning_ratio == params["pruning_ratio"], "Pruning ratio should remain unchanged."


def test_adjust_learning_rate_high_accuracy(default_parameters):
    # Simulate high accuracy
    sys_resources = {"cpu_memory_percent": 60, "gpu_memory_percent": 60}
    params = default_parameters.copy()
    params["priority_values"]["learning_rate"] = 0.8  # Highest priority
    params["accuracy_score"] = 0.97
    params["resources"] = sys_resources

    adjusted_batch_size, adjusted_pruning_ratio, adjusted_lr = adjust_training_parameters(**params)

    assert adjusted_lr == pytest.approx(0.0012, rel=1e-2), "Learning rate adjustment for high accuracy failed."
    assert adjusted_batch_size == params["batch_size"], "Batch size should remain unchanged."
    assert adjusted_pruning_ratio == params["pruning_ratio"], "Pruning ratio should remain unchanged."


def test_no_adjustments_when_no_priorities(default_parameters):
    # Simulate balanced memory and low priorities
    sys_resources = {"cpu_memory_percent": 60, "gpu_memory_percent": 60}
    params = default_parameters.copy()
    params["priority_values"] = {"batch_size": 0.0, "pruning": 0.0, "learning_rate": 0.0}
    params["resources"] = sys_resources

    adjusted_batch_size, adjusted_pruning_ratio, adjusted_lr = adjust_training_parameters(**params)

    assert adjusted_batch_size == params["batch_size"], "Batch size should remain unchanged with no priorities."
    assert adjusted_pruning_ratio == params["pruning_ratio"], "Pruning ratio should remain unchanged with no priorities."
    assert adjusted_lr == params["lr"], "Learning rate should remain unchanged with no priorities."
