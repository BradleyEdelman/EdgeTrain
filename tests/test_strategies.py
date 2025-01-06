import pytest
import tensorflow as tf
import torch
# import horovod.tensorflow as hvd_tf
# import horovod.torch as hvd_torch
from scaleml import resources, strategies

## Mock resources function for testing.
@pytest.fixture
def mock_resources():
    return {
        'CPU cores': 4,
        'GPU devices': 2
    }

@pytest.fixture
def mock_resources_nogpu():
    return {
        'CPU cores': 4,
        'GPU devices': 0
    }


## Tensorflow
# 'cpu'
def test_tensorflow_cpu_strategy(mock_resources):
    framework = {
        'model': 'tensorflow',
        'strategy': 'tensorflow'
    }
    resource_type = 'cpu'
    strategy = strategies(framework, mock_resources, resource_type)
    assert isinstance(strategy, tf.distribute.MirroredStrategy)

# 'gpu'
def test_tensorflow_gpu_strategy(mock_resources):
    framework = {
        'model': 'tensorflow',
        'strategy': 'tensorflow'
    }
    resource_type = 'gpu'
    strategy = strategies(framework, mock_resources, resource_type)
    assert isinstance(strategy, tf.distribute.MirroredStrategy)

# 'all'
def test_tensorflow_all_strategy(mock_resources):
    framework = {
        'model': 'tensorflow',
        'strategy': 'tensorflow'
    }
    resource_type = 'all'
    strategy = strategies(framework, mock_resources, resource_type)
    assert isinstance(strategy, tf.distribute.MirroredStrategy)


## PyTorch
# 'cpu'
def test_pytorch_cpu_strategy(mock_resources):
    framework = {
        'model': 'pytorch',
        'strategy': 'pytorch'
    }
    resource_type = 'cpu'
    strategy = strategies(framework, mock_resources, resource_type)
    assert isinstance(strategy, torch.device) and 'cpu' in str(strategy)

# 'gpu'
def test_pytorch_gpu_strategy(mock_resources):
    framework = {
        'model': 'pytorch',
        'strategy': 'pytorch'
    }
    resource_type = 'gpu'
    strategy = strategies(framework, mock_resources, resource_type)
    assert isinstance(strategy, torch.device) and 'cuda' in str(strategy)

# 'all'
def test_pytorch_all_strategy(mock_resources):
    framework = {
        'model': 'pytorch',
        'strategy': 'pytorch'
    }
    resource_type = 'all'
    strategy = strategies(framework, mock_resources, resource_type)
    assert isinstance(strategy, torch.device) and ('cpu' in str(strategy) or 'cuda' in str(strategy))


# ## Horovod
# # TensorFlow
# def test_horovod_tensorflow_strategy(mock_resources):
#     framework = {
#         'model': 'tensorflow',
#         'strategy': 'horovod'
#     }
#     resource_type = 'all'
#     strategy = strategies(framework, mock_resources, resource_type)
#     assert hasattr(strategy, 'rank')  # Horovod TensorFlow should return a horovod object with 'rank'

# # Test for Horovod strategy with PyTorch
# def test_horovod_pytorch_strategy(mock_resources):
#     framework = {
#         'model': 'pytorch',
#         'strategy': 'horovod'
#     }
#     resource_type = 'all'
#     strategy = strategies(framework, mock_resources, resource_type)
#     assert hasattr(strategy, 'rank')  # Horovod PyTorch should return a horovod object with 'rank'


## Framework issues
# Test when invalid framework model is provided (tensorflow with pytorch)
def test_invalid_model_for_tensorflow(mock_resources):
    framework = {
        'model': 'pytorch',
        'strategy': 'tensorflow'
    }
    resource_type = 'all'
    with pytest.raises(ValueError, match="Incompatible model for TensorFlow strategy. Please use a TensorFlow model."):
        strategies(framework, mock_resources, resource_type)

def test_invalid_model_for_pytorch(mock_resources):
    framework = {
        'model': 'tensorflow',
        'strategy': 'pytorch'
    }
    resource_type = 'all'
    with pytest.raises(ValueError, match="Incompatible model for PyTorch strategy. Please use a PyTorch model."):
        strategies(framework, mock_resources, resource_type)

# Test when an unsupported framework is passed
def test_unsupported_framework(mock_resources):
    framework = {
        'model': 'tensorflow',
        'strategy': 'unsupported_framework'
    }
    resource_type = 'all'
    with pytest.raises(ValueError, match="Unsupported framework or model. Choose from 'tensorflow' or 'pytorch'."):
        strategies(framework, mock_resources, resource_type)


## Missing resources
# tensorflow
def test_missing_resources_gpu_tensorflow(mock_resources_nogpu):
    framework = {
        'model': 'tensorflow',
        'strategy': 'tensorflow'
    }
    resource_type = 'gpu'
    strategy = strategies(framework, mock_resources_nogpu, resource_type)
    assert isinstance(strategy, tf.distribute.MirroredStrategy)

def test_missing_resources_all_tensorflow(mock_resources_nogpu):
    framework = {
        'model': 'tensorflow',
        'strategy': 'tensorflow'
    }
    resource_type = 'all'
    strategy = strategies(framework, mock_resources_nogpu, resource_type)
    assert isinstance(strategy, tf.distribute.MirroredStrategy)

# pytorch
def test_missing_resources_gpu_pytorch(mock_resources_nogpu):
    framework = {
        'model': 'pytorch',
        'strategy': 'pytorch'
    }
    resource_type = 'gpu'
    strategy = strategies(framework, mock_resources_nogpu, resource_type)
    assert isinstance(strategy, torch.device) and 'cpu' in str(strategy)

def test_missing_resources_all_pytorch(mock_resources_nogpu):
    framework = {
        'model': 'pytorch',
        'strategy': 'pytorch'
    }
    resource_type = 'all'
    strategy = strategies(framework, mock_resources_nogpu, resource_type)
    assert isinstance(strategy, torch.device) and 'cpu' in str(strategy)

if __name__ == "__main__":
    pytest.main()