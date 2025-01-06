import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pytest
from scaleml import resources, strategies

## Mock  resources function for testing.
@pytest.fixture
def mock_resources():
    return {
        'logical_cores': 4,
        'gpu_devices': ['/gpu:0', '/gpu:1']
    }


## Tensorflow
# 'cpu'
def test_tensorflow_cpu_strategy(mock_resources):
    framework = {
        'model': 'tensorflow',
        'strategy': 'tensorflow'
    }
    devices = 'cpu'
    strategy = strategies(framework, mock_resources, devices)
    assert isinstance(strategy, tf.distribute.OneDeviceStrategy) 

# 'gpu'
def test_tensorflow_gpu_strategy(mock_resources):
    framework = {
        'model': 'tensorflow',
        'strategy': 'tensorflow'
    }
    devices = 'gpu'
    strategy = strategies(framework, mock_resources, devices)
    assert isinstance(strategy, str)  # Should return a string representation of MirroredStrategy

# 'all'
def test_tensorflow_gpu_strategy(mock_resources):
    framework = {
        'model': 'tensorflow',
        'strategy': 'tensorflow'
    }
    devices = 'all'
    strategy = strategies(framework, mock_resources, devices)
    assert isinstance(strategy, str)  # Should return a string representation of MirroredStrategy


## PyTorch
# 'cpu'
def test_pytorch_cpu_strategy(mock_resources):
    framework = {
        'model': 'pytorch',
        'strategy': 'pytorch'
    }
    devices = 'cpu'
    strategy = strategies(framework, mock_resources, devices)
    assert isinstance(strategy, str)  # Should return a torch device object ('cpu')

# 'gpu'
def test_pytorch_gpu_strategy(mock_resources):
    framework = {
        'model': 'pytorch',
        'strategy': 'pytorch'
    }
    devices = 'gpu'
    strategy = strategies(framework, mock_resources, devices)
    assert isinstance(strategy, str)  # Should return a torch device object ('cuda')

# 'all'
def test_pytorch_gpu_strategy(mock_resources):
    framework = {
        'model': 'pytorch',
        'strategy': 'pytorch'
    }
    devices = 'all'
    strategy = strategies(framework, mock_resources, devices)
    assert isinstance(strategy, str)  # Should return a torch device object ('cuda')


## Horovod
# TensorFlow
def test_horovod_tensorflow_strategy(mock_resources):
    framework = {
        'model': 'tensorflow',
        'strategy': 'horovod'
    }
    devices = 'all'
    strategy = strategies(framework, mock_resources, devices)
    assert hasattr(strategy, 'rank')  # Horovod TensorFlow should return a horovod object with 'rank'

# Test for Horovod strategy with PyTorch
def test_horovod_pytorch_strategy(mock_resources):
    framework = {
        'model': 'pytorch',
        'strategy': 'horovod'
    }
    devices = 'all'
    strategy = strategies(framework, mock_resources, devices)
    assert hasattr(strategy, 'rank')  # Horovod PyTorch should return a horovod object with 'rank'


## Framework issues
# Test when invalid framework model is provided (tensorflow with pytorch)
def test_invalid_model_for_tensorflow(mock_resources):
    framework = {
        'model': 'pytorch',
        'strategy': 'tensorflow'
    }
    devices = 'all'
    with pytest.raises(ValueError, match="Incompatible model for TensorFlow strategy"):
        strategies(framework, mock_resources, devices)

# Test when an unsupported framework is passed
def test_unsupported_framework(mock_resources):
    framework = {
        'model': 'tensorflow',
        'strategy': 'unsupported_framework'
    }
    devices = 'all'
    with pytest.raises(ValueError, match="Unsupported framework or model"):
        strategies(framework, mock_resources, devices)


## Missing resources
# Test for no detected gpu
@pytest.fixture
def mock_resources():
    return {
        'logical_cores': 4,
        'gpu_devices': ['/gpu:0', '/gpu:1']
    }
    
def test_missing_resources(mock_resources):
    framework = {
        'model': 'tensorflow',
        'strategy': 'tensorflow'
    }
    devices = 'gpu'
    with pytest.raises(ValueError, match="Resources must be provided"):
        strategies(framework, Nomock_resourcesne, devices)