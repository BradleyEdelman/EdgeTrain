import pytest
import tensorflow as tf
import torch
from scaleml import create_model

@pytest.fixture
def input_shape():
    return (28, 28, 1) # Example input shape (e.g., for MNIST)

@pytest.fixture
def tensorflow_framework():
    return {'model': 'tensorflow', 'strategy': 'tensorflow'}

@pytest.fixture
def pytorch_framework():
    return {'model': 'pytorch', 'strategy': 'pytorch'}


## Correct model type
# TensorFlow
def test_tensorflow_strategy(tensorflow_framework, input_shape):
    model = create_model(tensorflow_framework, input_shape)
    assert isinstance(model, tf.keras.Model)  # Check that it returns a TensorFlow model
    
# PyTorch
def test_pytorch_strategy(pytorch_framework, input_shape):
    model = create_model(pytorch_framework, input_shape)
    assert isinstance(model, torch.nn.Module)  # Check that it returns a PyTorch model


## invalid framework type
def test_invalid_framework(input_shape):
    framework = {'model': 'invalid_framework', 'strategy': 'tensorflow'}
    with pytest.raises(ValueError, match="Unsupported framework. Please choose 'tensorflow' or 'pytorch'."):
        create_model(framework, input_shape)


## Missing input shape
# TensorFlow
def test_tensorflow_missing_input_shape(tensorflow_framework):
    with pytest.raises(ValueError, match="Input shape must be defined."):
        create_model(tensorflow_framework, None)

# PyTorch
def test_pytorch_missing_input_shape(pytorch_framework):
    with pytest.raises(ValueError, match="Input shape must be defined."):
        create_model(pytorch_framework, None)


if __name__ == "__main__":
    pytest.main()
