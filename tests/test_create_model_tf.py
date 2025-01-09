import pytest
import tensorflow as tf
from tensorflow.keras import models
from edgetrain import create_model_tf  # Replace 'your_module' with the actual module name

def test_create_model_without_path():
    """
    Test creating a model without providing a preloaded model path.
    """
    input_shape = (28, 28, 1)
    model = create_model_tf(input_shape)
    assert isinstance(model, models.Sequential), "Model should be an instance of Sequential"
    assert model.input_shape[1:] == input_shape, "Input shape does not match"


def test_create_model_with_valid_path(tmp_path):
    """
    Test creating a model by loading from a valid path.
    """
    input_shape = (28, 28, 1)
    model = create_model_tf(input_shape)
    model_path = tmp_path / "test_model"
    model.save(model_path)

    loaded_model = create_model_tf(input_shape, model_path=str(model_path))
    assert isinstance(loaded_model, models.Sequential), "Loaded model should be an instance of Sequential"


def test_create_model_invalid_input_shape():
    """
    Test that a ValueError is raised when input_shape is None.
    """
    with pytest.raises(ValueError, match="Input shape must be defined."):
        create_model_tf(None)
