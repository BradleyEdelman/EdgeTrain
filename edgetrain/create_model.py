import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def create_model_tf(input_shape, model_path=None):
    """
    Create a Convolutional Neural Network (CNN) model.

    Parameters:
    - input_shape (tuple): the shape of the input data (e.g., (1, 28, 28) for MNIST).
    - model_path (str): the path to load the model.

    Returns:
    - model: A compiled tensorflow model.
    """

    # Ensure that the input shape is provided
    if input_shape is None:
        raise ValueError("Input shape must be defined.")

    if model_path and tf.io.gfile.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        # Define a Sequential model with input layer, Conv2D, MaxPooling2D, Flatten, and Dense layers
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(10, activation="softmax"),
            ]
        )

    return model


def check_sparsity(model):
    """
    Calculate the sparsity of a given model.

    Parameters:
    - model (tf.keras.Model): The TensorFlow model to check sparsity for.

    Returns:
    - float: The sparsity of the model, defined as the ratio of zero-valued parameters to the total number of parameters.
    """

    total_params = 0
    zero_params = 0
    for layer in model.layers:
        if hasattr(layer, "weights"):
            for weight in layer.weights:
                weight_values = weight.numpy()
                total_params += np.prod(weight_values.shape)
                zero_params += np.sum(np.isclose(weight_values, 0))
    sparsity = (zero_params / total_params) if total_params > 0 else 0
    return sparsity
