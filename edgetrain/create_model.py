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
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
    return model


def create_model_torch(input_shape, model_path=None):
    """
    Create a Convolutional Neural Network (CNN) model.

    Parameters:
    - input_shape (tuple): the shape of the input data (e.g., (1, 28, 28) for MNIST).
    - model_path (str): the path to load the model.

    Returns:
    - model: A compiled pytorch model.
    """

    import torch.nn as nn
    import torch.nn.functional as F

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, 3)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * 5 * 5, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.flatten(x)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = CNN()
    return model