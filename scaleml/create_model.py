def create_model_tf(input_shape, model_path=None):
    """
    Create a Convolutional Neural Network (CNN) model.

    Parameters:
    - input_shape: tuple, the shape of the input data (e.g., (1, 28, 28) for MNIST).
    - model_path: str, the path to load the model.

    Returns:
    - model: A compiled tensorflow model.
    """
    
    # Ensure that the input shape is provided
    if input_shape is None:
        raise ValueError("Input shape must be defined.")
        
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # Define a Sequential model with input layer, Conv2D, MaxPooling2D, Flatten, and Dense layers
    model = models.Sequential([
        layers.InputLayer(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model with Adam optimizer and SparseCategoricalCrossentropy loss
    model.compile(optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy'])
    return model


def create_model_torch(input_shape, model_path=None):
    """
    Create a Convolutional Neural Network (CNN) model.

    Parameters:
    - input_shape: tuple, the shape of the input data (e.g., (1, 28, 28) for MNIST).
    - model_path: str, the path to load the model.

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