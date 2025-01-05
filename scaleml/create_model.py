def create_model(input_shape=None, framework):
    """
    Create a Convolutional Neural Network (CNN) model for the specified framework.

    Parameters:
    - input_shape: tuple, the shape of the input data (e.g., (28, 28, 1) for MNIST).
    - framework: str, the deep learning framework to use ('tensorflow', 'pytorch').

    Returns:
    - model: A compiled model for the specified framework.
    """
    
    if input_shape is None:
        raise ValueError("Input shape must be defined.")
    
    if framework == 'tensorflow':
        
        import tensorflow as tf
        # Define a Sequential model with Conv2D, MaxPooling2D, Flatten, and Dense layers
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        # Compile the model with Adam optimizer and SparseCategoricalCrossentropy loss
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model
        # Customize the model by adding/removing layers, changing activation functions, or modifying hyperparameters
    
    elif framework == 'pytorch':
        
        import torch.nn as nn
        import torch.nn.functional as F
        
        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.conv1 = nn.Conv2d(in_channels=input_shape[2], out_channels=32, kernel_size=3, activation='relu')
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv2 = nn.Conv2d(32, 64, 3, activation='relu')
                self.conv3 = nn.Conv2d(64, 64, 3, activation='relu')
                self.fc1 = nn.Linear(64 * 5 * 5, 64)
                self.fc2 = nn.Linear(64, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = F.relu(self.conv3(x))
                x = x.view(-1, 64 * 5 * 5)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        model = CNN()
        return model
    
    else:
        raise ValueError("Unsupported framework. Please choose 'tensorflow', 'pytorch'.")