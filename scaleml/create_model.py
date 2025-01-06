def create_model(strategy, framework, input_shape):
    """
    Create a Convolutional Neural Network (CNN) model for the specified framework.

    Parameters:
    - framework (dict): A dictionary containing the framework details.
                         Expected keys: 'model', 'strategy'.
                         Options for 'model': 'tensorflow', 'pytorch'.
                         Options for 'strategy': 'tensorflow', 'pytorch', 'horovod'.
    - input_shape: tuple, the shape of the input data (e.g., (1, 28, 28) for MNIST).

    Returns:
    - model: A compiled model for the specified model framework.
    """
    
    # Ensure that the input shape is provided
    if input_shape is None:
        raise ValueError("Input shape must be defined.")
    
    # Extract the model framework
    model_framework = framework.get('model')
    
    # Customize the model by adding/removing layers, changing activation functions, or modifying hyperparameters
    if model_framework == 'tensorflow':
        
        import tensorflow as tf
        with strategy.scope():
            # Define a Sequential model with input layer, Conv2D, MaxPooling2D, Flatten, and Dense layers
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(shape=input_shape),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            # Compile the model with Adam optimizer and SparseCategoricalCrossentropy loss
            model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
        return model
    
    elif model_framework == 'pytorch':
        
        import torch.nn as nn
        import torch.nn.functional as F
        
        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.conv1 = nn.Conv2d(in_channels=input_shape[2], out_channels=32, kernel_size=3)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv2 = nn.Conv2d(32, 64, 3)
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(64 * 5 * 5, 64)
                self.fc2 = nn.Linear(64, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = self.flatten(x)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        model = CNN()
        return model
    
    else:
        raise ValueError("Unsupported framework. Please choose 'tensorflow' or 'pytorch'.")