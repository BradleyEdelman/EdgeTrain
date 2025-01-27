import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from edgetrain import log_usage_once, create_model_tf, compute_scores, define_priorities, adjust_training_parameters

def dynamic_train(
    train_dataset, 
    epochs=10, 
    batch_size=32, 
    lr=1e-3, 
    pruning_ratio=0.2, 
    log_file="resource_log.csv", 
    dynamic_adjustments=True
):
    """
    Train the model with optional dynamic resource adjustment.
    
    Parameters:
    - train_dataset (dict): The training dataset.
    - epochs (int): Number of epochs to train the model.
    - batch_size (int): The base batch size to use.
    - lr (float): The initial learning rate.
    - pruning_ratio (float): Initial pruning ratio (for dynamic adjustment).
    - log_file (str): The path to the log file where resource usage is saved.
    - dynamic_adjustments (bool): A flag to control if dynamic adjustments are enabled (True) or not (False).
    
    Returns:
    - history_list (list): A list of training history for each epoch.
    """
    
    # Log initial resource usage
    log_usage_once(log_file, lr=lr, batch_size=batch_size, num_epoch=0)

    # Create the MirroredStrategy for distributed training
    strategy = tf.distribute.MirroredStrategy()

    # Initialize variables
    history_list = []
    prev_accuracy = 0.0

    # Create the model once
    train_images, train_labels = train_dataset['images'], train_dataset['labels']
    with strategy.scope():
        model = create_model_tf(input_shape=train_images[0].shape)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Compile the model with the current parameters
        with strategy.scope():
            optimizer = Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train for 1 epoch
        history = model.fit(
            train_images,
            train_labels,
            batch_size=batch_size,
            epochs=1
        )  
        
        # Save training history
        history_list.append(history.history)
        
        # Update accuracy
        curr_accuracy = history.history['accuracy'][-1]

        # Log resource usage for the current epoch
        log_usage_once(log_file, lr=lr, batch_size=batch_size, num_epoch=epoch + 1)

        if dynamic_adjustments:
            # Calculate performance and resource usage scores
            normalized_scores = compute_scores(prev_accuracy, curr_accuracy)

            # Define priority values based on normalized scores
            priority_value = define_priorities(normalized_scores)

            # Adjust training parameters
            batch_size, pruning_ratio, lr = adjust_training_parameters(
                priority_scores=priority_value,
                batch_size=batch_size,
                pruning_ratio=pruning_ratio,
                lr=lr,
                accuracy_score=curr_accuracy
            )

            print(f"Adjusted parameters for next epoch: batch_size={batch_size}, pruning_ratio={pruning_ratio}, learning_rate={lr}")

        # Update previous accuracy
        prev_accuracy = curr_accuracy

    return history_list
