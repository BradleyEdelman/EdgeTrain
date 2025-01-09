import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from edgetrain import log_usage_once, adjust_threads, adjust_batch_size, adjust_learning_rate, create_model_tf, get_edgetrain_folder

def dynamic_train(train_dataset, epochs=10, batch_size=32, lr=1e-3, grad_accum=1, log_file="resource_log.csv", dynamic_adjustments=True):
    """
    Train the model with optional dynamic resource adjustment.
    
    Parameters:
    - train_dataset (dict): The training dataset.
    - epochs (int): Number of epochs to train the model.
    - batch_size (int): The base batch size to use.
    - lr (float): The initial learning rate.
    - grad_accum (int): The number of gradient accumulation steps.
    - log_file (str): The path to the log file where resource usage is saved.
    - dynamic_adjustments (bool): A flag to control if dynamic adjustments are enabled (True) or not (False).
    
    Returns:
    - history_list (list): A list of training history for each epoch.
    """
    
    # Log resource usage (regardless of dynamic adjustments)
    log_usage_once(log_file, lr=lr, batch_size=batch_size, grad_accum=grad_accum, num_epoch=0)
    
    # Create the MirroredStrategy for distributed training
    strategy = tf.distribute.MirroredStrategy()

    history_list = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Adjust resources dynamically based on system usage
        if dynamic_adjustments:
            batch_size=adjust_batch_size(batch_size=batch_size)
            lr=adjust_learning_rate(lr=lr)
            # grad_accum=adjust_grad_accum(grad_accum=grad_accum)
        else:
            # Keep default batch size and workers fixed
            batch_size=batch_size
            lr=lr
            grad_accum=grad_accum
        
        # Deploy training
        train_images, train_labels = train_dataset['images'], train_dataset['labels']
        with strategy.scope():
            model = create_model_tf(input_shape=train_images[0].shape)
            
            # Create optimizer with learning rate
            optimizer = Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Train for 1 epoch at a time
            history = model.fit(
                train_images,
                train_labels,
                batch_size=batch_size,
                epochs=1
            )  
            
            # save results
            history_list.append(history.history)
        
        # Log resource usage for the current epoch
        log_usage_once(log_file, lr=lr, batch_size=batch_size, grad_accum=grad_accum, num_epoch=epoch)

    return history_list

