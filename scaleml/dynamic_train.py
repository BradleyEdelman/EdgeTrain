import tensorflow as tf
from scaleml import log_usage, adjust_workers, adjust_batch_size

def dynamic_train(model, train_dataset, epochs, base_batch_size=32, log_file="resource_log.csv", dynamic_adjustments=True):
    """
    Train the model with optional dynamic resource adjustment.
    
    Parameters:
    - model: The TensorFlow model to train.
    - train_dataset: The training dataset.
    - epochs: Number of epochs to train the model.
    - base_batch_size: The base batch size to use.
    - log_file: The path to the log file where resource usage is saved.
    - dynamic_adjustments: A flag to control if dynamic adjustments are enabled (True) or not (False).
    
    Returns:
    - None
    """
    # Log resource usage (regardless of dynamic adjustments)
    log_resource_usage(log_file, base_batch_size, 2)  # Default workers are 2 for logging

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        if dynamic_adjustments:
            # Adjust resources dynamically based on system usage
            cpu_percent, gpu_memory_usage, gpu_memory_total, gpu_percent = get_system_resources()
            num_workers = adjust_workers(cpu_threshold=80, gpu_threshold=80)  # Adjust workers based on resources
            batch_size = adjust_batch_size(cpu_percent, gpu_percent, base_batch_size)  # Adjust batch size
        else:
            # Keep default batch size and workers fixed
            num_workers = 2  # Default number of workers
            batch_size = base_batch_size  # Default batch size
        
        print(f"Using {num_workers} workers and batch size {batch_size}")
        
        # Create the MirroredStrategy for distributed training
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model.fit(train_dataset.batch(batch_size), epochs=1)  # Train for 1 epoch at a time

        # Log resource usage for the current epoch
        log_resource_usage(log_file, batch_size, num_workers)

