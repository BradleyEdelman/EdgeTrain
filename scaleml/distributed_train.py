def distributed_train(detected_resources, framework, strategy, train_dataset, log_resources=True):
    """
    Train a model using a ScaleML strategy and track resource usage (optional).
    
    Parameters:
    - detected_resources (dict): A dictionary containing the detected resources. 
                         Expected keys: 'logical_cores', 'gpu_devices'.
    - framework (dict): A dictionary containing the framework details.
                         Expected keys: 'model', 'strategy'.
                         Options for 'model': 'tensorflow', 'pytorch'.
                         Options for 'strategy': 'tensorflow', 'pytorch', 'horovod'.
    - strategy: ScaleML strategy for distributed training.
    - train_dataset: Dataset used for training.
    - log_resources: Boolean flag to enable/disable resource logging.
    """
    import tensorflow as tf
    import torch
    import horovod.tensorflow as hvd_tf
    import horovod.torch as hvd_torch

    # log resource use throughout training
    if log_resources:
        if not detected_resources:
            raise ValueError("Please include output from resources() to log resources")
        
        # Start logging resources in the background (you can adjust the interval as needed)
        log_resource_usage(detected_resources, interval=10)
    
    # Detect size of the first image from train_dataset
    for images, labels in train_dataset.take(1):
        input_shape = images.shape[1:]
        break

    # create model based on the detected framework
    model = create_model(framework, input_shape)
    
    # Start the training process with or without resource monitoring
    if isinstance(strategy, tf.distribute.Strategy):

        with strategy.scope():
            history = model.fit(train_dataset, epochs=20)
            return history

    elif isinstance(strategy, torch.device):
        model.to(strategy)
        history = train_pytorch_model(model, train_dataset, epochs=20)
        return history

    elif 'horovod' in str(type(strategy)).lower():
        
        if 'tensorflow' in str(type(strategy)).lower():
            history = model.fit(train_dataset, epochs=20)
            return history
        elif 'torch' in str(type(strategy)).lower():
            model.to(strategy)
        history = train_pytorch_model(model, train_dataset, epochs=20)
        return history

    if log_resources:
        # Stop the logging once training is complete
        stop_event.set()
    
    print("Training completed.")