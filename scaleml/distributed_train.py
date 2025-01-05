def distributed_train(strategy, framework, log_resources=True, train_dataset):
    """
    Train a model using a ScaleML strategy and track resource usage (optional).
    
    Parameters:
    - strategy: ScaleML strategy for distributed training.
    - train_dataset: Dataset used for training.
    - log_resources: Boolean flag to enable/disable resource logging.
    """
    
    # log resource use throughout training
    if log_resources:
        if not devices:
            raise ValueError("Please include devices obtained from resources()")
        
        # Start logging resources in the background (you can adjust the interval as needed)
        log_resource_usage(devices, interval=10)

    # create 
    model = create_model(input_shape=None, framework)
    
    # Start the training process with resource monitoring
    with strategy.scope():
        model.fit(train_dataset, epochs=5)

    if log_resources:
        # Stop the logging once training is complete
        stop_event.set()
    
    print("Training completed.")