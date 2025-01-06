def distributed_train(framework, strategy, train_dataset, log_resources=True):
    """
    Train a model using a specified framework and strategy, and track resource usage (optional).
    
    Parameters:
    - framework (dict): A dictionary containing the framework details.
                         Expected keys: 'model', 'strategy'.
                         Options for 'model': 'tensorflow', 'pytorch'.
                         Options for 'strategy': 'tensorflow', 'pytorch', 'horovod'.
    - strategy: distributed ScaleML training strategy appropriate for the model.
    - train_dataset: Dataset used for training.
    - log_resources: Boolean flag to enable/disable resource logging.
    """
    
    import tensorflow as tf
    import torch
    from datetime import datetime
    from multiprocessing import Process, Value
    import os, time, pickle
    # import horovod.tensorflow as hvd_tf
    # import horovod.torch as hvd_torch

    from scaleml import create_scaleml_folders, create_model, start_logging, stop_logging

    # create folders for saving
    scaleml_folder = create_scaleml_folders()
    print(f"Scaleml folder and subfolders are set up at: {scaleml_folder}")
    print()

    # log resource use throughout training
    log_dir = f"{scaleml_folder}/logs/"
    log_file = f"{log_dir}/{datetime.now().strftime('%Y%m%d')}_resource_usage_log.csv"
    if log_resources:        
        # Start logging resources in the background (adjust the interval as needed)
        log_process = start_logging(log_file, interval=10)
    print()
    
    # Detect size of the first image from train_dataset
    for images, labels in train_dataset.take(1):
        input_shape = images.shape[1:]
        break

    # create model based on the detected framework
    model = create_model(strategy, framework, input_shape)
    
    # Save the model
    log_time = log_file.split('/')[-1]
    log_time = log_time.split('_')[0]
    if framework == 'tensorflow':
        model.save(os.path.join(f"{scaleml_folder}/models/", f"{log_time}_tf_model.h5"))
    elif framework == 'pytorch':
        torch.save(model.state_dict(), os.path.join(f"{scaleml_folder}/models/", 'f"{log_time}_torch_model.pth'))
    
    # Start the training process with or without resource monitoring
    if isinstance(strategy, tf.distribute.Strategy):

        with strategy.scope():
            history = model.fit(train_dataset, epochs=2)
            history_file = os.path.join(f"{scaleml_folder}/models/", f"{log_time}_tf_model_history.pkl")
            with open(history_file, 'wb') as f:
                pickle.dump(history.history, f)

    elif isinstance(strategy, torch.device):
        model.to(strategy)
        # history = train_pytorch_model(model, train_dataset, epochs=2)
        history_file = os.path.join(f"{scaleml_folder}/models/", f"{log_time}_torch_model_history.pkl")
        with open(history_file, 'wb') as f:
                pickle.dump(history.history, f)

    # elif 'horovod' in str(type(strategy)).lower():

    if log_resources:
        # Stop the logging once training is complete
        stop_logging(log_process)
    
    print("Training completed.")
    return history, history_file, log_file