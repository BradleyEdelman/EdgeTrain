def strategies(resources=None, framework='tensorflow', devices='all'):
    """
    Sets up a distributed training strategy based on the chosen framework, available resources, and specified devices.

    Args:
        framework (str): The deep learning framework to use. 
                         Options: 'tensorflow', 'pytorch', 'horovod'.
        resources (dict): A dictionary containing the detected resources. 
                          Expected keys: 'logical_cores', 'gpu_devices'.
        devices (str): The type of devices to use. Options: 'cpu', 'gpu', 'all'.

    Returns:
        strategy or device setup appropriate for the framework.
    """
    
    if resources is None:
        raise ValueError("Resources must be provided. Usie the 'resources()' function to detect resources.")
        
    logical_cores = resources.get('logical_cores', 0)
    gpu_devices = resources.get('gpu_devices', [])
    
    if devices == 'gpu' and not gpu_devices:
        print("No GPUs exist. Only using CPU resources.")
        devices = 'cpu'

    
    # distributed strategy depending on specified framework
    if framework.lower() == 'tensorflow':
        
        import tensorflow as tf
        
        # TensorFlow strategy setup
        if devices == 'all':
            strategy = tf.distribute.MirroredStrategy()
        elif devices == 'gpu' and gpu_devices:
            strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{i}" for i in range(len(gpu_devices))])
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        print(f"TensorFlow strategy initialized: {strategy}")
        return strategy

    elif framework.lower() == 'pytorch':
        
        import torch
        
        # PyTorch device setup
        if torch.cuda.is_available() and gpu_devices:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"PyTorch device: {device}")
        return device

    elif framework.lower() == 'horovod':
        
        import horovod.tensorflow as hvd
        
        hvd.init()
        print(f"Horovod rank: {hvd.rank()}, size: {hvd.size()}")
        
        if gpu_devices:
            import tensorflow as tf
            tf.config.experimental.set_visible_devices(gpu_devices[hvd.local_rank()], 'GPU')
        return hvd

    else:
        raise ValueError("Unsupported framework. Choose from 'tensorflow', 'pytorch', or 'horovod'.")

