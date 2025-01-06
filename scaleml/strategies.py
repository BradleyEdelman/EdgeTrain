def strategies(framework, detected_resources, devices='all'):
    """
    Sets up a distributed training strategy based on the chosen framework, available resources, and specified devices.

    Args:
        framework (dict): A dictionary containing the framework details.
                         Expected keys: 'model', 'strategy'.
                         Options for 'model': 'tensorflow', 'pytorch'.
                         Options for 'strategy': 'tensorflow', 'pytorch', 'horovod'.
        detected_resources (dict): A dictionary containing the detected resources. 
                         Expected keys: 'logical_cores', 'gpu_devices'.
        devices (str): The type of devices to use. Options: 'cpu', 'gpu', 'all'.

    Returns:
        Distributed ScaleML training strategy appropriate for the model.
    """
    
    # Check if detected resources are provided, if not raise an error.
    if detected_resources is None:
        raise ValueError("Resources must be provided. Use the 'resources()' function to detect resources.")
        
    # Extract logical cores and GPU devices from detected resources.
    logical_cores = [i for i, detected_resources in enumerate(detected_resources) if 'cpu' in detected_resources]
    gpu_devices = [i for i, detected_resources in enumerate(detected_resources) if 'gpu' in detected_resources]
    if devices in ['gpu', 'all'] and not gpu_devices:
        print("No GPUs exist. Only using CPU resources.")
        devices = 'cpu'

    # extract framework assignments
    framework_model = framework.get('model')
    framework_strategy = framework.get('strategy')
    

    # Create a istributed strategy depending on the input framework
    if framework_strategy.lower() == 'tensorflow':
        
        if framework_model.lower() != 'tensorflow':
            raise ValueError("Incompatible model for TensorFlow strategy. Please use a TensorFlow model.")
        else:
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

    elif framework_strategy.lower() == 'pytorch':
        
        if framework_model.lower() != 'pytorch':
            raise ValueError("Incompatible model for PyTorch strategy. Please use a PyTorch model.")
        else:
            import torch
            
            # PyTorch device setup
            if devices == 'all':
                strategy = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif devices == 'gpu' and gpu_devices:
                strategy = torch.device("cuda")
            else:
                strategy = torch.device("cpu")
            print(f"PyTorch device: {strategy}")
            return strategy

    elif framework_strategy.lower() == 'horovod':
        
        import horovod.tensorflow as hvd_tf
        import horovod.torch as hvd_torch
        
        if framework_model.lower() == 'tensorflow':
            hvd_tf.init()
            print(f"Horovod TensorFlow rank: {hvd_tf.rank()}, size: {hvd_tf.size()}")
            
            if gpu_devices:
                import tensorflow as tf
                tf.config.experimental.set_visible_devices(gpu_devices[hvd_tf.local_rank()], 'GPU')
            return hvd_tf
        
        elif framework_model.lower() == 'pytorch':
            hvd_torch.init()
            print(f"Horovod PyTorch rank: {hvd_torch.rank()}, size: {hvd_torch.size()}")
            
            if gpu_devices:
                import torch
                torch.cuda.set_device(hvd_torch.local_rank())
            return hvd_torch

    else:
        raise ValueError("Unsupported framework or model. Choose from 'tensorflow', 'pytorch', or 'horovod'.")