def strategies(framework, detected_resources, resource_type='all'):
    """
    Sets up a distributed training strategy based on the chosen framework, available resources, and specified devices.

    Args:
        framework (dict): A dictionary containing the framework details.
                         Expected keys: 'model', 'strategy'.
                         Options for 'model': 'tensorflow', 'pytorch'.
                         Options for 'strategy': 'tensorflow', 'pytorch', 'horovod'.
        detected_resources (dict): A dictionary containing the detected resources. 
                         Expected keys: 'logical_cores', 'gpu_devices'.
        resource_type (str): The type of devices to use. Options: 'cpu', 'gpu', 'all'.

    Returns:
        strategy: distributed ScaleML training strategy appropriate for the model.
    """
    
    # Check if detected resources are provided, if not raise an error.
    if detected_resources is None:
        raise ValueError("Resources must be provided. Use the 'resources()' function to detect resources.")
        
    # Extract logical cores and GPU devices from detected resources.
    if resource_type in ['gpu', 'all'] and detected_resources.get('GPU devices') == 0:
        print("No GPUs available. Only using CPU resources.")
        resource_type = 'cpu'

    # extract framework assignments
    framework_model = framework.get('model')
    framework_strategy = framework.get('strategy')
    

    # Create a istributed strategy depending on the input framework
    if framework_strategy.lower() == 'tensorflow':
        
        if framework_model.lower() != 'tensorflow':
            raise ValueError("Incompatible model for TensorFlow strategy. Please use a TensorFlow model.")
        else:
            import tensorflow as tf
            
            # Create tensorflow friendly list of devices from detected resources
            logical_cores = [f'/cpu:{i}' for i in range(detected_resources.get('CPU cores', 0))]
            gpu_devices = [f'/gpu:{i}' for i in range(detected_resources.get('GPU devices', 0))]

            # TensorFlow strategy setup
            if resource_type == 'all':
                strategy = tf.distribute.MirroredStrategy(devices=logical_cores + gpu_devices)
            elif resource_type == 'gpu':
                strategy = tf.distribute.MirroredStrategy(devices=gpu_devices)
            else:
                strategy = tf.distribute.MirroredStrategy(devices=logical_cores)

            print(f"TensorFlow strategy initialized: {strategy}")
            return strategy

    elif framework_strategy.lower() == 'pytorch':
        
        if framework_model.lower() != 'pytorch':
            raise ValueError("Incompatible model for PyTorch strategy. Please use a PyTorch model.")
        else:
            import torch
            
            # PyTorch device setup (not yet assigning resources)
            if resource_type == 'all':
                strategy = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif resource_type == 'gpu':
                strategy = torch.device("cuda")
            else:
                strategy = torch.device("cpu")

            print(f"PyTorch device: {strategy}")
            return strategy

    # elif framework_strategy.lower() == 'horovod':
        
    #     import horovod.tensorflow as hvd_tf
    #     import horovod.torch as hvd_torch
        
    #     if framework_model.lower() == 'tensorflow':
    #         hvd_tf.init()
    #         print(f"Horovod TensorFlow rank: {hvd_tf.rank()}, size: {hvd_tf.size()}")
            
    #         if gpu_devices:
    #             import tensorflow as tf
    #             tf.config.experimental.set_visible_devices(gpu_devices[hvd_tf.local_rank()], 'GPU')
    #         return hvd_tf
        
    #     elif framework_model.lower() == 'pytorch':
    #         hvd_torch.init()
    #         print(f"Horovod PyTorch rank: {hvd_torch.rank()}, size: {hvd_torch.size()}")
            
    #         if gpu_devices:
    #             import torch
    #             torch.cuda.set_device(hvd_torch.local_rank())
    #         return hvd_torch

    else:
        raise ValueError("Unsupported framework or model. Choose from 'tensorflow' or 'pytorch'.")