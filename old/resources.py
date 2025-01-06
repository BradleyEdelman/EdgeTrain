def resources():
    """
    Sets up a distributed training strategy based on the chosen framework, available resources, and specified devices.

    Returns:
        detected_resources (dict): A dictionary containing the detected resources. 
                         Expected keys: 'logical_cores', 'gpu_devices'.
    """

    import GPUtil
    import psutil

    # Get the number of logical CPU cores
    logical_cores = psutil.cpu_count(logical=True)

    # List available GPU devices 
    gpu_devices = GPUtil.getGPUs()

    # Create a dict of devices
    detected_resources = {
        "CPU cores": logical_cores,
        "GPU devices": len(gpu_devices)
    }

    print(f"Number of logical CPU cores: {logical_cores}")
    print(f"Number of GPU devices: {len(gpu_devices)}")
    print(f"All resources: {logical_cores} CPU cores, {len(gpu_devices)} GPU devices")
    
    return detected_resources