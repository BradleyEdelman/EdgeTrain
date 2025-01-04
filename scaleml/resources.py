def resources():

    import GPUtil
    import psutil

    # Get the number of logical CPU cores
    logical_cores = psutil.cpu_count(logical=True)

    # List available GPU devices 
    gpu_devices = GPUtil.getGPUs()

    # Create a list of devices
    devices = [f"/cpu:{i}" for i in range(logical_cores)]  # Use one device per logical core for CPUs
    devices.extend([f"/gpu:{i}" for i in range(len(gpu_devices))])  # Add all available GPUs

    print(f"Number of logical CPU cores: {logical_cores}")
    print(f"Available GPU devices: {gpu_devices}")
    print(f"All devices (CPUs and GPUs): {devices}")
    
    return devices