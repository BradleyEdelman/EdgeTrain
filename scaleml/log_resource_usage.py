def log_resource_usage(devices, interval=10):
    """
    Log GPU and CPU resource usage.
    
    Parameters:
    - devices: List of devices (e.g., GPU/CPU).
    - interval: Time interval (in seconds) between logs.
    """
    
    import psutil, GPUtil, time
    from datetime import datetime
    
    # Generate log file name with current date
    log_file = f"resource_usage_log_{datetime.now().strftime('%Y%m%d')}.txt"
    
    # Log the available devices (GPU/CPU)
    print("Logging Resource Usage...")
    for device in devices:
        print(f"Device: {device}")
    
    with open(log_file, 'a') as f:
        # Monitor the resources in real-time
        while True:
            log_entry = "\n-- Resource Usage --\n"
            
            # Log CPU usage
            cpu_usage = psutil.cpu_percent(interval=interval)
            log_entry += f"CPU Usage: {cpu_usage}%\n"
            
            # Log GPU usage (if available)
            if devices and devices[0].device_type == 'GPU':
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    log_entry += f"GPU {gpu.id}: Memory Usage {gpu.memoryUsed}/{gpu.memoryTotal}MB, GPU Utilization {gpu.memoryUtil*100}%\n"
            
            # Print and save the log entry
            print(log_entry)
            f.write(log_entry)
            
            # Sleep for the interval before logging again
            time.sleep(interval)