def log_resource_usage(log_file, interval=10):
    """
    Log GPU and CPU resource usage to a log file.
    
    Parameters:
    - scaleml_folder: Directory to ScaleML folder.
    - interval: Time interval (in seconds) between logs. Default is 10 seconds.
    
    Returns:
    - log_file: The path to the log file where resource usage is saved.
    """
    
    import psutil, GPUtil, time, csv
    from datetime import datetime
    from scaleml import resources

    # Detect resources
    detected_resources = resources()
    cpu_count = detected_resources.get('CPU cores')

    # Create CSV header
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Timestamp']
        header.extend([f'CPU{i+1} Usage (%)' for i in range(cpu_count)])  # Add dynamic labels like CPU1, CPU2, etc.
        header.extend([f'GPU{i+1} Mem (MB)' for i in range(detected_resources.get('GPU devices', 0))])  # GPU memory usage
        header.extend([f'GPU{i+1} Util (%)' for i in range(detected_resources.get('GPU devices', 0))])  # GPU utilization
        writer.writerow(header)

    print(f"Logging resource usage to: {log_file}")

    # Log usage of all detected resources (even if not specified for use)
    while True:
        log_entry = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]

        # Log CPU usage
        cpu_percentages = psutil.cpu_percent(percpu=True, interval=interval)
        log_entry.extend(cpu_percentages)

        # Log GPU usage (if available)
        if detected_resources['GPU devices'] > 0:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                log_entry.extend([gpu.memoryUsed, gpu.memoryUtil * 100])
        else:
            log_entry.extend(['N/A', 'N/A'])  # Handle case when there are no GPUs

        # Write to CSV file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_entry)

        # Sleep for the interval before logging again
        time.sleep(interval)


def start_logging(log_file, interval=10):
    import multiprocessing, time
    
    log_process = multiprocessing.Process(target=log_resource_usage, args=(log_file, interval))
    log_process.start()
    return log_process

def stop_logging(log_process):
    log_process.terminate()
    log_process.join()
