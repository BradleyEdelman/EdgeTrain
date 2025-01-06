import psutil
import GPUtil
import time
import csv
from datetime import datetime

# Function to monitor system resources
def sys_resources():
    # Check CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Check GPU usage
    gpus = GPUtil.getGPUs()
    gpu_memory_usage = sum(gpu.memoryUsed for gpu in gpus)
    gpu_memory_total = sum(gpu.memoryTotal for gpu in gpus)
    gpu_percent = sum(gpu.memoryUtil for gpu in gpus) / len(gpus) if gpus else 0
    
    return cpu_percent, gpu_memory_usage, gpu_memory_total, gpu_percent

# Function to log resource usage and batch size
def log_usage(log_file, batch_size, num_workers, interval=10):
    """
    Log GPU and CPU resource usage along with batch size and number of workers to a log file.
    
    Parameters:
    - log_file: The path to the log file where resource usage is saved.
    - batch_size: The current batch size used during training.
    - num_workers: The current number of workers used during training.
    - interval: Time interval (in seconds) between logs. Default is 10 seconds.
    
    Returns:
    - None
    """
    # Create CSV header if file is empty
    try:
        with open(log_file, 'r') as f:
            pass
    except FileNotFoundError:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Timestamp', 'CPU Usage (%)', 'GPU Memory Used (MB)', 'GPU Memory Total (MB)', 'GPU Usage (%)', 'Batch Size', 'Num Workers']
            writer.writerow(header)

    # Log usage of all detected resources (even if not specified for use)
    while True:
        log_entry = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]

        # Log CPU usage
        cpu_percent, gpu_memory_usage, gpu_memory_total, gpu_percent = sys_resources()
        log_entry.extend([cpu_percent, gpu_memory_usage, gpu_memory_total, gpu_percent, batch_size, num_workers])

        # Write to CSV file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_entry)

        # Sleep for the interval before logging again
        time.sleep(interval)

# # Example usage:
# # Dummy values for batch size and workers to simulate non-adaptive scenario
# batch_size = 32  # Default batch size
# num_workers = 2  # Default number of workers

# # Log resource usage without adaptation (non-adaptive logging)
# log_file = "non_adaptive_resource_log.csv"
# log_resource_usage(log_file, batch_size, num_workers)
