import psutil, GPUtil, time, csv
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
def log_usage_once(log_file, batch_size, num_workers, num_epoch=0):
    """
    Log GPU and CPU resource usage once.
    
    Parameters:
    - log_file: Path to the log file.
    - batch_size: Current batch size.
    - num_workers: Number of workers.
    - num_epoch: Current epoch number.
    """
    
    # Create CSV header if the file doesn't exist
    try:
        with open(log_file, 'r') as f:
            pass
    except FileNotFoundError:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'Timestamp', 'Epoch #', 'CPU Usage (%)', 'GPU Memory Used (MB)', 
                'GPU Memory Total (MB)', 'GPU Usage (%)', 'Batch Size', 'Num Workers'
            ]
            writer.writerow(header)

    # Get resource usage
    cpu_percent, gpu_memory_usage, gpu_memory_total, gpu_percent = sys_resources()
    
    # Prepare log entry
    log_entry = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_epoch,
        cpu_percent,
        gpu_memory_usage,
        gpu_memory_total,
        gpu_percent,
        batch_size,
        num_workers
    ]

    # Append log entry to the file
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log_entry)
