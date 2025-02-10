import psutil, GPUtil, csv
from datetime import datetime
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlShutdown

def sys_resources():
    """
    Monitor system resources, including CPU and GPU utilization and memory usage.

    Returns:
    - dict: A dictionary containing the following keys:
        - cpu_cores (int): Number of logical CPU cores.
        - cpu_compute_percent (float): CPU utilization as a percentage.
        - cpu_memory_percent (float): RAM usage as a percentage.
        - gpu_compute_percent (float): Average GPU compute utilization as a percentage.
        - gpu_memory_usage (float): Total GPU memory used across all GPUs (in MB).
        - gpu_memory_total (float): Total available GPU memory across all GPUs (in MB).
        - gpu_memory_percent (float): Average GPU memory utilization as a percentage.
        - num_gpus (int): Number of GPUs available.
    """

    # Check CPU usage (compute and RAM)
    cpu_compute_percent = psutil.cpu_percent(interval=1)
    cpu_cores = psutil.cpu_count(logical=True)
    
    # Check GPU usage (memory and compute)
    gpus = GPUtil.getGPUs()
    num_gpus = len(gpus)
    gpu_memory_usage = sum(gpu.memoryUsed for gpu in gpus)
    gpu_memory_total = sum(gpu.memoryTotal for gpu in gpus)
    gpu_memory_percent = sum(gpu.memoryUtil for gpu in gpus) / num_gpus if gpus else 0
    
    # GPU compute utilization
    gpu_compute_percent = 0
    if num_gpus > 0:
        nvmlInit()
        try:
            gpu_compute_percent = sum(nvmlDeviceGetUtilizationRates(nvmlDeviceGetHandleByIndex(i)).gpu for i in range(num_gpus)) / num_gpus
        finally:
            nvmlShutdown()
    
    # Check system memory usage (RAM)
    cpu_memory_percent = psutil.virtual_memory().percent
    
    return {
        "cpu_cores": cpu_cores,
        "cpu_compute_percent": cpu_compute_percent,
        "cpu_memory_percent": cpu_memory_percent,
        "gpu_compute_percent": gpu_compute_percent,
        "gpu_memory_usage": gpu_memory_usage,
        "gpu_memory_total": gpu_memory_total,
        "gpu_memory_percent": gpu_memory_percent,
        "num_gpus": num_gpus
    }


def log_usage_once(log_file, pruning, batch_size, lr, normalize_scores, priority_value, num_epoch=0, resources=None):
    """
    Log GPU and CPU resource usage once.
    
    Parameters:
    - log_file (str): Path to the log file.
    - pruning (bool): Whether pruning is enabled.
    - batch_size (int): Current batch size.
    - lr (float): Learning rate.
    - normalize_scores (dict): Dictionary of normalized scores.
    - priority_value (dict): Dictionary of priority values.
    - num_epoch (int, optional): Current epoch number. Default is 0.
    - resources (dict, optional): Dictionary containing system resource usage metrics. If None, system resources will be fetched.
    """
    
    # Create CSV header if the file doesn't exist
    try:
        with open(log_file, 'r') as f:
            pass
    except FileNotFoundError:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'Timestamp', 'Epoch #', 'CPU Usage (%)', 'CPU RAM (%)',
                'GPU RAM (%)', 'GPU Usage (%)',
                'Mem Score', 'Acc Score',
                'Priority Batch Size', 'Priority Learning Rate',
                'Pruning', 'Batch Size', 'Learning Rate', 
            ]
            writer.writerow(header)

    # Get resource usage
    if resources is None:
        resources = sys_resources()
    cpu_compute_percent = resources.get('cpu_compute_percent')
    cpu_memory_percent = resources.get('cpu_memory_percent')
    gpu_compute_percent = resources.get('gpu_compute_percent')
    gpu_memory_percent = resources.get('gpu_memory_percent')

    memory_score = normalize_scores.get('memory_score')
    accuracy_score = normalize_scores.get('accuracy_score')
    batch_size_priority_value = priority_value.get('batch_size')
    learning_rate_priority_value = priority_value.get('learning_rate')

    # Prepare log entry
    log_entry = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_epoch,
        cpu_compute_percent,
        cpu_memory_percent,
        gpu_compute_percent,
        gpu_memory_percent,
        memory_score,
        accuracy_score,
        batch_size_priority_value,
        learning_rate_priority_value,
        pruning,
        batch_size,
        lr
    ]

    # Append log entry to the file
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log_entry)