import tensorflow as tf
import psutil
import GPUtil

# Function to dynamically adjust workers based on resource usage
def adjust_workers(cpu_threshold=80, gpu_threshold=80):
    """
    Adjusts the number of workers based on CPU and GPU usage.
    
    Returns:
    - num_workers: Number of workers to use.
    """
    cpu_percent, gpu_memory_usage, gpu_memory_total, gpu_percent = get_system_resources()
    
    if cpu_percent > cpu_threshold or gpu_percent > gpu_threshold:
        print(f"High resource usage detected: CPU={cpu_percent}%, GPU={gpu_percent}%")
        # Reduce the number of workers if resources are high
        num_workers = 1
    else:
        print(f"Resources are under control: CPU={cpu_percent}%, GPU={gpu_percent}%")
        # Increase the number of workers if resources are available
        num_workers = 2
    
    return num_workers

# Function to dynamically adjust batch size based on system resources
def adjust_batch_size(cpu_percent, gpu_percent, base_batch_size=32):
    """
    Adjusts the batch size based on CPU and GPU usage.
    
    Returns:
    - batch_size: The adjusted batch size.
    """
    if cpu_percent > 80 or gpu_percent > 80:
        return base_batch_size // 2  # Reduce batch size if resources are overused
    return base_batch_size * 2  # Increase batch size if resources are available
