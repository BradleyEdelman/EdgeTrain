import tensorflow as tf
import psutil, GPUtil
from optitrain import sys_resources

def adjust_threads(cpu_threshold=[20, 80], gpu_threshold=[20, 80], increment=1):
    """
    Dynamically adjust the number of inter-op and intra-op threads based on CPU and GPU computation usage.
    
    Parameters:
    - cpu_threshold (list): CPU computation usage upper and lower threshold (%) to trigger adjustment.
    - gpu_threshold (list): GPU computation usage upper and lower threshold (%) to trigger adjustment.
    - increment (int): Number of threads to increase or decrease at a time.
    
    Returns:
    - inter_threads (int): Updated inter-op threads.
    - intra_threads (int): Updated intra-op threads.
    """
    # Get current resource usage
    resources = sys_resources()
    cpu_compute_percent = resources.get('cpu_compute_percent')
    gpu_compute_percent = resources.get('gpu_compute_percent')

    # Get current threading settings
    inter_threads = tf.config.threading.get_inter_op_parallelism_threads()
    intra_threads = tf.config.threading.get_intra_op_parallelism_threads()

    # Set min and max thresholds for inter and intra threads
    min_threads = 1
    max_threads = resources.get('cpu_cores')

    if cpu_compute_percent > cpu_threshold[1] or gpu_compute_percent > gpu_threshold[1]:
        print(f"High resource usage detected: CPU={cpu_compute_percent}%, GPU={gpu_compute_percent}%")
        inter_threads = max(inter_threads - increment, min_threads)  # Reduce threads if above threshold
        intra_threads = max(intra_threads - increment, min_threads)  
    elif cpu_compute_percent < cpu_threshold[0] or gpu_compute_percent < gpu_threshold[0]:
        print(f"Low resource usage detected: CPU={cpu_compute_percent}%, GPU={gpu_compute_percent}%")
        inter_threads = min(inter_threads + increment, max_threads)  # Increase threads if below threshold
        intra_threads = min(intra_threads + increment, max_threads) 
    else:
        print(f"Resources are under control: CPU={cpu_compute_percent}%, GPU={gpu_compute_percent}%")
    
    # Apply adjusted thread settings
    tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    tf.config.threading.set_intra_op_parallelism_threads(intra_threads)

    print(f"Updated thread settings: inter-op = {inter_threads}, intra-op = {intra_threads}")

    return inter_threads, intra_threads


# Function to dynamically adjust batch size based on system resources
def adjust_batch_size(batch_size, cpu_threshold=[20, 80], gpu_threshold=[20, 80], increment=8):
    """
    Adjusts the batch size based on CPU and GPU memory usage.
    
    Parameters:
    - batch_size (int): The initial batch size to be adjusted.
    - cpu_threshold (list): CPU memory usage upper and lower threshold (%) to trigger adjustment.
    - gpu_threshold (list): GPU memory usage upper and lower threshold (%) to trigger adjustment.
    - increment (int): The amount by which to increase or decrease the batch size.
    
    Returns:
    - batch_size (int): The adjusted batch size.
    """
    resources = sys_resources()
    cpu_memory_percent = resources.get('cpu_memory_percent')
    gpu_memory_percent = resources.get('gpu_memory_percent')

    min_batch_size = 8
    max_batch_size = 128

    if cpu_memory_percent > cpu_threshold[1] or gpu_memory_percent > gpu_threshold[1]:
        print(f"High memory usage detected: CPU={cpu_memory_percent}%, GPU={gpu_memory_percent}%")
        batch_size_new = max(batch_size - increment, min_batch_size) # Reduce batch_size if above threshold
    elif cpu_memory_percent < cpu_threshold[0] and gpu_memory_percent < gpu_threshold[0]:
        print(f"Low memory usage detected: CPU={cpu_memory_percent}%, GPU={gpu_memory_percent}%")
        batch_size_new = min(batch_size + increment, max_batch_size) # Increase batch_size if below threshold
    else:
        print(f"Memory usage is under control: CPU={cpu_memory_percent}%, GPU={gpu_memory_percent}%")
        batch_size_new = batch_size # Keep batch_size the same

    print(f"Updated batch size:{batch_size_new}")

    batch_size = batch_size_new
    return batch_size 


def adjust_learning_rate(lr, cpu_threshold=[20, 80], gpu_threshold=[20, 80], increment=0.01):
    """
    Dynamically adjust the learning rate based on CPU and GPU resource usage.
    
    Parameters:
    - lr (float): Current learning rate.
    - cpu_threshold (list): CPU compute usage upper and lower threshold (%) to trigger adjustment.
    - gpu_threshold (list): GPU compute usage upper and lower threshold (%) to trigger adjustment.
    - increment (float): Learning rate adjustment per iteration.
    
    Returns:
    - adjusted_lr (float): Updated learning rate.
    """

    resources = sys_resources() 
    cpu_compute_percent = resources.get('cpu_compute_percent')
    gpu_compute_percent = resources.get('gpu_compute_percent')

    min_lr = 1e-6
    max_lr = 0.1

    # Adjust learning rate based on CPU/GPU compute
    if cpu_compute_percent > cpu_threshold[1] or gpu_compute_percent > gpu_threshold[1]:
        print(f"High resource usage detected: CPU={cpu_compute_percent}%, GPU={gpu_compute_percent}%")
        adjusted_lr = max(lr*(1-increment), min_lr)  # Decrease learning rate to slow down training
    elif cpu_compute_percent < cpu_threshold[0] or gpu_compute_percent < gpu_threshold[0]:
        print(f"Low resource usage detected: CPU={cpu_compute_percent}%, GPU={gpu_compute_percent}%")
        adjusted_lr = min(lr*(1+increment), max_lr)  # Increase learning rate to speed up training
    else:
        adjusted_lr = lr  # Keep the current learning rate if within acceptable thresholds
    
    print(f"Adjusted learning rate: {adjusted_lr}")

    lr = adjusted_lr
    return lr


def adjust_grad_accum(cpu_threshold=[20, 80], gpu_memory_threshold=[50, 90], current_grad_accum=1, increment=1):
    """
    Dynamically adjust the number of gradient accumulation steps based on memory usage and compute resource utilization.
    
    Parameters:
    - cpu_threshold (list): CPU computation usage upper and lower threshold (%) to trigger adjustment.
    - gpu_memory_threshold (list): GPU memory usage upper and lower threshold (%) to trigger adjustment.
    - current_grad_accum (int): Current number of gradient accumulation steps.
    - increment (int): Number of gradient accumulation steps to increase or decrease at a time.
    
    Returns:
    - grad_accum_steps (int): Updated number of gradient accumulation steps.
    """

    resources = sys_resources() 
    cpu_compute_percent = resources.get('cpu_compute_percent')
    gpu_memory_percent = resources.get('gpu_memory_percent')

    # Adjust gradient accumulation based on memory and compute
    if gpu_memory_percent > gpu_memory_threshold[1]:
        print(f"High GPU memory usage detected: {gpu_memory_percent}%")
        grad_accum_steps = min(current_grad_accum + increment, 16)  # Increase gradient accumulation (larger batch sim.)
    elif gpu_memory_percent < gpu_memory_threshold[0] and cpu_compute_percent < cpu_threshold[0]:
        print(f"Low resource usage detected: CPU={cpu_compute_percent}%, GPU={gpu_memory_percent}%")
        grad_accum_steps = max(current_grad_accum - increment, 1)  # Decrease gradient accumulation to speed up training
    else:
        grad_accum_steps = current_grad_accum 
    
    print(f"Adjusted gradient accumulation steps: {grad_accum_steps}")
    return grad_accum_steps
