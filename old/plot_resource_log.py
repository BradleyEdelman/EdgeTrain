def plot_resource_log(log_file=None):
    
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    if log_file is None:
        raise ValueError("Log file name must be provided.")
    
    # Load the resource usage log CSV file
    df = pd.read_csv(log_file)
    
    # Convert Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Plot CPU usage over time
    plt.figure(figsize=(14, 7))
    plt.plot(df['Timestamp'], df['CPU Usage'], label='CPU Usage')
    plt.xlabel('Time')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot GPU memory usage and utilization over time
    gpu_memory_cols = [col for col in df.columns if 'Memory Usage' in col]
    gpu_util_cols = [col for col in df.columns if 'Utilization' in col]
    
    plt.figure(figsize=(14, 7))
    for col in gpu_memory_cols:
        plt.plot(df['Timestamp'], df[col], label=col)
    plt.xlabel('Time')
    plt.ylabel('GPU Memory Usage (MB)')
    plt.title('GPU Memory Usage Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(14, 7))
    for col in gpu_util_cols:
        plt.plot(df['Timestamp'], df[col], label=col)
    plt.xlabel('Time')
    plt.ylabel('GPU Utilization (%)')
    plt.title('GPU Utilization Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Highlight bottlenecks
    cpu_bottleneck_threshold = 90  # Example threshold for CPU bottleneck
    gpu_bottleneck_threshold = 90  # Example threshold for GPU bottleneck
    
    cpu_bottlenecks = df[df['CPU Usage'] > cpu_bottleneck_threshold]
    gpu_bottlenecks = df[(df[gpu_util_cols] > gpu_bottleneck_threshold).any(axis=1)]
    
    print("CPU Bottlenecks:")
    display(cpu_bottlenecks)
    
    print("GPU Bottlenecks:")
    display(gpu_bottlenecks)