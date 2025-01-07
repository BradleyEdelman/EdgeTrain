import matplotlib.pyplot as plt
import pandas as pd

def log_usage_plot(log_file):
    """
    Load the resource usage log from the CSV file and plot CPU and GPU usage, 
    as well as batch size and number of workers over time (epochs).
    
    Parameters:
    - log_file: The path to the log file (CSV format) that contains the resource usage data.
    
    Returns:
    - None
    """
    # Load the log file into a DataFrame
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Log file '{log_file}' not found.")
        return
    
    # Create the figure and axes for plotting
    plt.figure(figsize=(14, 10))

    # Plot CPU and GPU usage over time on the same plot with workers on a separate y axis
    fig, ax1 = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df['Epoch #'], df['CPU Usage (%)'], label='CPU Usage (%)', color='tab:blue')
    ax1.plot(df['Epoch #'], df['GPU Usage (%)'], label='GPU Usage (%)', color='tab:orange')
    ax1.set_xlabel('Epoch #')
    ax1.set_ylabel('Usage (%)')
    ax1.set_title('CPU and GPU Usage Over Time')
    ax1.legend(loc='upper left')

    # Plot CPU and GPU RAM usage over time on the same plot with batch size on a separate y axis
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(df['Epoch #'], df['CPU RAM (%)'], label='CPU RAM (%)', color='tab:blue')
    ax2.plot(df['Epoch #'], df['GPU RAM (%)'], label='GPU RAM (%)', color='tab:orange')
    ax2.set_xlabel('Epoch #')
    ax2.set_ylabel('RAM (%)')
    ax2.set_title('CPU and GPU RAM Usage Over Time')
    ax2.legend(loc='upper left')

    # Plot Batch Size, Learning Rate, and Grad Accum over time on the same plot
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(df['Epoch #'], df['Batch Size'], label='Batch Size', color='tab:green')
    ax3.plot(df['Epoch #'], df['Learning Rate'], label='Learning Rate', color='tab:blue')
    ax3.plot(df['Epoch #'], df['Grad Accum'], label='Grad Accum', color='tab:orange')
    ax3.set_xlabel('Epoch #')
    ax3.set_ylabel('Values')
    ax3.set_title('Batch Size, Learning Rate, and Grad Accum Over Time')
    ax3.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def log_train_time(log_file):
    """
    Calculate and print the total training time from the log file based on timestamps.
    
    Parameters:
    - log_file: The path to the log file (CSV format) containing the timestamps.
    
    Returns:
    - None
    """
    # Load the log file into a DataFrame
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Log file '{log_file}' not found.")
        return
    
    # Convert the 'Timestamp' column to datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Get the first and last timestamps from the log
    start_time = df['Timestamp'].iloc[0]
    end_time = df['Timestamp'].iloc[-1]

    # Calculate the total training time
    total_training_time = end_time - start_time
    print(f"Total Training Time: {total_training_time}")

    return total_training_time