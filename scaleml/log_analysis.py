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
    
    # Convert the 'Timestamp' column to datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Create the figure and axes for plotting
    plt.figure(figsize=(14, 10))

    # Plot CPU usage over time
    plt.subplot(3, 1, 1)
    plt.plot(df['Timestamp'], df['CPU Usage (%)'], label='CPU Usage (%)', color='tab:blue')
    plt.xlabel('Timestamp')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage Over Time')
    plt.legend()

    # Plot GPU usage over time
    plt.subplot(3, 1, 2)
    plt.plot(df['Timestamp'], df['GPU Usage (%)'], label='GPU Usage (%)', color='tab:orange')
    plt.xlabel('Timestamp')
    plt.ylabel('GPU Usage (%)')
    plt.title('GPU Usage Over Time')
    plt.legend()

    # Plot Batch Size and Number of Workers over time
    plt.subplot(3, 1, 3)
    plt.plot(df['Timestamp'], df['Batch Size'], label='Batch Size', color='tab:green', linestyle='--')
    plt.plot(df['Timestamp'], df['Num Workers'], label='Num Workers', color='tab:red', linestyle='--')
    plt.xlabel('Timestamp')
    plt.ylabel('Batch Size / Num Workers')
    plt.title('Batch Size and Number of Workers Over Time')
    plt.legend()

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
