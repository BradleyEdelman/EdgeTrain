import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from edgetrain import get_edgetrain_folder

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
    
    # Plot CPU and GPU usage over time on the same plot with workers on a separate y axis
    fig, ax1 = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    ax1[0].plot(df['Epoch #'], df['CPU Usage (%)'], label='CPU Usage (%)', color='tab:blue', linewidth=1.5)
    ax1[0].plot(df['Epoch #'], df['GPU Usage (%)'], label='GPU Usage (%)', color='tab:orange', linewidth=1.5)
    ax1[0].set_ylabel('Compute (%)')
    ax1[0].set_ylim(-5, 100)
    ax1[0].set_title('CPU and GPU Usage Over Time')
    ax1[0].legend(loc='upper left')
    ax1[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot CPU and GPU RAM usage over time on the same plot with batch size on a separate y axis
    ax1[1].plot(df['Epoch #'], df['CPU RAM (%)'], label='CPU RAM (%)', color='tab:blue', linewidth=1.5)
    ax1[1].plot(df['Epoch #'], df['GPU RAM (%)'], label='GPU RAM (%)', color='tab:orange', linewidth=1.5)
    ax1[1].set_ylabel('RAM (%)')
    ax1[1].set_title('CPU and GPU RAM Usage Over Time')
    ax1[1].legend(loc='upper left')
    ax1[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Plot Batch Size, Learning Rate, and Grad Accum over time on the same plot
    ax2 = ax1[2].twinx()
    ax1[2].plot(df['Epoch #'], df['Batch Size'], label='Batch Size', color='tab:green', linewidth=1.5)
    ax1[2].plot(df['Epoch #'], df['Grad Accum'], label='Grad Accum', color='tab:red', linewidth=1.5)
    ax1[2].set_ylim(0, 50)
    ax2.plot(df['Epoch #'], df['Learning Rate'], label='Learning Rate', color='tab:purple', linewidth=1.5)
    ax2.set_ylabel('Values (Learning Rate)')
    ax2.set_ylim(0, .0025)
    ax1[2].set_xlabel('Epoch #')
    ax1[2].set_ylabel('Values (Batch size, Grad accum)')
    ax1[2].set_title('Batch Size, Learning Rate, and Grad Accum Over Time')
    ax1[2].legend(loc='upper left')
    ax1[2].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set x tick marks as integers every 5
    for ax in ax1:
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):d}'))

    plt.tight_layout()
    plt.show()
    fig.canvas.draw()

    # Save the figure to the images folder
    edgetrain_folder = get_edgetrain_folder()
    img_dir = f'{edgetrain_folder}/images/'
    timestamp = '_'.join(log_file.split('/')[-1].split('_')[:2])
    fig.savefig(f'{img_dir}/{timestamp}_resource_usage_plot.png')


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


def training_history_plot(history_list, log_file):
    """
    Plot the training loss and accuracy over epochs.

    Parameters:
    - history_list: List of dictionaries containing 'accuracy' and 'loss' for each epoch.
      Example: [{'accuracy': 0.8, 'loss': 0.5}, {'accuracy': 0.85, 'loss': 0.4}, ...]
    - log_file: The path to the log file (CSV format) that contains the resource usage data.

    Returns:
    - None
    """
    
    accuracy_values = np.array([epoch['accuracy'] for epoch in history_list])
    loss_values = np.array([epoch['loss'] for epoch in history_list])

    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='red')
    ax1.plot(range(1, len(loss_values) + 1), loss_values, 'r', label='Loss')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_ylim(0, 15)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='black')
    ax2.plot(range(1, len(accuracy_values) + 1), accuracy_values * 100, 'k', label='Accuracy (%)')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, 100)

    fig.tight_layout()
    plt.title('Training Loss and Accuracy')
    plt.show()
    fig.canvas.draw()

    # Save the figure to the images folder
    edgetrain_folder = get_edgetrain_folder()
    img_dir = f'{edgetrain_folder}/images/'
    timestamp = '_'.join(log_file.split('/')[-1].split('_')[:2])
    fig.savefig(f'{img_dir}/{timestamp}_training_history_plot.png')