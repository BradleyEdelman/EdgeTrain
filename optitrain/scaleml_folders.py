import os


def optitrain_folders():
    """
    Creates the necessary folder structure for the OptiTrain project.

    This function navigates from the current working directory to the root directory,
    then creates a "OptiTrain" directory with subdirectories for models, logs, and images
    if they do not already exist.

    Returns:
        str: The path to the "OptiTrain" directory.
    """

    # Get the current working directory (assumed to be within the 'notebooks' folder)
    cwd = os.getcwd()
    
    # Navigate two levels back from the notebooks folder to the root directory
    root_dir = os.path.dirname(os.path.dirname(cwd))
    
    # Define the base "OptiTrain" directory path
    optitrain_dir = os.path.join(root_dir, "OptiTrain")
    
    # Check if the OptiTrain folder exists, and create it if not
    if not os.path.exists(optitrain_dir):
        os.makedirs(optitrain_dir)
    
    # Define the subfolders (models, logs, images)
    subfolders = ["models", "logs", "images"]
    
    # Create subfolders under OptiTrain if they don't exist
    for subfolder in subfolders:
        subfolder_path = os.path.join(optitrain_dir, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

    # Return the path to the "OptiTrain" folder
    return optitrain_dir