import os


def get_edgetrain_folder():
    """
    Create the necessary folder structure for the EdgeTrain project.

    This function navigates from the current working directory to the root directory,
    then creates an "EdgeTrain" directory with subdirectories for models, logs, and images
    if they do not already exist.

    Returns:
    - str: The path to the "EdgeTrain" directory.
    """

    # Get the current working directory (assumed to be within the 'notebooks' folder)
    cwd = os.getcwd()

    # Navigate two levels back from the notebooks folder to the root directory
    root_dir = os.path.dirname(os.path.dirname(cwd))

    # Define the base "EdgeTrain" directory path
    edgetrain_dir = os.path.join(root_dir, "EdgeTrain")

    # Check if the EdgeTrain folder exists, and create it if not
    if not os.path.exists(edgetrain_dir):
        os.makedirs(edgetrain_dir)

    # Define the subfolders
    subfolders = ["models", "logs", "images", "results"]

    # Create subfolders under EdgeTrain if they don't exist
    for subfolder in subfolders:
        subfolder_path = os.path.join(edgetrain_dir, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

    # Return the path to the "EdgeTrain" folder
    return edgetrain_dir
