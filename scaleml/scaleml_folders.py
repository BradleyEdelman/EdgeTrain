def scaleml_folders():

    import os

    # Get the current working directory (assumed to be within the 'notebooks' folder)
    cwd = os.getcwd()
    
    # Navigate two levels back from the notebooks folder to the root directory
    root_dir = os.path.dirname(os.path.dirname(cwd))
    
    # Define the base "ScaleML" directory path
    scaleml_dir = os.path.join(root_dir, "ScaleML")
    
    # Check if the ScaleML folder exists, and create it if not
    if not os.path.exists(scaleml_dir):
        os.makedirs(scaleml_dir)
    
    # Define the subfolders (models, logs, images)
    subfolders = ["models", "logs", "images"]
    
    # Create subfolders under ScaleML if they don't exist
    for subfolder in subfolders:
        subfolder_path = os.path.join(scaleml_dir, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

    # Return the path to the "ScaleML" folder
    return scaleml_dir


