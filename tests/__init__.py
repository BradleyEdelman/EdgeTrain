import sys
import os

# Add the 'edgetrain' directory to sys.path to make the modules accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'edgetrain')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'edgetrain')))

# Import modules from scaleml
from edgetrain.resource_monitor import sys_resources, log_usage_once
from edgetrain.calculate_scores import compute_scores, normalize_scores
from edgetrain.calculate_priorities import define_priorities
from edgetrain.adjust_train_parameters import adjust_training_parameters
from edgetrain.edgetrain_folder import get_edgetrain_folder
from edgetrain.train_visualize import log_usage_plot, log_train_time, training_history_plot
from edgetrain.create_model import create_model_tf, create_model_torch
from edgetrain.dynamic_train import dynamic_train

# Import other testing libraries
import pytest
import tensorflow as tf
import psutil
import GPUtil




