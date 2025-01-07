import sys
import os

# Add the 'scaleml' directory to sys.path to make the modules accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scaleml')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scaleml')))

# Import modules from scaleml
from scaleml.resource_monitor import sys_resources, log_usage_once
from scaleml.resource_adjust import adjust_workers, adjust_batch_size
from scaleml.log_analysis import log_usage_plot, log_train_time
from scaleml.scaleml_folders import scaleml_folders
from scaleml.create_model import create_model_tf, create_model_torch
from scaleml.dynamic_train import dynamic_train

# Import other testing libraries
import pytest
import tensorflow as tf
import psutil
import GPUtil




