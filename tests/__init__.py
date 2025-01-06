import sys
import os

# Add the 'scaleml' directory to sys.path to make the modules accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scaleml')))

# Import modules from scaleml
from .resources import resources
from .strategies import strategies
from .log_resource_usage import log_resource_usage
from .plot_resource_log import plot_resource_log
from .distributed_train import distributed_train
from .create_model import create_model

# Import other testing libraries
import pytest
import tensorflow as tf
import psutil
import GPUtil




