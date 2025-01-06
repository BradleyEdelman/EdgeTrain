import sys
import os

# Add the 'scaleml' directory to sys.path to make the modules accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scaleml')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scaleml')))

# Import modules from scaleml
from scaleml.resources import resources
from scaleml.strategies import strategies
from scaleml.log_resource_usage import log_resource_usage
from scaleml.plot_resource_log import plot_resource_log
from scaleml.distributed_train import distributed_train
from scaleml.create_model import create_model

# Import other testing libraries
import pytest
import tensorflow as tf
# import horovod
# import horovod.tensorflow as hvd_tf
# import horovod.torch as hvd_torch
import psutil
import GPUtil




