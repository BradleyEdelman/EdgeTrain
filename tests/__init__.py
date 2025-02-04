import sys
import os

# Debug: Print the path being added (for troubleshooting)
edgetrain_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'edgetrain'))

# Ensure edgetrain is in the path BEFORE importing anything
if edgetrain_path not in sys.path:
    sys.path.insert(0, edgetrain_path)

# Import pytest (only needed here)
import pytest

