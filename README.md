# EdgeTrain: Automated Resource Adjustment for Efficient Edge AI Training 
**Version: 1.0.0** 

EdgeTrain is a Python package designed to dynamically adjust training parameters and strategies based on CPU and GPU performance. It optimizes the training process by adjusting batch size, model pruning strategies, and gradient accumulation steps to ensure efficient training without overutilizing or underutilizing available resources. This package is specifically designed to reduce memory usage for model training on edge AI devices, such as the Jetson Orin Nano (Plus), that have limited RAM. <br />

## Features
### Automated Resource Adjustment
EdgeTrain dynamically adjusts the following hyperparameters based on CPU/GPU usage:
- Batch Size: Automatically adjusts batch size for better memory optimization.
- Model Pruning: Removes unimportant model weights to reduce RAM usage.
- Gradient Accumulation Steps: Adjusts gradient accumulation steps to balance GPU memory usage and performance.
These adjustments optimize resource utilization throughout training, enabling efficient use of available resources.
<br />

### Resource Logging & Visualization
EdgeTrain logs critical system metrics (e.g., CPU and GPU usage) and training parameters (batch size, model weights, gradient accumulation steps) for each epoch. The logs enable post-hoc visualization and analysis of:
- Resource utilization over time
- Training parameter adjustments across epochs
- Correlations between resource usage and model performance <br />

### Customizable
EdgeTrain is highly customizable. You can easily modify:
- Resource Adjustment Thresholds: Set CPU/GPU usage ranges to trigger adjustments.
- Training Configuration Settings: Adjust batch size increment, model pruning strategies, and gradient accumulation steps.
- Tailor the optimization process to fit your various setups with limited resources. <br />

## Installation
You can install EdgeTrain via pip:

```bash
# Copy code
pip install edgetrain
```

Alternatively, clone the repository and install manually:
```bash
# Clone the repository
git clone https://github.com/BradleyEdelman/edgetrain.git
cd optitrain

# Install the package
pip install 
```

## Usage
To use EdgeTrain, simply import the package and configure your training environment:

```python
# Import library
import optitrain
```

## File Tree
```bash
EdgeTrain/
├── edgetrain/
│   ├── __init__.py
│   ├── dynamic_train.py
│   ├── resource_monitor.py
│   ├── resource_adjust.py
│   ├── log_analysis.py
├── tests/
│   ├── 
│   ├── 
├── example_notebooks/
│   ├── EdegTrain_tf_ex.ipynb
│   ├── 
│   ├── 
├── setup.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
```

## Contributions
Report bugs or request features <br />
Improve the documentation <br />
Add new training strategies or features <br />

## License
This project is licensed under the MIT License - see the LICENSE file for details.
