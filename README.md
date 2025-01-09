# EdgeTrain: Automated Resource Adjustment for Efficient Edge AI Training  
**Version: 0.1.0-alpha**

EdgeTrain is a Python package designed to dynamically adjust training parameters and strategies based on CPU and GPU performance. It optimizes the training process by adjusting batch size and learning rate to ensure efficient training without overutilizing or underutilizing available resources. This package is specifically designed to reduce memory usage for model training on edge AI devices, laptops or other setups that have limited memory.  

## Features

### Automated Resource Adjustment
EdgeTrain currently adjusts the following hyperparameters based on CPU/GPU usage:
- **Batch Size**: Automatically adjusts batch size for better memory optimization based on resource usage.
- **Learning Rate**: Dynamically adjusts the learning rate to improve training efficiency.

These adjustments optimize resource utilization throughout training, enabling efficient use of available resources on edge AI devices.

### Resource Logging & Visualization
EdgeTrain logs critical system metrics (e.g., CPU and GPU usage) and training parameters (batch size, learning rate) for each epoch. The logs enable post-hoc visualization and analysis of:
- Resource utilization over time.
- Training parameter adjustments across epochs.
- Correlations between resource usage and model performance.
  
The built-in **visualization tools** help you understand how system resources are being utilized and how training parameters evolve during training.

### Customizable
EdgeTrain is highly customizable. You can easily modify:
- **Resource Adjustment Thresholds**: Set CPU/GPU usage ranges to trigger adjustments.
- **Training Configuration Settings**: Adjust batch size increment, learning rate adjustments, and more.
- Tailor the optimization process to fit various setups, especially on edge devices with limited resources.

## Installation
You can install EdgeTrain via pip:

```bash
pip install edgetrain
```

Alternatively, clone the repository and install manually:

```
# Clone the repository
git clone https://github.com/BradleyEdelman/edgetrain.git
cd edgetrain

# Install the package
pip install .
```

## Usage
To use EdgeTrain, simply import the package and configure your training environment. Below is an example of using EdgeTrain with a TensorFlow model:
```
# Import library
import edgetrain

# Example of resource monitoring and training with dynamic adjustments
train_dataset = {'images': train_images, 'labels': train_labels}
history = edgetrain.dynamic_train(train_dataset, epochs=10, batch_size=32, lr=1e-3, log_file="resource_log.csv", dynamic_adjustments=True)
```

## File Tree
EdgeTrain/
├── edgetrain/
│   ├── __init__.py
│   ├── dynamic_train.py
│   ├── resource_monitor.py
│   ├── resource_adjust.py
│   ├── log_analysis.py
├── tests/
│   ├── test_dynamic_train.py
│   ├── test_resource_monitor.py
│   ├── test_resource_adjust.py
├── example_notebooks/
│   ├── EdgeTrain_tf_ex.ipynb  # Example notebook for using EdgeTrain with TensorFlow
├── setup.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore

## Contributions
You can contribute by:
- Reporting bugs or requesting features: [GitHub Issues](https://github.com/BradleyEdelman/edgetrain/issues)
- Improving the documentation
- Adding new training strategies or features

## Reporting bugs or requesting features.
Improving the documentation.
Adding new training strategies or features (e.g., model pruning, quantization, support for additional frameworks).
License
This project is licensed under the MIT License - see the LICENSE file for details.

## Known Limitations (Alpha)
The package currently supports TensorFlow only. Support for other frameworks like PyTorch is planned for future releases.
Model pruning and quantization are future features.
Resource usage thresholds for dynamic adjustments are in the initial phase and may require tuning based on the training setup.
