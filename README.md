# EdgeTrain: Automated Resource Adjustment for Efficient Edge AI Training  
**Version: 0.1.1-alpha**

EdgeTrain is a Python package designed to dynamically adjust deep learning training parameters and strategies based on CPU and GPU performance. It optimizes the training process by adjusting batch size and learning rate to ensure efficient training without overutilizing or underutilizing available resources. This package is specifically designed to reduce memory usage for model training on edge AI devices, laptops or other setups that have limited memory.  

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

## Release Notes
Version: 0.1.1-alpha
- Fixed circular import issue in `create_model.py`. Now users should not encounter import errors during initialization.

## Installation
You can install the latest version of EdgeTrain via pip:

```bash
pip install https://github.com/BradleyEdelman/EdgeTrain/releases/download/v0.1.1-alpha/edgetrain-0.1.1a0.tar.gz
```

Alternatively, clone the repository and install manually:

```
# Clone the repository
git clone https://github.com/BradleyEdelman/edgetrain.git

# Checkout the desired version
cd edgetrain
git checkout tags/v0.1.1-alpha

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
```
EdgeTrain/
├── edgetrain/
│   ├── __init__.py
│   ├── create_model.py
│   ├── dynamic_train.py
│   ├── edgetrain_folder
│   ├── resource_adjust.py
│   ├── resource_monitor.py
│   ├── train_visualize.py
├── tests/
│   ├── __init__.py
│   ├── test_adjust_batch_size.py
│   ├── test_adjust_learning_rate.py
│   ├── test_create_model_tf.py
│   ├── test_log_usage_once.py
│   ├── test_sys_resources.py
│   ├── test_dynamic_train.py
├── example_notebooks/
│   ├── EdgeTrain_example.ipynb
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── README.md
├── requirements-dev.txt
├── requirements.txt
├── setup.py
```

## Contributions
You can contribute by:
- Reporting bugs or requesting features: [GitHub Issues](https://github.com/BradleyEdelman/edgetrain/issues)

## Reporting bugs or requesting features.
Improving the documentation.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Known Limitations (Alpha)
- The package currently supports TensorFlow only. Support for other frameworks, especially lightweight ones is planned for future releases.
- Model pruning and quantization are future features.
- Resource usage thresholds for dynamic adjustments are in the initial phase and may require tuning based on the training setup.
