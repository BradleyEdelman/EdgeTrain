# EdgeTrain: Automated Resource Adjustment for Efficient Edge AI Training  
**Version: 0.2.0**

EdgeTrain is a Python package designed to dynamically adjust deep learning training parameters and strategies based on CPU and GPU performance. It optimizes the training process by adjusting batch size and learning rate to ensure efficient training without overutilizing or underutilizing available resources. This package is specifically designed to reduce memory usage for model training on edge AI devices, laptops or other setups that have limited memory.  

## Features

### Dynamic Resource-Based Training Adjustments
EdgeTrain monitors CPU and GPU usage in real-time and automatically adjusts hyperparameters during training:
- **Batch Size**: Increases or decreases to optimize memory usage.
- **Learning Rate**:  Adjusts based on model performance to improve training efficiency.

### Resource Logging & Visualization
EdgeTrain logs system performance and training parameters, allowing post-hoc visualization of:
- Resource utilization over time.
- Training parameter adjustments across epochs.
- Correlations between resource usage and model performance.
  
The provided visualization tools help you understand how system resources are being utilized and how training parameters evolve during training.

### Customization and control
EdgeTrain is highly customizable. You can easily modify:
- **Resource Adjustment Thresholds**: Set CPU/GPU usage ranges to trigger adjustments.
- **Training Configuration Settings**: Adjust batch size increment, learning rate adjustments, and more.
- **Fixed Pruning Strategy**: Pruning is applied with a constant ratio and stripped at the end to improve deployment efficiency.

## Release Notes for v0.2.0
This version introduces a **refined adaptive training strategy with a constant pruning ratio**. Key updates:

- **Score Calculation**: This version now computes an **accuracy score** and a **memory score** based on resource usage and model performance
- **Parameter Prioritization**: Accuracy and memory scores are weighted according to default or user-defined priority weighting to idenfity a priority list for parameter adjustment. Now, only the top priority paramater is adjusted in each epoch.
    - **Batch size priority** is weighted by memory usage.
    - **Learning rate priority** is inversely weighted by accuracy improvement (i.e. increases if accuracy stagnates).
- **Fixed Pruning Ratio**: Pruning is constant and is stripped at the end.
- **Code Quality Improvements**: Added pre-commit hooks and CI linting for consistency.

## Installation
You can install the latest version of EdgeTrain via pip:

```bash
pip install https://github.com/BradleyEdelman/EdgeTrain/releases/download/v0.2.0-alpha/edgetrain-0.2.0.tar.gz
```

Alternatively, clone the repository and install manually:

```
# Clone the repository
git clone https://github.com/BradleyEdelman/edgetrain.git

# Checkout the desired version
cd edgetrain
git checkout tags/v0.2.0

# Install the package
pip install .
```

## Usage Example
To use EdgeTrain, simply import the package and configure your training environment. Below is an example of using EdgeTrain with a TensorFlow model:
```
import edgetrain

# Example of resource monitoring and training with dynamic adjustments
train_dataset = {'images': train_images, 'labels': train_labels}
final_model, history = edgetrain.dynamic_train(
    train_dataset, 
    epochs=10, 
    batch_size=32, 
    lr=1e-3, 
    log_file="resource_log.csv", 
    dynamic_adjustments=True
)

# Plot resource usage, parameter scoring and prioritization, and parameter values over time
edgetrain.log_usage_plot("resource_log.csv")
```

## File Tree
```
EdgeTrain/
├── edgetrain/
│   ├── __init__.py
│   ├── adjust_train_parameters.py
│   ├── calculate_priorities.py
│   ├── calculate_scores.py
│   ├── create_model.py
│   ├── dynamic_train.py
│   ├── edgetrain_folder.py
│   ├── resource_monitor.py
│   └── train_visualize.py
│
├── notebooks/
│   └── EdgeTrain_example.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_adjust_train_parameters.py
│   ├── test_calculate_priorities.py
│   ├── test_calculate_scores.py
│   ├── test_create_model_tf.py
│   ├── test_log_usage_once.py
│   └── test_sys_resources.py
│
├── .github/workflows/
│   ├── ci.yml
│   └──lint.yml
│
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── LICENSE
├── README.md
├── requirements-dev.txt
├── requirements.txt
├── setup.py
```

## Contributions
Contributions are welcomed:
- Reporting bugs or requesting features: [GitHub Issues](https://github.com/BradleyEdelman/edgetrain/issues)
- Improve documentation: Help refine explanations and add examples
- Testing: Test EdgeTrain using mode complex models and datasets in heavily resource-constrained environments


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Known Limitations (v0.2.0)
- Currently supports **TensorFlow only**. Future updates will expand framework support.
- **Gradient accumulation**: Planned for a future release to further optimize memory usage
- **Resource usage thresholds** are still in an experimental phase and may require fine-tuning.
