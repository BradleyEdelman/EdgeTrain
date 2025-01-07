# OptiTrain: Automated Resource Adjustment for Efficient Training
**Version: 1.0.0** 

OptiTrain is a Python package designed to dynamically adjust training hyperparameters and system resource utilization based on CPU and GPU performance. It optimizes the training process by adjusting key hyperparameters—batch size, learning rate, and gradient accumulation steps—ensuring efficient training without overutilizing or underutilizing available resources. <br />

## Features
### Automated Resource Adjustment
OptiTrain dynamically adjusts the following hyperparameters based on CPU/GPU usage:
- Batch Size: Automatically adjusts batch size for better memory optimization.
- Learning Rate: Scales learning rate according to available compute resources.
- Gradient Accumulation Steps: Adjusts gradient accumulation steps to balance GPU memory usage and performance.
These adjustments optimize resource utilization throughout training, enabling efficient use of available resources.
<br />

### Resource Logging & Visualization
OptiTrain logs critical system metrics (e.g., CPU and GPU usage) and training parameters (batch size, learning rate, gradient accumulation steps) for each epoch. The logs enable post-hoc visualization and analysis of:
- Resource utilization over time
- Training parameter adjustments across epochs
- Correlations between resource usage and model performance
<br />

### Customizable
OptiTrain is highly customizable. You can easily modify:
- Resource Adjustment Thresholds: Set CPU/GPU usage ranges to trigger adjustments.
- Training Configuration Settings: Adjust batch size increment, learning rate scaling, and gradient accumulation steps.
- Tailor the optimization process to fit your specific needs, whether you're working with a single machine or multi-node setups.<br />

## Installation
You can install OptiTrain via pip:

```bash
# Copy code
pip install optitrain
```

Alternatively, clone the repository and install manually:
```bash
# Clone the repository
git clone https://github.com/BradleyEdelman/optitrain.git
cd optitrain

# Install the package
pip install 
```

## Usage
To use OptiTrain, simply import the package and configure your training environment:

```python
# Import library
import optitrain
```

## File Tree
```bash
OptiTrain/
├── optitrain/
│   ├── __init__.py
│   ├── dynamic_train.py
│   ├── resource_monitor.py
│   ├── resource_adjust.py
│   ├── log_analysis.py
├── tests/
│   ├── 
│   ├── 
├── example_notebooks/
│   ├── ScaleML_tf_ex.ipynb
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
