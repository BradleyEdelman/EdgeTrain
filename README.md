# ScaleML
**Version: 1.0.0** 

ScaleML is a utility package designed to optimize distributed machine learning (ML) training strategies by dynamically adjusting system resources (such as CPU cores and GPUs) during training. It automates the scaling of resources based on real-time usage to maximize training efficiency and minimize resource waste. <br />

## Features
Automated Resource Adjustment: Dynamically adjusts the number of workers and batch size based on CPU/GPU usage, optimizing resource utilization throughout training. <br />
Resource Logging & Visualization: Logs CPU/GPU usage and training parameters, enabling real-time visualization of resource utilization and training performance. <br />
Scalability: Automatically scales training to available resources, adjusting training strategies as required. <br />
Customizable: Easily modify resource adjustment thresholds and training configurations to meet your specific needs. <br />

## Installation
You can install ScaleML via pip:

```bash
# Copy code
pip install scaleml
```

Alternatively, clone the repository and install manually:
```bash
# Clone the repository
git clone https://github.com/BradleyEdelman/scaleml.git
cd scaleml

# Install the package
pip install
```

## Usage
To use ScaleML, simply import the package and configure your training environment:

```python
# Import library
import scaleml
```

## File Tree
```bash
ScaleML/
├── scaleml/
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
