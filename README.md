# ScaleML
**Version: 1.0.0** 

ScaleML is a utility package designed to simplify and automate the setup of distributed machine learning (ML) training strategies. It aims to make it easier to scale ML workflows across multiple nodes and environments, providing seamless integration with popular frameworks like TensorFlow, PyTorch, etc.

## Features
<u>Automated Setup:</u> Quickly configure distributed training strategies for single- and multi-node setups with CPU cores and/or GPUs. <br />
<u>Easy Integration:</u> Compatible with TensorFlow, PyTorch, and other major ML frameworks. <br />
<u>Resource Logging:</u> Log and visualize CPU/GPU usage during training to monitor system performance and optimize resource utilization. <br />
<u>Scalability:</u> Automatically adjusts to the available infrastructure to maximize resource usage. <br />
<u>Customizable:</u> Easily modify training configurations to meet your specific needs. <br />

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
│   ├── resources.py
│   ├── strategies.py
│   ├── distributed_train.py
│   ├── log_resource_usage.py
│   ├── plot_resource_usage.py
├── tests/
│   ├── test_resources.py
│   ├── test_strategies.py
├── example_notebooks/
│   ├── single_node_tensorflow.ipynb
│   ├── single_node_pytorch.ipynb
│   ├── single_node_horovod.ipynb
├── setup.py
├── README.md
├── LICENSE
├── .gitignore
```

## Contributions
Report bugs or request features <br />
Improve the documentation <br />
Add new training strategies or features <br />

## License
This project is licensed under the MIT License - see the LICENSE file for details.
