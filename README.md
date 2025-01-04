# ScaleML

ScaleML is a utility package designed to simplify and automate the setup of distributed machine learning (ML) training strategies. It aims to make it easier to scale ML workflows across multiple nodes and environments, providing seamless integration with popular frameworks like TensorFlow, PyTorch, etc.

Features
Automated Setup: Quickly configure distributed training strategies for single- and multi-node setups with CPU cores and/or GPUs.
Easy Integration: Compatible with TensorFlow, PyTorch, and other major ML frameworks.
Scalability: Automatically adjusts to the available infrastructure to maximize resource usage.
Customizable: Easily modify training configurations to meet your specific needs.
Installation
You can install ScaleML via pip:

bash
Copy code
pip install scaleml
Alternatively, clone the repository and install manually:

bash
Copy code
git clone https://github.com/BradleyEdelman/scaleml.git
cd scaleml
pip install .
Usage
To use ScaleML, simply import the package and configure your training environment:

python
Copy code
import scaleml

Report bugs or request features
Improve the documentation
Add new training strategies or features
License
This project is licensed under the MIT License - see the LICENSE file for details.