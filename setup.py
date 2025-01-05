from setuptools import setup, find_packages

setup(
    name="scaleml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "tensorflow>=2.0",
        "numpy",
        "scipy",
        "gputil" #gpu detection
        "psutil" # cpu detection
        "distributed",  # For distributed systems
        "dill",  # For serialization of objects
    ],
    description="A utility for distributed machine learning training strategies.",
    author="Bradley Edelman",
    author_email="bjedelma@gmail.com",
    url="https://github.com/BradleyEdelman/scaleml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
