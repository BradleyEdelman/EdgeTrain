from setuptools import find_packages, setup

setup(
    name="edgetrain",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "psutil>=5.0.0",
        "GPUtil>=1.4.0",
        "matplotlib>=3.7.0",
        "pandas>=1.5.0",
        "pynvml>=8.0.0",
        "tensorflow==2.12.0",
        "keras==2.12.0",
        "tensorflow-model-optimization==0.7.3",
        "jupyter",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "pre-commit",
            "black",
            "black[jupyter]",
            "flake8",
            "isort",
        ]
    },
    description="A utility for machine learning training with limited resources.",
    author="Bradley Edelman",
    author_email="bjedelma@gmail.com",
    url="https://github.com/BradleyEdelman/edgetrain",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
