from setuptools import setup, find_packages

setup(
    name="edgetrain",
    version="0.1.1-alpha",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.0.0",
        "psutil>=5.0.0",
        "GPUtil>=1.4.0",
        "matplotlib>=3.7.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "pynvml>=8.0.0",
	"torch>=2.5.1",
    ],
    extras_require={
	'dev': [
	     'pytest', # for testing
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
