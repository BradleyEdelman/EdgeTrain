# Changelog
### [0.1.0-alpha] - 2025.10.01 Initial release
### [0.1.1-alpha] - 2025.10.01 Circular import bug fix
### [0.2.0] - 2025.11.02
#### Added
- Dynamic adjustment of **batch size** and **learning rate** based on resource usage.
- Priority-based parameter tuning to balance memory constraints and accuracy.
- Logging and visualization of training adjustments and system resource usage.

#### Changed
- Pruning strategy now maintains a **constant ratio** while adjusting batch size and learning rate dynamically.
- Improved accuracy and memory scoring to better balance learning rate and batch sizeadjustments.

#### Fixed
- Pre-commit hooks now properly enforce **Black, isort, and Flake8** styling.
- Addressed circular imports affecting test execution.

#### Removed
- Old manual tuning logic replaced with automated **adaptive training strategies**.
