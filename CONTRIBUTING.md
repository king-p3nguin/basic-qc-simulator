# Contribution guideline

Please feel free to open a PR for improvements.

## Install packages required for testing

```shell
pip install -e .[qiskit,test]
```

## Debug logging

```python
import logging

logger = logging.getLogger("basic_qc_simulator")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())
```
