# Contribution guideline

Please feel free to open a PR for improvements.

## Install packages required for testing

```shell
pip install -e .[qiskit,test]
```

## Debug logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("basic_qc_simulator")
```
