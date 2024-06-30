# basic-qc-simulator

Basic implementation of quantum circuit simulator in Python for easy understanding

## Setup

Install via pip:

```shell
pip install git+https://github.com/king-p3nguin/basic-qc-simulator.git#egg=basic-qc-simulator
```

Install via pip with qiskit libraries:

```shell
pip install git+https://github.com/king-p3nguin/basic-qc-simulator.git#egg=basic-qc-simulator[qiskit]
```

## Basic Usage

```python
from basic_qc_simulator import Circuit
from basic_qc_simulator.simulators import StateVectorSimulator

# Create a quantum circuit
c = Circuit(2)
c.h(0)
c.cx(0, 1)
c.save_result("state_vector")

# Run the state vector simulator
sim = StateVectorSimulator()
sim.run(c)

# Get the result
sim.results[0].result
```

Output:

```text
[0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
```

Please refer to `examples/` directory for more advanced usage.
