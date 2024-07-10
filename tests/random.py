"""
Module for generating random quantum circuits
"""

import numpy as np

from basic_qc_simulator import Circuit
from basic_qc_simulator.gates import GATETYPES_TO_GATE


def generate_random_circuit(num_qubits: int, depth: int, seed: int) -> Circuit:
    """
    Generate a random quantum circuit

    Args:
        num_qubits (int): number of qubits in the circuit
        depth (int): depth of the circuit

    Returns:
        Circuit: random quantum circuit
    """
    rng = np.random.default_rng(seed)
    circuit = Circuit(num_qubits)
    for _ in range(depth):
        # one-qubit gates
        for qubit in range(num_qubits):
            gate_type = rng.choice(
                ["i", "x", "y", "z", "h", "s", "t", "phase", "rx", "ry", "rz"]
            )
            if gate_type in ["phase", "rx", "ry", "rz"]:
                circuit.add_gate(
                    GATETYPES_TO_GATE[gate_type](rng.uniform(0, 2 * np.pi)),
                    [qubit],
                )
            else:
                circuit.add_gate(GATETYPES_TO_GATE[gate_type](), [qubit])
        # two-qubit gate
        gate = GATETYPES_TO_GATE[rng.choice(["cx", "ccx", "swap"])]()
        if num_qubits < gate.num_qubits:
            continue
        qubit = rng.integers(num_qubits - gate.num_qubits + 1)
        circuit.add_gate(gate, [qubit + i for i in range(gate.num_qubits)])
    return circuit
