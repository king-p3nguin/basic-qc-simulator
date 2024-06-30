"""
Test the state vector simulator.
"""

import numpy as np
import pytest
from qiskit_aer import AerSimulator

from basic_qc_simulator.simulators.state_vector import StateVectorSimulator

from ..random import generate_random_circuit


@pytest.mark.parametrize(
    "num_qubits, depth, seed",
    [
        (2, 1, 0),
        (2, 1, 1),
        (2, 1, 2),
        (3, 2, 0),
        (3, 2, 1),
        (3, 2, 2),
        (4, 3, 0),
        (4, 3, 1),
        (4, 3, 2),
    ],
)
def test_state_vector_simulator(num_qubits, depth, seed):
    """
    Test the state vector simulator
    """

    circuit = generate_random_circuit(num_qubits, depth, seed)
    circuit.save_result("state_vector")

    # little endian -> big endian
    qiskit_circuit = circuit.to_qiskit().reverse_bits()

    simulator = StateVectorSimulator()
    simulator.run(circuit)
    actual_state_vector = simulator.results[0].result
    qiskit_simulator = AerSimulator(method="statevector")
    qiskit_result = qiskit_simulator.run(qiskit_circuit).result()
    expected_state_vector = qiskit_result.get_statevector()
    np.testing.assert_array_almost_equal(actual_state_vector, expected_state_vector)
