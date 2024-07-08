"""
Test the state vector simulator.
"""

import itertools

import numpy as np
import pytest
from qiskit_aer import AerSimulator

from basic_qc_simulator.simulators.state_vector import StateVectorSimulator

from ..random import generate_random_circuit


@pytest.mark.parametrize(
    "num_qubits, depth, seed",
    list(itertools.product(range(2, 5), range(1, 4), range(3))),
)
def test_state_vector_simulator(num_qubits, depth, seed):
    """
    Test the state vector simulator
    """

    circuit = generate_random_circuit(num_qubits, depth, seed)
    circuit.save_result("state_vector")

    # little endian -> big endian
    qiskit_circuit = circuit.to_qiskit().reverse_bits()
    qiskit_circuit.save_statevector()  # pylint: disable=no-member

    simulator = StateVectorSimulator()
    simulator.run(circuit)
    actual_state_vector = simulator.results[0].result
    qiskit_simulator = AerSimulator(method="statevector")
    qiskit_result = qiskit_simulator.run(qiskit_circuit).result()
    expected_state_vector = qiskit_result.get_statevector()
    np.testing.assert_array_almost_equal(actual_state_vector, expected_state_vector)
