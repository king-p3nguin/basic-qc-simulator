"""
Test the matrix product state simulator.
"""

import itertools

import numpy as np
import pytest
from qiskit_aer import AerSimulator

from basic_qc_simulator.simulators.matrix_product_state import (
    MatrixProductStateSimulator,
)

from ..random import generate_random_circuit


@pytest.skip
@pytest.mark.parametrize(
    "num_qubits, depth, seed",
    list(itertools.product(range(2, 5), range(1, 4), range(3))),
)
def test_matrix_product_state_simulator(num_qubits, depth, seed):
    """
    Test the matrix product state simulator
    """

    circuit = generate_random_circuit(num_qubits, depth, seed)
    circuit.save_result("matrix_product_state")

    # little endian -> big endian
    qiskit_circuit = circuit.to_qiskit().reverse_bits()
    qiskit_circuit.save_matrix_product_state()  # pylint: disable=no-member

    simulator = MatrixProductStateSimulator()
    simulator.run(circuit)
    actual_matrix_product_state = simulator.results[0].result
    qiskit_simulator = AerSimulator(method="matrix_product_state")
    qiskit_result = qiskit_simulator.run(qiskit_circuit).result()
    expected_matrix_product_state = qiskit_result.data(0)["matrix_product_state"]
    np.testing.assert_array_almost_equal(
        actual_matrix_product_state, expected_matrix_product_state
    )
