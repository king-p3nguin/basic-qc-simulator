"""
Test the density matrix simulator.
"""

import itertools

import numpy as np
import pytest
from qiskit_aer import AerSimulator

from basic_qc_simulator.simulators.density_matrix import DensityMatrixSimulator

from ..random import generate_random_circuit


@pytest.mark.parametrize(
    "num_qubits, depth, seed",
    list(itertools.product(range(2, 5), range(1, 4), range(3))),
)
def test_density_matrix_simulator(num_qubits, depth, seed):
    """
    Test the state vector simulator
    """

    circuit = generate_random_circuit(num_qubits, depth, seed)
    circuit.save_result("density_matrix")

    # little endian -> big endian
    qiskit_circuit = circuit.to_qiskit().reverse_bits()
    qiskit_circuit.save_density_matrix()  # pylint: disable=no-member

    simulator = DensityMatrixSimulator()
    simulator.run(circuit)
    actual_density_matrix = simulator.results[0].result
    qiskit_simulator = AerSimulator(method="density_matrix")
    qiskit_result = qiskit_simulator.run(qiskit_circuit).result()
    expected_density_matrix = qiskit_result.data(0)["density_matrix"]
    np.testing.assert_array_almost_equal(actual_density_matrix, expected_density_matrix)
