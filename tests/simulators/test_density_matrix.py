"""
Test the density matrix simulator.
"""

import itertools

import numpy as np
import pytest
import qiskit_aer.noise
from qiskit_aer import AerSimulator

from basic_qc_simulator.noise import CustomNoiseModel, isotropic_depolarizing_channel
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


def test_density_matrix_simulator_with_noise():
    """
    Test the state vector simulator with noise
    """

    num_qubits = 4
    depth = 6
    seed = 0

    circuit = generate_random_circuit(num_qubits, depth, seed)
    circuit.save_result("density_matrix")

    # little endian -> big endian
    qiskit_circuit = circuit.to_qiskit().reverse_bits()
    qiskit_circuit.save_density_matrix()  # pylint: disable=no-member

    noise_model = CustomNoiseModel()
    noise_model.add_noise_to_gate(
        "h", isotropic_depolarizing_channel(0.001, 1, definition="qiskit")
    )
    noise_model.add_noise_to_gate(
        "cx", isotropic_depolarizing_channel(0.01, 2, definition="qiskit")
    )

    simulator = DensityMatrixSimulator(noise_model=noise_model)
    simulator.run(circuit)
    actual_density_matrix = simulator.results[0].result

    qiskit_noise_model = qiskit_aer.noise.NoiseModel()
    qiskit_noise_model.add_all_qubit_quantum_error(
        qiskit_aer.noise.depolarizing_error(0.001, 1), "h"
    )
    qiskit_noise_model.add_all_qubit_quantum_error(
        qiskit_aer.noise.depolarizing_error(0.01, 2), "cx"
    )

    qiskit_simulator = AerSimulator(
        method="density_matrix", noise_model=qiskit_noise_model
    )
    qiskit_result = qiskit_simulator.run(qiskit_circuit).result()
    expected_density_matrix = qiskit_result.data(0)["density_matrix"]
    np.testing.assert_array_almost_equal(actual_density_matrix, expected_density_matrix)
