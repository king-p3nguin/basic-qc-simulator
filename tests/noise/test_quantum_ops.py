"""
Test the quantum operations module
"""

import itertools

import numpy as np
import pytest
import qiskit_aer.noise
from qiskit.quantum_info.random import random_density_matrix

from basic_qc_simulator.noise import noise_channel


def test_composed_error_channel():
    """
    Test the composed error channel
    """
    error_channel_1 = noise_channel.isotropic_depolarizing_channel(
        0.1, 1, definition="qiskit"
    )
    error_channel_2 = noise_channel.isotropic_depolarizing_channel(
        0.2, 1, definition="qiskit"
    )
    composed_error_channel = error_channel_1.compose(error_channel_2)

    actual_superoperator = composed_error_channel.to_superoperator()

    qiskit_error_channel_1 = qiskit_aer.noise.depolarizing_error(0.1, 1)
    qiskit_error_channel_2 = qiskit_aer.noise.depolarizing_error(0.2, 1)
    qiskit_composed_error_channel = qiskit_error_channel_1.compose(
        qiskit_error_channel_2
    )

    expected_superoperator = qiskit_composed_error_channel.to_quantumchannel()

    np.testing.assert_array_almost_equal(actual_superoperator, expected_superoperator)


def test_tensor_product_error_channel():
    """
    Test the tensor product error channel
    """
    error_channel_1 = noise_channel.isotropic_depolarizing_channel(
        0.1, 1, definition="qiskit"
    )
    error_channel_2 = noise_channel.isotropic_depolarizing_channel(
        0.2, 1, definition="qiskit"
    )
    tensor_product_error_channel = error_channel_1.tensordot(error_channel_2)

    actual_superoperator = tensor_product_error_channel.to_superoperator()

    qiskit_error_channel_1 = qiskit_aer.noise.depolarizing_error(0.1, 1)
    qiskit_error_channel_2 = qiskit_aer.noise.depolarizing_error(0.2, 1)
    qiskit_tensor_product_error_channel = qiskit_error_channel_1.tensor(
        qiskit_error_channel_2
    )

    expected_superoperator = qiskit_tensor_product_error_channel.to_quantumchannel()

    np.testing.assert_array_almost_equal(actual_superoperator, expected_superoperator)


@pytest.mark.parametrize(
    "p, qubits, seed",
    list(
        itertools.product(
            [0.1, 0.2, 0.3], [[0], [1], [0, 1], [0, 2], [0, 1, 2]], [0, 1]
        )
    ),
)
def test_applying_operators_to_density_matrix(p, qubits, seed):
    """
    Test applying operators to a density matrix
    """
    num_qubits = len(qubits)

    # Create a random 3-qubit density matrix
    density_matrix = random_density_matrix(2**3, seed=seed).data

    kraus_operators = noise_channel.isotropic_depolarizing_channel(p, num_qubits)
    actual_density_matrix_1 = kraus_operators.apply(density_matrix, qubits)

    actual_density_matrix_2 = kraus_operators.to_superoperator().apply(
        density_matrix, qubits
    )

    np.testing.assert_array_almost_equal(
        actual_density_matrix_1, actual_density_matrix_2
    )
