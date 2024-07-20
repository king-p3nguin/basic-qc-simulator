"""
Test for the MatrixProductState class.
"""

import itertools

import numpy as np
import pytest

from basic_qc_simulator.quantum_info.states import MatrixProductState, StateVector


@pytest.mark.parametrize(
    "seed, num_qubits", list(itertools.product(range(3), range(2, 5)))
)
def test_matrix_product_state_state_vector_conversion(seed, num_qubits):
    """
    Test the conversion between matrix product state and state vector
    """
    rng = np.random.default_rng(seed)
    random_state_vector = rng.random(2**num_qubits) + 1j * rng.random(2**num_qubits)
    random_state_vector /= np.linalg.norm(random_state_vector)
    random_state_vector = StateVector(random_state_vector)

    mps = MatrixProductState.from_state_vector(random_state_vector)
    state_vector = mps.to_state_vector()

    np.testing.assert_array_almost_equal(random_state_vector, state_vector)
