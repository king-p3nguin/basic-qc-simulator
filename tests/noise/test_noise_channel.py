"""
Test the noise channel module.
"""

import itertools

import numpy as np
import pytest
import qiskit_aer.noise

from basic_qc_simulator.gates import IGate, XGate, YGate, ZGate
from basic_qc_simulator.noise import noise_channel


@pytest.mark.parametrize(
    "p, num_qubits",
    list(itertools.product([0.1, 0.2, 0.3], [1, 2, 3])),
)
def test_isotropic_depolarizing_channel_qiskit_definition(p, num_qubits):
    """
    Test the isotropic depolarizing channel with qiskit's definition
    """
    kraus_operators = noise_channel.isotropic_depolarizing_channel(
        p, num_qubits, definition="qiskit"
    )
    actual_superoperator = kraus_operators.to_superoperator()

    expected_superoperator = qiskit_aer.noise.depolarizing_error(
        p, num_qubits
    ).to_quantumchannel()

    np.testing.assert_array_almost_equal(actual_superoperator, expected_superoperator)


def test_isotropic_depolarizing_channel_original_definition():
    """
    Test the isotropic depolarizing channel with an original definition
    """
    actual_kraus_operators = noise_channel.isotropic_depolarizing_channel(
        0.1, 1, definition="original"
    )
    expected_kraus_operators = [
        np.sqrt(0.9) * IGate().matrix,
        np.sqrt(0.1 / 3) * XGate().matrix,
        np.sqrt(0.1 / 3) * YGate().matrix,
        np.sqrt(0.1 / 3) * ZGate().matrix,
    ]
    np.testing.assert_array_almost_equal(
        actual_kraus_operators, expected_kraus_operators
    )


def test_isotropic_depolarizing_channel_invalid_definition():
    """
    Test the isotropic depolarizing channel with an invalid definition
    """
    with pytest.raises(ValueError):
        noise_channel.isotropic_depolarizing_channel(0.1, 1, definition="invalid")


def test_amplitude_damping_channel():
    """
    Test the amplitude damping channel
    """
    kraus_operators = noise_channel.amplitude_damping_channel(0.1)
    actual_superoperator = kraus_operators.to_superoperator()

    expected_superoperator = qiskit_aer.noise.amplitude_damping_error(
        0.1
    ).to_quantumchannel()

    np.testing.assert_array_almost_equal(actual_superoperator, expected_superoperator)


def test_phase_damping_channel():
    """
    Test the phase damping channel
    """
    kraus_operators = noise_channel.phase_damping_channel(0.1)
    actual_superoperator = kraus_operators.to_superoperator()

    expected_superoperator = qiskit_aer.noise.phase_damping_error(
        0.1
    ).to_quantumchannel()

    np.testing.assert_array_almost_equal(actual_superoperator, expected_superoperator)
