"""
Module for noise channel
"""

import itertools

import numpy as np

from ..gates import IGate, XGate, YGate, ZGate
from .quantum_ops import KrausOperators


def isotropic_depolarizing_channel(
    param: float, num_qubits: int, definition: str = "original"
) -> KrausOperators:
    r"""Create the kraus operators for a isotropic depolarizing channel

    original definition:

    .. math::

            E(\rho) = (1 - p) \rho + \frac{p}{4^n-1} \sum_{i=1}^{4^n} P_i \rho P_i^\dagger

    qiskit's definition:

    .. math::

            E(\rho) = (1 - \frac{4^n-1}{4^n}\lambda) \rho
            + \frac{\lambda}{4^n} \sum_{i=1}^{4^n} P_i \rho P_i^\dagger

    where
    - :math:`P_i`: tensor product of Pauli matrices on each qubit in the i-th basis state
        except for the identity matrix for all :math:`i`
    - :math:`n`: number of qubits

    Args:
        p (float): probability of error
        num_qubits (int): number of qubits

    Returns:
        KrausOperators: kraus operators for the depolarizing channel
    """
    if definition == "original":
        if not 0 <= param <= 1:
            raise ValueError("p must be between 0 and 1")
        converted_param = param
    elif definition == "qiskit":
        if not 0 <= param <= 4**num_qubits / (4**num_qubits - 1):
            raise ValueError("lambda must be between 0 and 1")
        converted_param = (4**num_qubits - 1) / 4**num_qubits * param
    else:
        raise ValueError("definition must be 'original' or 'qiskit'")
    if num_qubits < 1:
        raise ValueError("num_qubits must be greater than 0")
    pauli_matrices = [IGate().matrix, XGate().matrix, YGate().matrix, ZGate().matrix]
    error_gates = list(itertools.product(pauli_matrices, repeat=num_qubits))
    probabilities = [1 - converted_param] + [
        converted_param / (4**num_qubits - 1)
    ] * (4**num_qubits - 1)

    kraus_operators = []
    for prob, gates in zip(probabilities, error_gates):
        full_gate = gates[0]
        for gate in gates[1:]:
            full_gate = np.kron(full_gate, gate)
        kraus_operators.append(np.sqrt(prob) * full_gate)

    return KrausOperators(
        kraus_operators, f"Isotropic depolarizing channel with p={converted_param}"
    )


def amplitude_damping_channel(
    param: float, equilibrium_excited_state_population: float = 0.0
) -> KrausOperators:
    r"""Create the kraus operators for an single-qubit amplitude damping channel

    Definition:

    .. math::

        \sqrt{1 - p} \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix},
        \sqrt{1 - p} \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix},
        \sqrt{p} \begin{pmatrix} \sqrt{1 - \gamma} & 0 \\ 0 & 1 \end{pmatrix},
        \sqrt{p} \begin{pmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{pmatrix}

    where :math:`p = \text{equilibrium_excited_state_population}`.

    Args:
        param (float): the amplitude damping error parameter.
        equilibrium_excited_state_population (float): the population of :math:`|1\rangle`
            state at equilibrium. Default is 0.0.

    Returns:
        KrausOperators: the Kraus operators of the combined phase and amplitude damping channel.
    """

    if not 0 <= param <= 1:
        raise ValueError("Amplitude damping parameter must be between 0 and 1")
    if not 0 <= equilibrium_excited_state_population <= 1:
        raise ValueError("Equilibrium excited state population must be between 0 and 1")
    p0 = np.sqrt(1 - equilibrium_excited_state_population)
    p1 = np.sqrt(equilibrium_excited_state_population)
    k1 = p0 * np.array([[1, 0], [0, np.sqrt(1 - param)]])
    k2 = p0 * np.array([[0, np.sqrt(param)], [0, 0]])
    k3 = p1 * np.array([[np.sqrt(1 - param), 0], [0, 1]])
    k4 = p1 * np.array([[0, 0], [np.sqrt(param), 0]])
    kraus_ops = [
        kraus
        for kraus in [k1, k2, k3, k4]
        if not np.isclose(np.linalg.norm(kraus), 0.0)
    ]
    return KrausOperators(
        kraus_ops, name=f"Amplitude damping channel with gamma={param}"
    )


def phase_damping_channel(param: float) -> KrausOperators:
    r"""Create the kraus operators for a single-qubit phase damping channel

    Definition:

    .. math::

        \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix},
        \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\gamma} \end{pmatrix}

    Args:
        param (float): the phase damping error parameter.

    Returns:
        KrausOperators: the Kraus operators of the combined phase and amplitude damping channel.
    """

    if not 0 <= param <= 1:
        raise ValueError("Phase damping parameter must be between 0 and 1")
    k1 = np.array([[1, 0], [0, np.sqrt(1 - param)]])
    k2 = np.array([[0, 0], [0, np.sqrt(param)]])
    return KrausOperators([k1, k2], name=f"Phase damping channel with gamma={param}")
