"""
Module for state vector representation of a quantum state.
"""

import logging
from copy import copy

import numpy as np

from ...gates import Gate

logger = logging.getLogger(__name__)


class StateVector:
    """
    Class for the state vector representation of a quantum state.
    """

    def __init__(self, state_vector: list | np.ndarray) -> None:
        """
        Args:
            state_vector (np.ndarray): state vector representation of a quantum state
        """
        if not isinstance(state_vector, np.ndarray):
            state_vector = np.array(state_vector)
        if state_vector.ndim == 1:  # flat state vector
            self.num_qubits = int(np.log2(state_vector.shape[0]))
            # Reshape the flattened state vector to (2, 2, ..., 2) tensor
            self.state_vector = state_vector.reshape((2,) * self.num_qubits)
        else:  # folded state vector
            if state_vector.shape == (2,) * state_vector.ndim:
                ValueError("Shape of the folded state vector is not (2,) * num_qubits")
            self.num_qubits = state_vector.ndim
            self.state_vector = state_vector

    def __repr__(self) -> str:
        return f"StateVector(state_vector={self.state_vector.flatten()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateVector):
            return False
        return np.array_equal(self.state_vector, other.state_vector)

    def __array__(self) -> np.ndarray:
        return self.state_vector.flatten()

    def apply_gate(
        self, gate: Gate, qargs: int | list[int], inplace: bool = False
    ) -> "StateVector":
        if isinstance(qargs, int):
            qargs = [qargs]

        #           sv_tensor_indices                    new_sv_tensor_indices
        #        ┌──┐
        #   q_0: ┤ H ├─  0 ───────────────── 0
        #        │   │                 ┌──┐
        #   q_1: ┤   ├─  1 ─ qb[0] ─┤   ├─  n  ───── n
        #        │   │                 │   │
        #   q_2: ┤   ├─  2 ─ qb[1] ─┤   ├─ n+1 ───── n+1
        #        │   │                 │   │
        #   q_3: ┤   ├─  3 ─ qb[2] ─┤   ├─ n+2 ───── n+2
        #    :   │   │                └──┘
        #    :   │   │           gate_tensor_indices
        #    :   │   │
        # q_n-1: ┤   ├─ n-1─────────────────  n-1
        #        └──┘

        gate_matrix = gate.matrix.reshape((2,) * gate.num_qubits * 2)
        gate_num_qubits = gate.num_qubits
        new_state_vector = self.state_vector if inplace else copy(self.state_vector)

        # state_vector_tensor_indices = [0, 1, 2, ..., n-1]
        state_vector_tensor_indices = list(range(self.num_qubits))
        # gate_tensor_indices
        #   = [n, n+1, n+2, ..., n+m-1, qubits[0], qubits[1], ..., qubits[m-1]]
        gate_tensor_indices = list(
            range(self.num_qubits, self.num_qubits + gate_num_qubits)
        ) + list(qargs)
        # new_state_vector_tensor_indices = [0, 1, 2, ..., n-1]
        # with qubits[i] replaced by gate_tensor_indices[i]
        new_state_vector_tensor_indices = copy(state_vector_tensor_indices)
        for i in range(gate_num_qubits):
            new_state_vector_tensor_indices[qargs[i]] = gate_tensor_indices[i]

        logger.debug(
            msg=f"Applying gate '{gate.name}' "
            f"to qubits {qargs}\n"
            f"input indices: {state_vector_tensor_indices}\n"
            f"gate indices: {gate_tensor_indices}\n"
            f"output indices: {new_state_vector_tensor_indices}\n"
        )
        # Apply the gate by contracting the state vector tensor with the gate tensor
        new_state_vector = np.einsum(
            new_state_vector,
            state_vector_tensor_indices,
            gate_matrix,
            gate_tensor_indices,
            new_state_vector_tensor_indices,
        )
        return StateVector(new_state_vector)
