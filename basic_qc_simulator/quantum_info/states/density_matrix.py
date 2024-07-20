"""
Module for density matrix representation of quantum states.
"""

import logging
from copy import copy

import numpy as np

from ...gates import Gate

logger = logging.getLogger(__name__)


class DensityMatrix:
    """
    Class for the density matrix representation of a quantum state.
    """

    def __init__(self, density_matrix: list | np.ndarray) -> None:
        """
        Args:
            density_matrix (list | np.ndarray): density matrix representation of a quantum state
        """
        if not isinstance(density_matrix, np.ndarray):
            density_matrix = np.array(density_matrix)
        if density_matrix.ndim == 2:  # matrix representation
            self.num_qubits = int(np.log2(density_matrix.shape[0]))
            # Reshape to (2, 2, ..., 2) tensor
            self.density_matrix = density_matrix.reshape((2,) * self.num_qubits * 2)
        else:  # folded tensor representation
            if not (
                density_matrix.shape == (2,) * density_matrix.ndim
                and density_matrix.ndim % 2 == 0
            ):
                raise ValueError(
                    "Shape of the density matrix is not (2,) * num_qubits * 2"
                )
            self.num_qubits = density_matrix.ndim // 2
            self.density_matrix = density_matrix

    def __repr__(self) -> str:
        return "DensityMatrix({})".format(
            self.density_matrix.reshape(2**self.num_qubits, 2**self.num_qubits)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DensityMatrix):
            return False
        return np.array_equal(self.density_matrix, other.density_matrix)

    def __array__(self) -> np.ndarray:
        return self.density_matrix.reshape(2**self.num_qubits, 2**self.num_qubits)

    def __add__(self, other: "DensityMatrix") -> "DensityMatrix":
        return DensityMatrix(self.density_matrix + other.density_matrix)

    def apply_gate(
        self, gate: Gate, qargs: int | list[int], inplace: bool = False
    ) -> "DensityMatrix":
        """
        Apply a gate to the density matrix.

        Args:
            gate (Gate): gate to apply
            qargs (int | list[int]): qubits to apply the gate to
            inplace (bool): apply the gate in place (default: False)

        Returns:
            DensityMatrix: resulting density matrix
        """
        if isinstance(qargs, int):
            qargs = [qargs]

        gate_num_qubits = gate.num_qubits
        gate_matrix = gate.matrix.reshape((2,) * gate_num_qubits * 2)
        gate_matrix_conj = np.conj(gate.matrix).T.reshape((2,) * gate_num_qubits * 2)
        new_density_matrix = (
            self.density_matrix if inplace else copy(self.density_matrix)
        )

        # density_matrix_tensor_indices = [0, 1, 2, ..., n-1, n, n+1, ..., 2n-1]
        density_matrix_tensor_indices = list(range(self.num_qubits * 2))

        # gate_tensor_indices
        #   = [2n, 2n+1, 2n+2, ..., 2n+m-1, qubits[0], qubits[1], ..., qubits[m-1]]
        gate_tensor_indices = list(
            range(self.num_qubits * 2, self.num_qubits * 2 + gate_num_qubits)
        ) + list(qargs)

        # new_density_matrix_tensor_indices = [0, 1, 2, ..., n-1, n, n+1, ..., 2n-1]
        #   with qubits[i] replaced by gate_tensor_indices[i]
        new_density_matrix_tensor_indices = copy(density_matrix_tensor_indices)
        for i in range(gate_num_qubits):
            new_density_matrix_tensor_indices[qargs[i]] = gate_tensor_indices[i]

        logger.debug(
            msg=f"Applying gate '{gate.name}' to qubits {qargs}\n"
            f"input indices: {density_matrix_tensor_indices}\n"
            f"gate indices: {gate_tensor_indices}\n"
            f"output indices: {new_density_matrix_tensor_indices}\n"
        )
        # Apply the gate by contracting the density matrix tensor with the gate tensor
        new_density_matrix = np.einsum(
            new_density_matrix,
            density_matrix_tensor_indices,
            gate_matrix,
            gate_tensor_indices,
            new_density_matrix_tensor_indices,
        )

        # gate_tensor_conj_indices
        #   = [n+qubits[0], n+qubits[1], ..., n+qubits[m-1], 2n, 2n+1, 2n+2, ..., 2n+m-1]
        gate_tensor_conj_indices = [self.num_qubits + q for q in qargs] + list(
            range(self.num_qubits * 2, self.num_qubits * 2 + gate_num_qubits)
        )

        # new_density_matrix_tensor_indices = [0, 1, 2, ..., n-1, n, n+1, ..., 2n-1]
        #   with n+qubits[i] replaced by gate_tensor_conj_indices[i]
        new_density_matrix_tensor_indices = copy(density_matrix_tensor_indices)
        for i in range(gate_num_qubits):
            new_density_matrix_tensor_indices[self.num_qubits + qargs[i]] = (
                gate_tensor_conj_indices[gate_num_qubits + i]
            )

        logger.debug(
            msg=f"Applying gate '{gate.name}' conjugate "
            f"to qubits {qargs}\n"
            f"input indices: {density_matrix_tensor_indices}\n"
            f"gate indices: {gate_tensor_conj_indices}\n"
            f"output indices: {new_density_matrix_tensor_indices}\n"
        )
        # Apply the conjugate gate by contracting the density matrix tensor
        # with the conjugate gate tensor
        new_density_matrix = np.einsum(
            new_density_matrix,
            density_matrix_tensor_indices,
            gate_matrix_conj,
            gate_tensor_conj_indices,
            new_density_matrix_tensor_indices,
        )
        return DensityMatrix(new_density_matrix)
