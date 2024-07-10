"""
Module for quantum operations
"""

import logging
from copy import copy

import numpy as np

from ..circuit import Instruction
from ..gates import Gate, GateTypes
from ..simulators.density_matrix import DensityMatrixSimulator

logger = logging.getLogger(__name__)


class KrausOperators(list):
    """
    Class for kraus operators
    """

    def __init__(self, kraus_operators: list[np.ndarray], name: str) -> None:
        super().__init__(kraus_operators)
        self._name = name

    def __repr__(self) -> str:
        return f"KrausOperators({list(self)},\nname={self._name})"

    @property
    def name(self) -> str:
        """Return the name of the kraus operators

        Returns:
            str: name of the kraus operators
        """
        return self._name

    def apply(self, density_matrix: np.ndarray, qubits: int | list[int]) -> np.ndarray:
        """
        Apply the kraus operators to the density matrix

        Args:
            density_matrix (np.ndarray): density matrix to apply the kraus operators to
            quibits (int | list[int]): qubits to apply the kraus operators to

        Returns:
            np.ndarray: resulting density matrix
        """
        if isinstance(qubits, int):
            qubits = [qubits]
        if density_matrix.ndim != 2:
            raise ValueError(
                f"Density matrix must be 2D, got shape {density_matrix.shape}"
            )
        num_qubits = int(np.log2(density_matrix.shape[0]))
        kraus_num_qubits = int(np.log2(self[0].shape[0]))
        reshaped_density_matrix = np.reshape(density_matrix, (2,) * num_qubits * 2)
        new_density_matrix = np.zeros_like(reshaped_density_matrix)
        for kraus_operator in self:
            kraus_gate = Gate(GateTypes.KRAUS, kraus_operator, kraus_num_qubits)
            instruction = Instruction(kraus_gate, qubits)
            new_density_matrix += DensityMatrixSimulator._apply_gate(
                instruction, reshaped_density_matrix
            )

        return new_density_matrix.reshape((2**num_qubits, 2**num_qubits))

    def to_superoperator(self) -> "Superoperator":
        """
        Convert the kraus operators to a superoperator

        Returns:
            Superoperator: superoperator
        """
        superop = np.sum(
            [np.kron(np.conj(krausop), krausop) for krausop in self], axis=0
        )
        return Superoperator(superop, self.name)

    def compose(self, other: "KrausOperators") -> "KrausOperators":
        """
        Compose the kraus operators with another set of kraus operators

        Args:
            other (KrausOperators): kraus operators to compose with

        Returns:
            KrausOperators: composed kraus operators
        """
        composed_kraus = [
            np.dot(krausop1, krausop2) for krausop1 in self for krausop2 in other
        ]
        return KrausOperators(composed_kraus, f"{self.name} composed with {other.name}")

    def tensordot(self, other: "KrausOperators") -> "KrausOperators":
        """
        Apply tensor product to the kraus operators with another set of kraus operators

        Args:
            other (KrausOperators): kraus operators to apply tensor product with

        Returns:
            KrausOperators: tensor product of kraus operators
        """
        tensor_kraus = [
            np.kron(krausop1, krausop2) for krausop1 in self for krausop2 in other
        ]
        return KrausOperators(
            tensor_kraus, f"{self.name} tensor product with {other.name}"
        )


class Superoperator:
    """
    Class for a superoperator
    """

    def __init__(self, superoperator: np.ndarray, name: str) -> None:
        self._superoperator = superoperator
        self._name = name

    def __array__(self) -> np.ndarray:
        return self._superoperator

    def __repr__(self) -> str:
        return f"Superoperator({self._superoperator},\nname={self._name})"

    @property
    def name(self) -> str:
        """Return the name of the superoperator

        Returns:
            str: name of the superoperator
        """
        return self._name

    def apply(self, density_matrix: np.ndarray, qubits: int | list[int]) -> np.ndarray:
        """
        Apply the superoperator to the density matrix

        Args:
            density_matrix (np.ndarray): density matrix to apply the superoperator to
            quibits (int | list[int]): qubits to apply the superoperator to

        Returns:
            np.ndarray: resulting density matrix
        """
        if isinstance(qubits, int):
            qubits = [qubits]
        if density_matrix.ndim != 2:
            raise ValueError(
                f"Density matrix must be 2D, got shape {density_matrix.shape}"
            )
        num_qubits = int(np.log2(density_matrix.shape[0]))
        superop_num_qubits = int(np.log2(self._superoperator.shape[0])) // 2
        if len(qubits) != superop_num_qubits:
            raise ValueError(
                f"Number of qubits must match: length of qubits {len(qubits)} != "
                f"number of qubits of superoperator {superop_num_qubits}"
            )
        if num_qubits < superop_num_qubits:
            raise ValueError(
                f"Number of qubits of density matrix must be greater than or equal to "
                f"number of qubits of superoperator: {num_qubits} < {superop_num_qubits}"
            )

        reshaped_density_matrix = np.reshape(density_matrix, (2,) * num_qubits * 2)
        reshaped_superoperator = np.reshape(
            self._superoperator, (2,) * superop_num_qubits * 4
        )
        density_matrix_indices = list(range(num_qubits * 2))
        superoperator_indices = [
            num_qubits * 2 + i for i in range(superop_num_qubits * 4)
        ]
        output_indices = copy(density_matrix_indices)
        for idx, qubit_idx in enumerate(qubits):
            superoperator_indices[idx + superop_num_qubits * 3] = qubit_idx + num_qubits
            output_indices[qubit_idx + num_qubits] = superoperator_indices[
                idx + superop_num_qubits
            ]
            superoperator_indices[idx + superop_num_qubits * 2] = qubit_idx
            output_indices[qubit_idx] = superoperator_indices[idx]
        logger.debug(
            msg=f"Applying superoperator '{self._name}' to qubits {qubits}\n"
            f"input indices: {density_matrix_indices}\n"
            f"superoperator indices: {superoperator_indices}\n"
            f"output indices: {output_indices}\n"
        )
        reshaped_density_matrix = np.einsum(
            reshaped_density_matrix,
            density_matrix_indices,
            reshaped_superoperator,
            superoperator_indices,
            output_indices,
        )
        return reshaped_density_matrix.reshape((2**num_qubits, 2**num_qubits))
