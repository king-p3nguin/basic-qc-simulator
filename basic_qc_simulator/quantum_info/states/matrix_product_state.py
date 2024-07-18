"""
Module for matrix product states.
"""

import logging

import numpy as np

from ...gates import Gate
from .state_vector import StateVector

logger = logging.getLogger(__name__)


class MatrixProductState:
    """
    Class for matrix product states.
    Vidal's matrix product state representation is used.

    Reference:
    G. Vidal, Efficient Classical Simulation of Slightly Entangled Quantum Computations,
    Phys. Rev. Lett. 91, 147902 (2003).

    U. Schollwöck, The Density-Matrix Renormalization Group in the Age of Matrix Product States,
    Ann. Phys. 326, 96 (2011).
    """

    def __init__(self, gammas: list[np.ndarray], lambdas: list[np.ndarray]) -> None:
        if len(gammas) != len(lambdas) + 1:
            raise ValueError(
                "Number of gammas must be equal to the number of lambdas + 1"
            )
        self.gammas = gammas
        self.lambdas = lambdas
        self.num_qubits = len(gammas)

    def __repr__(self) -> str:
        gammas_str = "[\n\t" + ",\n\t".join(map(str, self.gammas)) + ",\n]"
        lambdas_str = "[\n\t" + ",\n\t".join(map(str, self.lambdas)) + ",\n]"
        return "MatrixProductState(gammas={}, lambdas={})".format(
            gammas_str, lambdas_str
        )

    def apply_gate(
        self, gate: Gate, qargs: int | list[int], inplace: bool = False
    ) -> "MatrixProductState":
        """Apply a gate to the matrix product state

        Args:
            gate (Gate): gate
            qargs (list[int]): qubits to apply the gate to
            inplace (bool): apply the gate inplace (default: False)

        Returns:
            MatrixProductState: resulting matrix product state
        """
        if isinstance(qargs, int):
            qargs = [qargs]
        if gate.num_qubits == 1:
            return MatrixProductState._apply_single_qubit_gate(gate, qargs, inplace)
        elif gate.num_qubits == 2:
            return MatrixProductState._apply_two_qubit_gate(gate, qargs, inplace)
        else:
            raise ValueError("Only single and two qubit gates are supported")

    def _apply_single_qubit_gate(
        self, gate: Gate, qargs: list[int], inplace: bool = False
    ) -> "MatrixProductState":
        """Apply a single qubit gate to the matrix product state

        Args:
            gate (Gate): gate
            qargs (list[int]): qubits to apply the gate to
            inplace (bool): apply the gate inplace (default: False)

        Returns:
            MatrixProductState: resulting matrix product state
        """
        gate_matrix = gate.matrix
        qubit = qargs[0]
        logger.debug(msg=f"Applying gate '{gate.name}' " f"to qubits {qargs}")
        self.gammas[qubit] = np.tensordot(gate_matrix, self.gammas[qubit], axes=1)
        # if qubit == 0:  # leftmost qubit
        #     gammas[qubit] = np.einsum("ij,jl->il", gate_matrix, gammas[qubit])
        # elif qubit == len(gammas) - 1:  # rightmost qubit
        #     gammas[qubit] = np.einsum("ij,lj->li", gate_matrix, gammas[qubit])
        # else:  # middle qubit
        #     gammas[qubit] = np.einsum("ij,kjl->kil", gate_matrix, gammas[qubit])
        logger.debug(msg=f"gamma[{qubit}] after application: {self.gammas[qubit]}")

    def _apply_two_qubit_gate(
        self, gate: Gate, qargs: list[int], inplace: bool = False
    ) -> "MatrixProductState":
        """Apply a two qubit gate to the matrix product state

        Args:
            gate (Gate): gate
            qargs (list[int]): qubits to apply the gate to
            inplace (bool): apply the gate inplace (default: False)

        Returns:
            MatrixProductState: resulting matrix product state
        """
        raise NotImplementedError
        gammas, lambdas = state
        gate_matrix = gate.matrix
        qubit1, qubit2 = instruction.qubits

        # Combine the two qubits into a single one
        gammas[qubit1] = np.tensordot(gammas[qubit1], gammas[qubit2], axes=(0, 0))
        gammas[qubit1] = np.tensordot(
            gammas[qubit1], gate_matrix, axes=([1, 2], [0, 1])
        )
        gammas.pop(qubit2)

    @staticmethod
    def from_state_vector(
        state_vector: StateVector, truncate: bool = True
    ) -> "MatrixProductState":
        r"""Convert a state vector to a matrix product state.

        Args:
            state_vector (np.ndarray): state vector

        Returns:
            MatrixProductState: Vidal's matrix product state representation
        """
        # pylint: disable=invalid-name
        num_qubits = state_vector.num_qubits
        gammas, lambdas = [], []

        logger.debug(
            "Converting state vector to Vidal's matrix product state representation"
        )
        rank = 1
        for i in range(num_qubits - 1):
            # Reshape the state vector to a matrix
            #     2   2   2   2   2   2
            #  ┌┴─┴─┴─┴─┴─┴┐           ┌───────────┐
            #  │                      │ →  2r  ─┤                      ├─ 2^(n-1)
            #  └───────────┘           └───────────┘
            col_dim = 2 * rank
            row_dim = 2 ** (num_qubits - i - 1)
            state_vector = np.reshape(state_vector, (col_dim, row_dim))
            U, S, V_dagger = np.linalg.svd(state_vector, full_matrices=True)
            logger.debug(
                "Singular Value Decomposition: U=%s, S=%s, V_dagger=%s",
                U.shape,
                S.shape,
                V_dagger.shape,
            )
            if truncate:
                S = S[np.isclose(S, 0.0) == False]
                logger.debug(f"Nonzero Singular Values: {S}")
            rank = S.shape[0]
            if col_dim <= row_dim:
                state_vector = np.diag(S) @ V_dagger[:rank, :]
            else:
                U = U[:, :rank]
                state_vector = np.diag(S) @ V_dagger
            gammas.append(
                (np.expand_dims(U[0, :rank], 0), np.expand_dims(U[1, :rank], 0))
            )
            lambdas.append(S)

        gammas.append(
            (
                np.expand_dims(V_dagger[:rank, 0], 0),
                np.expand_dims(V_dagger[:rank, 1], 0),
            )
        )

        return MatrixProductState(gammas, lambdas)

    def get_amplitude(self, bit_string: str, little_endian: bool = False) -> complex:
        """Get the amplitude of a bit string.

        Args:
            bit_string (str): bit string
            little_endian (bool): little endian (default: False)

        Returns:
            complex: amplitude of the bit string
        """
        if len(bit_string) != self.num_qubits:
            raise ValueError(
                f"length of bit_string ({len(bit_string)}) is not equal to "
                f"the number of qubits ({self.num_qubits}) of the MPS"
            )
        if little_endian:
            bit_string = bit_string[::-1]

        bit_number = int(bit_string[0])
        amplitude = self.gammas[0][bit_number]
        for idx, bit_number in enumerate(bit_string[1:]):
            bit_number = int(bit_number)
            amplitude *= self.lambdas[idx] @ self.gammas[idx + 1][bit_number]
        return amplitude

    def to_state_vector(self, little_endian: bool = False) -> StateVector:
        """Convert the matrix product state to a state vector.

        Args:
            little_endian (bool): little endian (default: False)

        Returns:
            StateVector: state vector
        """
        sv = []
        for i in range(2**self.num_qubits):
            amp = self.get_amplitude(bin(i)[2:].zfill(self.num_qubits), little_endian)
            sv.append(amp)
        return StateVector(sv)
