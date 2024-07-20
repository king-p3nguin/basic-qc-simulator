"""
Module for matrix product states.
"""

import logging
from copy import copy

import numpy as np

from ...gates import Gate
from .state_vector import StateVector

logger = logging.getLogger(__name__)

# TODO: use tensorly, quimb or tenpy for testing


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
        left_rank = 1
        right_rank = 1
        lam_inv = [1.0]
        d = 2
        for i in range(num_qubits - 1):
            # Reshape the state vector to a matrix
            #     2   2   2   2   2   2
            #  ┌┴─┴─┴─┴─┴─┴┐           ┌───────────┐
            #  │                      │ →  2r  ─┤                      ├─ 2^(n-1)
            #  └───────────┘           └───────────┘
            col_dim = d * left_rank
            row_dim = d ** (num_qubits - i - 1)
            state_vector = np.reshape(state_vector, (col_dim, row_dim))
            U, S, V_dagger = np.linalg.svd(state_vector, full_matrices=False)
            logger.debug(
                "Singular Value Decomposition: U=%s, S=%s, V_dagger=%s",
                U.shape,
                S.shape,
                V_dagger.shape,
            )
            if truncate:
                S = S[np.isclose(S, 0.0) == False]
                logger.debug(f"Nonzero Singular Values: {S}")
            lambdas.append(S)
            right_rank = S.size

            U = U[:, :right_rank]
            V_dagger = V_dagger[:right_rank, :]
            state_vector = np.diag(S) @ V_dagger

            gammas.append(
                np.tensordot(
                    np.diag(lam_inv), U.reshape(left_rank, d, right_rank), (1, 0)
                ).transpose(1, 0, 2)
            )
            lam_inv = 1.0 / S
            left_rank = right_rank

        V_dagger = V_dagger.reshape(d, d, 1).transpose(1, 0, 2)
        gammas.append(V_dagger)

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
        amplitude = self.gammas[0][bit_number]  # this is vector
        for idx, bit_number in enumerate(bit_string[1:]):
            bit_number = int(bit_number)
            amplitude = (
                amplitude
                @ np.diag(self.lambdas[idx])
                @ self.gammas[idx + 1][bit_number]
            )
        return amplitude

    def to_state_vector(self) -> StateVector:
        """Convert the matrix product state to a state vector.

        Returns:
            StateVector: state vector
        """
        # sv = []
        # for i in range(2**self.num_qubits):
        #     amp = self.get_amplitude(bin(i)[2:].zfill(self.num_qubits))
        #     sv.append(amp)
        # return StateVector(sv)

        rank = self.gammas[0].shape[2]
        d = 2
        vec = np.reshape(self.gammas[0], (d, rank))

        for i in range(len(self.lambdas)):
            vec = np.tensordot(
                np.tensordot(vec, np.diag(self.lambdas[i]), (i + 1, 0)),
                self.gammas[i + 1],
                (i + 1, 1),
            )
        return StateVector(vec.flatten())
