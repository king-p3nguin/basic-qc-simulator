"""
Module for the matrix product state simulator.
"""

import logging

import numpy as np

from ..circuit import Instruction
from ..simulator_result import SimulatorResult, SimulatorResultTypes
from .abstract_simulator import AbstractSimulator

logger = logging.getLogger(__name__)


class MatrixProductStateSimulator(AbstractSimulator):
    """
    Class for the matrix product state simulator
    """

    def _prepare_state(self, num_qubits: int) -> np.ndarray:
        """Prepare the initial matrix product state in Vidal's representation

        Args:
            num_qubits (int): number of qubits

        Returns:
            np.ndarray: initial matrix product state in Vidal's representation
        """
        gamma = (np.array([[1.0 + 0.0j]]), np.array([[0.0 + 0.0j]]))
        lambda_ = np.array([1.0])
        return ([gamma] * num_qubits, [lambda_] * (num_qubits - 1))

    @staticmethod
    def _apply_gate(instruction: Instruction, state: np.ndarray) -> np.ndarray:
        """Apply a gate to the matrix product state

        Args:
            instruction (Instruction): gate instruction
            state (np.ndarray): matrix product state

        Returns:
            np.ndarray: resulting matrix product state
        """
        if instruction.gate.num_qubits == 1:
            return MatrixProductStateSimulator._apply_single_qubit_gate(
                instruction, state
            )
        elif instruction.gate.num_qubits == 2:
            return MatrixProductStateSimulator._apply_two_qubit_gate(instruction, state)
        else:
            raise ValueError("Only single and two qubit gates are supported")

    @staticmethod
    def _apply_single_qubit_gate(instruction: Instruction, state: np.ndarray) -> None:
        """Apply a single qubit gate to the matrix product state

        Args:
            instruction (Instruction): gate instruction
            state (np.ndarray): matrix product state
        """
        gammas, _ = state
        gate_matrix = instruction.gate.matrix
        qubit = instruction.qubits[0]
        logger.debug(
            msg=f"Applying gate '{instruction.gate.name}' "
            f"to qubits {instruction.qubits}"
        )
        gammas[qubit] = np.tensordot(gate_matrix, gammas[qubit], axes=1)
        # if qubit == 0:  # leftmost qubit
        #     gammas[qubit] = np.einsum("ij,jl->il", gate_matrix, gammas[qubit])
        # elif qubit == len(gammas) - 1:  # rightmost qubit
        #     gammas[qubit] = np.einsum("ij,lj->li", gate_matrix, gammas[qubit])
        # else:  # middle qubit
        #     gammas[qubit] = np.einsum("ij,kjl->kil", gate_matrix, gammas[qubit])
        logger.debug(msg=f"gamma[{qubit}] after application: {gammas[qubit]}")

    @staticmethod
    def _apply_two_qubit_gate(instruction: Instruction, state: np.ndarray) -> None:
        """Apply a two qubit gate to the matrix product state

        Args:
            instruction (Instruction): gate instruction
            state (np.ndarray): matrix product state
        """
        raise NotImplementedError
        gammas, lambdas = state
        gate_matrix = instruction.gate.matrix
        qubit1, qubit2 = instruction.qubits

        # Combine the two qubits into a single one
        gammas[qubit1] = np.tensordot(gammas[qubit1], gammas[qubit2], axes=(0, 0))
        gammas[qubit1] = np.tensordot(
            gammas[qubit1], gate_matrix, axes=([1, 2], [0, 1])
        )
        gammas.pop(qubit2)

    def _save_result(self, save_resut_dict: dict, state: np.ndarray) -> None:
        """Save the result of the simulation

        Args:
            save_resut_dict (dict): dictionary with the saving result information
            state (np.ndarray): matrix product state to save
        """
        if save_resut_dict["result_type"] != SimulatorResultTypes.MATRIX_PRODUCT_STATE:
            raise ValueError(
                "State vector simulator does not support saving "
                f"{save_resut_dict['result_type']} result."
            )
        self._results.append(
            SimulatorResult(
                result_type=SimulatorResultTypes.MATRIX_PRODUCT_STATE,
                result=state,
            )
        )

    @staticmethod
    def _state_vector_to_left_canonical_matrix_product_state(
        state_vector: np.ndarray, truncate: bool = False
    ) -> np.ndarray:
        """Convert a state vector to a left-canonical matrix product state representation

        Args:
            state_vector (np.ndarray): state vector

        Returns:
            np.ndarray: left-canonical matrix product state representation
        """
        # pylint: disable=invalid-name
        assert state_vector.ndim == 1
        num_qubits = int(np.log2(state_vector.shape[0]))
        matrices = []

        logger.debug(
            "Converting state vector to left-canonical matrix product state representation"
        )
        rank = 1
        for i in range(num_qubits - 1):
            # Reshape the state vector to a matrix
            #     2   2   2   2   2   2
            #  ┌┴─┴─┴─┴─┴─┴┐           ┌───────────┐
            #  │                      │ →  2r  ─┤                      ├─ 2^(n-1)
            #  └───────────┘           └───────────┘
            state_vector = state_vector.reshape(2 * rank, 2 ** (num_qubits - i - 1))
            U, S, V_dagger = np.linalg.svd(state_vector, full_matrices=True)
            logger.debug(
                f"Singular Value Decomposition: U={U.shape}, S={S.shape}, V_dagger={V_dagger.shape}\n"
                f"Singular Values: {S}"
            )
            if truncate:
                S = S[np.nonzero(S)]
                logger.debug(f"Nonzero Singular Values: {S}")
            rank = S.shape[0]
            state_vector = np.diag(S) @ V_dagger[:rank, :]
            matrices.append(U)

        matrices.append(state_vector)

        return matrices

    @staticmethod
    def _state_vector_to_vidal_matrix_product_state(
        state_vector: np.ndarray, truncate: bool = True
    ) -> np.ndarray:
        """Convert a state vector to a Vidal's matrix product state representation

        Reference:
        G. Vidal, Efficient Classical Simulation of Slightly Entangled Quantum Computations,
        Phys. Rev. Lett. 91, 147902 (2003).

        U. Schollwöck, The Density-Matrix Renormalization Group in the Age of Matrix Product States,
        Ann. Phys. 326, 96 (2011).

        Args:
            state_vector (np.ndarray): state vector

        Returns:
            np.ndarray: Vidal's matrix product state representation
        """
        # pylint: disable=invalid-name
        assert state_vector.ndim == 1
        num_qubits = int(np.log2(state_vector.shape[0]))
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
            state_vector = state_vector.reshape(col_dim, row_dim)
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
            gammas.append((U[0, :rank], U[1, :rank]))
            lambdas.append(S)

        gammas.append((V_dagger[:rank, 0], V_dagger[:rank, 1]))

        return gammas, lambdas
