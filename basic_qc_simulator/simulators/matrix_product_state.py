"""
Module for the matrix product state simulator.
"""

import logging

import numpy as np

from ..circuit import Instruction
from .abstract_simulator import AbstractSimulator

logger = logging.getLogger(__name__)


class MatrixProductStateSimulator(AbstractSimulator):
    """
    Class for the matrix product state simulator
    """

    def _prepare_state(self, num_qubits: int) -> np.ndarray:
        """Prepare the initial matrix product state

        Args:
            num_qubits (int): number of qubits

        Returns:
            np.ndarray: initial matrix product state
        """
        raise NotImplementedError

    @staticmethod
    def _apply_gate(instruction: Instruction, state: np.ndarray) -> np.ndarray:
        """Apply a gate to the matrix product state

        Args:
            instruction (Instruction): gate instruction
            state (np.ndarray): matrix product state

        Returns:
            np.ndarray: resulting matrix product state
        """
        raise NotImplementedError

    def _save_result(self, save_resut_dict: dict, state: np.ndarray) -> None:
        """Save the result of the simulation

        Args:
            save_resut_dict (dict): dictionary with the saving result information
            state (np.ndarray): matrix product state to save
        """
        raise NotImplementedError

    @staticmethod
    def _state_vector_to_matrix_product_state(state_vector: np.ndarray) -> np.ndarray:
        """Convert a state vector to a matrix product state

        Args:
            state_vector (np.ndarray): state vector

        Returns:
            np.ndarray: matrix product state
        """
        # pylint: disable=invalid-name
        assert state_vector.ndim == 1
        num_qubits = int(np.log2(state_vector.shape[0]))
        arrs = []

        r = 1
        for i in range(num_qubits - 1):
            # Reshape the state vector to a matrix
            #     2   2   2   2   2   2
            #  ┌┴─┴─┴─┴─┴─┴┐        ┌───────────┐
            #  │                      │ →  	─┤                      ├─
            #  └───────────┘        └───────────┘
            state_vector = state_vector.reshape(2 * r, 2 ** (num_qubits - i - 1))
            U, S, V_dagger = np.linalg.svd(state_vector, full_matrices=False)
            tmp_A = U @ np.diag(S)
            state_vector = V_dagger
            r = tmp_A.shape[1]
            col = tmp_A.shape[0] // 2
            arrs.append([tmp_A[:col, :], tmp_A[col : 2 * col, :]])

        col2 = state_vector.shape[1] // 2
        arrs.append([state_vector[:, :col2], state_vector[:, col2 : 2 * col2]])
        return arrs
