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
    Class for the state vector simulator
    """

    def _prepare_state(self, num_qubits: int) -> np.ndarray:
        """Prepare the initial state vector

        Args:
            num_qubits (int): number of qubits

        Returns:
            np.ndarray: initial state vector
        """
        raise NotImplementedError

    @staticmethod
    def _apply_gate(instruction: Instruction, state: np.ndarray) -> np.ndarray:
        """Apply a gate to the state vector

        Args:
            instruction (Instruction): gate instruction
            state (np.ndarray): state vector

        Returns:
            np.ndarray: resulting state vector
        """
        raise NotImplementedError

    def _save_result(self, save_resut_dict: dict, state: np.ndarray) -> None:
        """Save the result of the simulation

        Args:
            save_resut_dict (dict): dictionary with the saving result information
            state (np.ndarray): state vector to save
        """
        raise NotImplementedError
