"""
Module for the density matrix simulator.
"""

import logging
from copy import copy

import numpy as np

from ..circuit import Instruction
from ..quantum_info.ops import KrausOperators
from ..quantum_info.states.density_matrix import DensityMatrix
from ..simulator_result import SimulatorResult, SimulatorResultTypes
from .abstract_simulator import AbstractSimulator

logger = logging.getLogger(__name__)


class DensityMatrixSimulator(AbstractSimulator):
    """
    Class for the density matrix simulator
    """

    def _prepare_state(self, num_qubits: int) -> DensityMatrix:
        """Prepare the initial density matrix

        Args:
            num_qubits (int): number of qubits

        Returns:
            DensityMatrix: initial density matrix
        """
        # Initialize the state vector to |0>^n<0|^n
        density_matrix = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
        density_matrix[0, 0] = 1.0
        return DensityMatrix(density_matrix)

    @staticmethod
    def _apply_gate(instruction: Instruction, state: DensityMatrix) -> DensityMatrix:
        """Apply a gate to the density matrix

        Args:
            instruction (Instruction): gate instruction
            state (DensityMatrix): density matrix

        Returns:
            DensityMatrix: resulting density matrix
        """
        return state.apply_gate(instruction.gate, qargs=instruction.qubits)

    def _apply_noise(
        self, state: DensityMatrix, noise: KrausOperators, qubits: list[int]
    ) -> DensityMatrix:
        """Apply noise to the density matrix

        Args:
            state (DensityMatrix): density matrix to apply the noise to
            noise (KrausOperators): noise to apply
            qubits (list[int]): qubits to apply the noise to

        Returns:
            DensityMatrix: resulting density matrix
        """
        return noise.to_superoperator().apply(state, qubits)

    def _save_result(self, save_resut_dict: dict, state: np.ndarray) -> None:
        """Save the result of the simulation

        Args:
            save_resut_dict (dict): dictionary with the saving result information
        """
        if save_resut_dict["result_type"] != SimulatorResultTypes.DENSITY_MATRIX:
            raise ValueError(
                "Density matrix simulator does not support saving "
                f"{save_resut_dict['result_type']} result."
            )
        self._results.append(
            SimulatorResult(
                result_type=SimulatorResultTypes.DENSITY_MATRIX,
                result=copy(state),
            )
        )
