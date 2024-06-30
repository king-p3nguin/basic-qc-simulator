"""
Module for the density matrix simulator.
"""

from copy import copy

import numpy as np

from ..circuit import Circuit
from ..simulator_result import SimulatorResult, SimulatorResultTypes
from .abstract_simulator import AbstractSimulator


class DensityMatrixSimulator(AbstractSimulator):
    """
    Class for the density matrix simulator
    """

    def __init__(self) -> None:
        super().__init__()
        self._density_matrix: np.ndarray

    def run(self, circuit: Circuit) -> None:
        """Run the circuit on the simulator

        Args:
            circuit (Circuit): circuit to run
        """
        # Initialize the state vector to |0>^n<0|^n
        self._density_matrix = np.zeros(
            (2**circuit.num_qubits, 2**circuit.num_qubits), dtype=complex
        )
        self._density_matrix[0, 0] = 1.0
        # Reshape to (2, 2, ..., 2) tensor
        self._density_matrix = np.reshape(
            self._density_matrix, (2,) * circuit.num_qubits * 2
        )
        raise NotImplementedError

        # Apply the gates in the circuit
        # for index, instruction in enumerate(circuit.instructions):
        #     # Save the result if needed
        #     if circuit.saving_results.get(index) is not None:
        #         self._save_result(circuit.saving_results[index])

        # # Save the result if needed
        # if circuit.saving_results.get(len(circuit.instructions)) is not None:
        #     self._save_result(circuit.saving_results[len(circuit.instructions)])

    def _save_result(self, save_resut_dict: dict) -> None:
        """Save the result of the simulation

        Args:
            save_resut_dict (dict): dictionary with the saving result information
        """
        raise NotImplementedError
        # if save_resut_dict["result_type"] != SimulatorResultTypes.DENSITY_MATRIX:
        #     raise ValueError(
        #         "Density matrix simulator does not support saving "
        #         f"{save_resut_dict['result_type']} result."
        #     )
        # self._results.append(
        #     SimulatorResult(
        #         result_type=SimulatorResultTypes.DENSITY_MATRIX,
        #         result=self._density_matrix.reshape(
        #             (2**self._num_qubits, 2**self._num_qubits)
        #         ).copy(),
        #     )
        # )
