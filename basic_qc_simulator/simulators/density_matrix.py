"""
Module for the density matrix simulator.
"""

import logging
from copy import copy
from typing import TYPE_CHECKING

import numpy as np

from ..circuit import Instruction
from ..simulator_result import SimulatorResult, SimulatorResultTypes
from .abstract_simulator import AbstractSimulator

if TYPE_CHECKING:
    from ..noise.noise_channel import KrausOperators

logger = logging.getLogger(__name__)


class DensityMatrixSimulator(AbstractSimulator):
    """
    Class for the density matrix simulator
    """

    def _prepare_state(self, num_qubits: int) -> np.ndarray:
        """Prepare the initial density matrix

        Args:
            num_qubits (int): number of qubits

        Returns:
            np.ndarray: initial density matrix
        """
        # Initialize the state vector to |0>^n<0|^n
        density_matrix = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
        density_matrix[0, 0] = 1.0
        # Reshape to (2, 2, ..., 2) tensor
        density_matrix = np.reshape(density_matrix, (2,) * num_qubits * 2)
        return density_matrix

    @staticmethod
    def _apply_gate(instruction: Instruction, state: np.ndarray) -> np.ndarray:
        """Apply a gate to the density matrix

        Args:
            instruction (Instruction): gate instruction
        """
        circuit_num_qubits = len(state.shape) // 2
        gate_num_qubits = instruction.gate.num_qubits
        gate_matrix = instruction.gate.matrix.reshape((2,) * gate_num_qubits * 2)
        gate_matrix_conj = np.conj(instruction.gate.matrix).T.reshape(
            (2,) * gate_num_qubits * 2
        )

        # density_matrix_tensor_indices = [0, 1, 2, ..., n-1, n, n+1, ..., 2n-1]
        density_matrix_tensor_indices = list(range(circuit_num_qubits * 2))

        # gate_tensor_indices
        #   = [2n, 2n+1, 2n+2, ..., 2n+m-1, qubits[0], qubits[1], ..., qubits[m-1]]
        gate_tensor_indices = list(
            range(circuit_num_qubits * 2, circuit_num_qubits * 2 + gate_num_qubits)
        ) + list(instruction.qubits)

        # new_density_matrix_tensor_indices = [0, 1, 2, ..., n-1, n, n+1, ..., 2n-1]
        #   with qubits[i] replaced by gate_tensor_indices[i]
        new_density_matrix_tensor_indices = copy(density_matrix_tensor_indices)
        for i in range(gate_num_qubits):
            new_density_matrix_tensor_indices[instruction.qubits[i]] = (
                gate_tensor_indices[i]
            )

        logger.debug(
            msg=f"Applying gate '{instruction.gate.name}' to qubits {instruction.qubits}\n"
            f"input indices: {density_matrix_tensor_indices}\n"
            f"gate indices: {gate_tensor_indices}\n"
            f"output indices: {new_density_matrix_tensor_indices}\n"
        )
        # Apply the gate by contracting the density matrix tensor with the gate tensor
        state = np.einsum(
            state,
            density_matrix_tensor_indices,
            gate_matrix,
            gate_tensor_indices,
            new_density_matrix_tensor_indices,
        )

        # gate_tensor_conj_indices
        #   = [n+qubits[0], n+qubits[1], ..., n+qubits[m-1], 2n, 2n+1, 2n+2, ..., 2n+m-1]
        gate_tensor_conj_indices = [
            circuit_num_qubits + q for q in instruction.qubits
        ] + list(
            range(circuit_num_qubits * 2, circuit_num_qubits * 2 + gate_num_qubits)
        )

        # new_density_matrix_tensor_indices = [0, 1, 2, ..., n-1, n, n+1, ..., 2n-1]
        #   with n+qubits[i] replaced by gate_tensor_conj_indices[i]
        new_density_matrix_tensor_indices = copy(density_matrix_tensor_indices)
        for i in range(gate_num_qubits):
            new_density_matrix_tensor_indices[
                circuit_num_qubits + instruction.qubits[i]
            ] = gate_tensor_conj_indices[gate_num_qubits + i]

        logger.debug(
            msg=f"Applying gate '{instruction.gate.name}' conjugate "
            f"to qubits {instruction.qubits}\n"
            f"input indices: {density_matrix_tensor_indices}\n"
            f"gate indices: {gate_tensor_conj_indices}\n"
            f"output indices: {new_density_matrix_tensor_indices}\n"
        )
        # Apply the conjugate gate by contracting the density matrix tensor
        # with the conjugate gate tensor
        state = np.einsum(
            state,
            density_matrix_tensor_indices,
            gate_matrix_conj,
            gate_tensor_conj_indices,
            new_density_matrix_tensor_indices,
        )
        return state

    def _apply_noise(
        self, state: np.ndarray, noise: "KrausOperators", qubits: list[int]
    ) -> np.ndarray:
        """Apply noise to the density matrix

        Args:
            state (np.ndarray): density matrix to apply the noise to
            noise (KrausOperators): noise to apply
            qubits (list[int]): qubits to apply the noise to

        Returns:
            np.ndarray: resulting density matrix
        """
        num_qubits = len(state.shape) // 2
        state = np.reshape(state, (2**num_qubits, 2**num_qubits))
        state = noise.to_superoperator().apply(state, qubits)
        return state.reshape((2,) * num_qubits * 2)

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
        circuit_num_qubits = len(state.shape) // 2
        self._results.append(
            SimulatorResult(
                result_type=SimulatorResultTypes.DENSITY_MATRIX,
                result=state.reshape(
                    (2**circuit_num_qubits, 2**circuit_num_qubits)
                ).copy(),
            )
        )
