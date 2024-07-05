"""
Module for the density matrix simulator.
"""

import logging
from copy import copy

import numpy as np

from ..circuit import Circuit, Instruction
from ..simulator_result import SimulatorResult, SimulatorResultTypes
from .abstract_simulator import AbstractSimulator

logger = logging.getLogger(__name__)


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

        # Apply the gates in the circuit
        for index, instruction in enumerate(circuit.instructions):
            # Save the result if needed
            if circuit.saving_results.get(index) is not None:
                self._save_result(circuit.saving_results[index])

            self._apply_gate(instruction)

        # Save the result if needed
        if circuit.saving_results.get(len(circuit.instructions)) is not None:
            self._save_result(circuit.saving_results[len(circuit.instructions)])

    def _apply_gate(self, instruction: Instruction) -> None:
        """Apply a gate to the density matrix

        Args:
            instruction (Instruction): gate instruction
        """
        circuit_num_qubits = len(self._density_matrix.shape) // 2
        gate_matrix = instruction.gate.matrix.reshape(
            (2,) * instruction.gate.num_qubits * 2
        )
        gate_matrix_conj = np.conj(gate_matrix)
        gate_num_qubits = instruction.gate.num_qubits

        # gate_tensor_indices
        #   = [2n, 2n+1, 2n+2, ..., 2n+m-1, qubits[0], qubits[1], ..., qubits[m-1]]
        gate_tensor_indices = list(
            range(circuit_num_qubits * 2, circuit_num_qubits * 2 + gate_num_qubits)
        ) + list(instruction.qubits)
        # gate_tensor_conj_indices
        #   = [2n+m, 2n+m+1, 2n+m+2, ..., 2n+2m-1, n+qubits[0], n+qubits[1], ..., n+qubits[m-1]]
        gate_tensor_conj_indices = list(
            range(
                circuit_num_qubits * 2 + gate_num_qubits,
                circuit_num_qubits * 2 + gate_num_qubits * 2,
            )
        ) + [circuit_num_qubits + q for q in instruction.qubits]

        # density_matrix_tensor_indices = [0, 1, 2, ..., n-1, n, n+1, ..., 2n-1]
        density_matrix_tensor_indices = list(range(circuit_num_qubits * 2))

        # new_density_matrix_tensor_indices = [0, 1, 2, ..., n-1, n, n+1, ..., 2n-1]
        # with qubits[i] replaced by gate_tensor_indices[i]
        # and n+qubits[i] replaced by gate_tensor_conj_indices[i]
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
        self._density_matrix = np.einsum(
            self._density_matrix,
            density_matrix_tensor_indices,
            gate_matrix,
            gate_tensor_indices,
            new_density_matrix_tensor_indices,
        )

        # new_density_matrix_tensor_indices = [0, 1, 2, ..., n-1, n, n+1, ..., 2n-1]
        # with n+qubits[i] replaced by gate_tensor_conj_indices[i]
        new_density_matrix_tensor_indices = copy(density_matrix_tensor_indices)
        for i in range(gate_num_qubits):
            new_density_matrix_tensor_indices[
                circuit_num_qubits + instruction.qubits[i]
            ] = gate_tensor_conj_indices[i]

        logger.debug(
            msg=f"Applying gate '{instruction.gate.name}' conjugate "
            f"to qubits {instruction.qubits}\n"
            f"input indices: {density_matrix_tensor_indices}\n"
            f"gate indices: {gate_tensor_conj_indices}\n"
            f"output indices: {new_density_matrix_tensor_indices}\n"
        )
        # Apply the conjugate gate by contracting the density matrix tensor
        # with the conjugate gate tensor
        self._density_matrix = np.einsum(
            self._density_matrix,
            density_matrix_tensor_indices,
            gate_matrix_conj,
            gate_tensor_conj_indices,
            new_density_matrix_tensor_indices,
        )

    def _save_result(self, save_resut_dict: dict) -> None:
        """Save the result of the simulation

        Args:
            save_resut_dict (dict): dictionary with the saving result information
        """
        if save_resut_dict["result_type"] != SimulatorResultTypes.DENSITY_MATRIX:
            raise ValueError(
                "Density matrix simulator does not support saving "
                f"{save_resut_dict['result_type']} result."
            )
        circuit_num_qubits = len(self._density_matrix.shape) // 2
        self._results.append(
            SimulatorResult(
                result_type=SimulatorResultTypes.DENSITY_MATRIX,
                result=self._density_matrix.reshape(
                    (2**circuit_num_qubits, 2**circuit_num_qubits)
                ).copy(),
            )
        )
