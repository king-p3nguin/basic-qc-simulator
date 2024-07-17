"""
Module for the state vector simulator.
"""

import logging
from copy import copy

import numpy as np

from ..circuit import Instruction
from ..simulator_result import SimulatorResult, SimulatorResultTypes
from .abstract_simulator import AbstractSimulator

logger = logging.getLogger(__name__)

# TODO: make StateVector class and add it to quantum_info.states


class StateVectorSimulator(AbstractSimulator):
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
        # Initialize the state vector to |0>^n
        state_vector = np.zeros(2**num_qubits, dtype=complex)
        state_vector[0] = 1.0
        # Reshape the flattened state vector to (2, 2, ..., 2) tensor
        state_vector = np.reshape(state_vector, (2,) * num_qubits)
        return state_vector

    @staticmethod
    def _apply_gate(instruction: Instruction, state: np.ndarray) -> np.ndarray:
        """Apply a gate to the state vector

        Args:
            instruction (Instruction): gate instruction
            state (np.ndarray): state vector

        Returns:
            np.ndarray: resulting state vector
        """
        circuit_num_qubits = len(state.shape)
        gate_matrix = instruction.gate.matrix.reshape(
            (2,) * instruction.gate.num_qubits * 2
        )
        gate_num_qubits = instruction.gate.num_qubits

        #           sv_tensor_indices                    new_sv_tensor_indices
        #        ┌──┐
        #   q_0: ┤ H ├─  0 ───────────────── 0
        #        │   │                 ┌──┐
        #   q_1: ┤   ├─  1 ─ qb[0] ─┤   ├─  n  ───── n
        #        │   │                 │   │
        #   q_2: ┤   ├─  2 ─ qb[1] ─┤   ├─ n+1 ───── n+1
        #        │   │                 │   │
        #   q_3: ┤   ├─  3 ─ qb[2] ─┤   ├─ n+2 ───── n+2
        #    :   │   │                └──┘
        #    :   │   │           gate_tensor_indices
        #    :   │   │
        # q_n-1: ┤   ├─ n-1─────────────────  n-1
        #        └──┘

        # state_vector_tensor_indices = [0, 1, 2, ..., n-1]
        state_vector_tensor_indices = list(range(circuit_num_qubits))
        # gate_tensor_indices
        #   = [n, n+1, n+2, ..., n+m-1, qubits[0], qubits[1], ..., qubits[m-1]]
        gate_tensor_indices = list(
            range(circuit_num_qubits, circuit_num_qubits + gate_num_qubits)
        ) + list(instruction.qubits)
        # new_state_vector_tensor_indices = [0, 1, 2, ..., n-1]
        # with qubits[i] replaced by gate_tensor_indices[i]
        new_state_vector_tensor_indices = copy(state_vector_tensor_indices)
        for i in range(gate_num_qubits):
            new_state_vector_tensor_indices[instruction.qubits[i]] = (
                gate_tensor_indices[i]
            )

        logger.debug(
            msg=f"Applying gate '{instruction.gate.name}' "
            f"to qubits {instruction.qubits}\n"
            f"input indices: {state_vector_tensor_indices}\n"
            f"gate indices: {gate_tensor_indices}\n"
            f"output indices: {new_state_vector_tensor_indices}\n"
        )
        # Apply the gate by contracting the state vector tensor with the gate tensor
        state = np.einsum(
            state,
            state_vector_tensor_indices,
            gate_matrix,
            gate_tensor_indices,
            new_state_vector_tensor_indices,
        )
        return state

    def _save_result(self, save_resut_dict: dict, state: np.ndarray) -> None:
        """Save the result of the simulation

        Args:
            save_resut_dict (dict): dictionary with the saving result information
            state (np.ndarray): state vector to save
        """
        if save_resut_dict["result_type"] != SimulatorResultTypes.STATE_VECTOR:
            raise ValueError(
                "State vector simulator does not support saving "
                f"{save_resut_dict['result_type']} result."
            )
        self._results.append(
            SimulatorResult(
                result_type=SimulatorResultTypes.STATE_VECTOR,
                result=state.flatten().copy(),
            )
        )
