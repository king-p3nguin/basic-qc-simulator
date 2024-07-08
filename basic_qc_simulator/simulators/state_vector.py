"""
Module for the state vector simulator.
"""

import logging
from copy import copy

import numpy as np

from ..circuit import Circuit
from ..simulator_result import SimulatorResult, SimulatorResultTypes
from .abstract_simulator import AbstractSimulator

logger = logging.getLogger(__name__)


class StateVectorSimulator(AbstractSimulator):
    """
    Class for the state vector simulator
    """

    def __init__(self) -> None:
        super().__init__()
        self._state_vector: np.ndarray

    def run(self, circuit: Circuit) -> None:
        """Run the circuit on the simulator

        Args:
            circuit (Circuit): circuit to run
        """
        # Initialize the state vector to |0>^n
        self._state_vector = np.zeros(2**circuit.num_qubits, dtype=complex)
        self._state_vector[0] = 1.0
        # Reshape the flattened state vector to (2, 2, ..., 2) tensor
        self._state_vector = np.reshape(self._state_vector, (2,) * circuit.num_qubits)

        # Apply the gates in the circuit
        for index, instruction in enumerate(circuit.instructions):
            # Save the result if needed
            if circuit.saving_results.get(index) is not None:
                self._save_result(circuit.saving_results[index])

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
            state_vector_tensor_indices = list(range(circuit.num_qubits))
            # gate_tensor_indices
            #   = [n, n+1, n+2, ..., n+m-1, qubits[0], qubits[1], ..., qubits[m-1]]
            gate_tensor_indices = list(
                range(circuit.num_qubits, circuit.num_qubits + gate_num_qubits)
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
            self._state_vector = np.einsum(
                self._state_vector,
                state_vector_tensor_indices,
                gate_matrix,
                gate_tensor_indices,
                new_state_vector_tensor_indices,
            )

        # Save the result if needed
        if circuit.saving_results.get(len(circuit.instructions)) is not None:
            self._save_result(circuit.saving_results[len(circuit.instructions)])

    def _save_result(self, save_resut_dict: dict) -> None:
        """Save the result of the simulation

        Args:
            save_resut_dict (dict): dictionary with the saving result information
        """
        if save_resut_dict["result_type"] != SimulatorResultTypes.STATE_VECTOR:
            raise ValueError(
                "State vector simulator does not support saving "
                f"{save_resut_dict['result_type']} result."
            )
        self._results.append(
            SimulatorResult(
                result_type=SimulatorResultTypes.STATE_VECTOR,
                result=self._state_vector.flatten().copy(),
            )
        )
