"""
Module for the state vector simulator.
"""

import logging
from copy import copy

import numpy as np

from ..circuit import Instruction
from ..quantum_info.states.state_vector import StateVector
from ..simulator_result import SimulatorResult, SimulatorResultTypes
from .abstract_simulator import AbstractSimulator

logger = logging.getLogger(__name__)


class StateVectorSimulator(AbstractSimulator):
    """
    Class for the state vector simulator
    """

    def _prepare_state(self, num_qubits: int) -> StateVector:
        """Prepare the initial state vector

        Args:
            num_qubits (int): number of qubits

        Returns:
            StateVector: initial state vector
        """
        # Initialize the state vector to |0>^n
        state_vector = np.zeros(2**num_qubits, dtype=complex)
        state_vector[0] = 1.0
        return StateVector(state_vector)

    @staticmethod
    def _apply_gate(instruction: Instruction, state: StateVector) -> StateVector:
        """Apply a gate to the state vector

        Args:
            instruction (Instruction): gate instruction
            state (StateVector): state vector

        Returns:
            StateVector: resulting state vector
        """
        return state.apply_gate(instruction.gate, qargs=instruction.qubits)

    def _save_result(self, save_resut_dict: dict, state: StateVector) -> None:
        """Save the result of the simulation

        Args:
            save_resut_dict (dict): dictionary with the saving result information
            state (StateVector): state vector to save
        """
        if save_resut_dict["result_type"] != SimulatorResultTypes.STATE_VECTOR:
            raise ValueError(
                "State vector simulator does not support saving "
                f"{save_resut_dict['result_type']} result."
            )
        self._results.append(
            SimulatorResult(
                result_type=SimulatorResultTypes.STATE_VECTOR,
                result=copy(state),
            )
        )
