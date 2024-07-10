"""
Module for the base simulator class
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..circuit import Circuit, Instruction
from ..simulator_result import SimulatorResult

if TYPE_CHECKING:
    from ..noise.noise_channel import KrausOperators
    from ..noise.noise_model import CustomNoiseModel

logger = logging.getLogger(__name__)


class AbstractSimulator(ABC):
    """
    Abstract base class for simulators
    """

    def __init__(self, noise_model: Optional["CustomNoiseModel"] = None) -> None:
        super().__init__()
        self._results: list[SimulatorResult] = []
        self._noise_model = noise_model

    @property
    def results(self) -> list[SimulatorResult]:
        """Return the results of the simulation

        Returns:
            list[SimulatorResult]: results of the simulation
        """
        return self._results

    def run(self, circuit: Circuit) -> None:
        """Run the circuit on the simulator

        Args:
            circuit (Circuit): circuit to run
        """
        state = self._prepare_state(circuit.num_qubits)

        # Apply the gates in the circuit
        for index, instruction in enumerate(circuit.instructions):
            # Save the result if needed
            if circuit.saving_results.get(index) is not None:
                self._save_result(circuit.saving_results[index], state)

            state = self._apply_gate(instruction, state)

            if (
                self._noise_model is not None
                and instruction.gate.name in self._noise_model.gate_errors
            ):
                state = self._apply_noise(
                    state,
                    self._noise_model.gate_errors[instruction.gate.name],
                    instruction.qubits,
                )

        # Save the result if needed
        if circuit.saving_results.get(len(circuit.instructions)) is not None:
            self._save_result(circuit.saving_results[len(circuit.instructions)], state)

    @abstractmethod
    def _prepare_state(self, num_qubits: int) -> np.ndarray:
        """Prepare the initial state

        Args:
            num_qubits (int): number of qubits

        Returns:
            np.ndarray: initial state
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _apply_gate(instruction: Instruction, state: np.ndarray) -> np.ndarray:
        """Apply a gate to the state

        Args:
            instruction (Instruction): gate instruction
            state (np.ndarray): state to apply the gate to

        Returns:
            np.ndarray: resulting state
        """
        raise NotImplementedError

    # @staticmethod
    # @abstractmethod
    def _apply_noise(
        self, state: np.ndarray, noise: "KrausOperators", qubits: list[int]
    ) -> np.ndarray:
        """Apply noise to the state

        Args:
            state (np.ndarray): state to apply the noise to
            noise (KrausOperators): noise to apply
            quibits (list[int]): qubits to apply the noise to

        Returns:
            np.ndarray: resulting state
        """
        raise NotImplementedError

    @abstractmethod
    def _save_result(self, save_resut_dict: dict, state: np.ndarray) -> None:
        """Save the result of the simulation

        Args:
            save_resut_dict (dict): dictionary with the saving result information
            state (np.ndarray): state to save
        """
        raise NotImplementedError
