"""
Module for the base simulator class
"""

from abc import ABC, abstractmethod

from ..circuit import Circuit
from ..simulator_result import SimulatorResult


class AbstractSimulator(ABC):
    """
    Abstract base class for simulators
    """

    def __init__(self) -> None:
        super().__init__()
        self._results: list[SimulatorResult] = []

    @property
    def results(self) -> list[SimulatorResult]:
        """Return the results of the simulation

        Returns:
            list[SimulatorResult]: results of the simulation
        """
        return self._results

    @abstractmethod
    def run(self, circuit: Circuit) -> None:
        """Run the simulation

        Args:
            circuit (Circuit): quantum circuit to simulate
        """
        raise NotImplementedError

    @abstractmethod
    def _save_result(self, save_resut_dict: dict) -> None:
        """Save the result of the simulation

        Args:
            save_resut_dict (dict): dictionary with the saving result information
        """
        raise NotImplementedError
