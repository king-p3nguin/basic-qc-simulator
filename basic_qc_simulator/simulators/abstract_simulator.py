"""
Module for the base simulator class
"""

from abc import ABC, abstractmethod

from ..circuit import Circuit


class AbstractSimulator(ABC):
    """
    Abstract base class for simulators
    """

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
