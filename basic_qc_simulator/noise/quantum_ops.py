"""
Module for quantum operations
"""

import numpy as np


class KrausOperators(list):
    """
    Class for kraus operators
    """

    def __init__(self, kraus_operators: list[np.ndarray], name: str) -> None:
        super().__init__(kraus_operators)
        self._name = name

    @property
    def name(self) -> str:
        """Return the name of the kraus operators

        Returns:
            str: name of the kraus operators
        """
        return self._name

    def apply(self, density_matrix: np.ndarray) -> np.ndarray:
        """
        Apply the kraus operators to the density matrix

        Args:
            density_matrix (np.ndarray): density matrix to apply the kraus operators to

        Returns:
            np.ndarray: resulting density matrix
        """
        for kraus_operator in self:
            density_matrix = np.dot(
                np.dot(kraus_operator, density_matrix), np.conj(kraus_operator).T
            )
        return density_matrix

    def to_superoperator(self) -> "Superoperator":
        """
        Convert the kraus operators to a superoperator

        Returns:
            Superoperator: superoperator
        """
        superop = np.sum(
            [np.kron(np.conj(krausop), krausop) for krausop in self], axis=0
        )
        return Superoperator(superop, self.name)


class Superoperator:
    """
    Class for a superoperator
    """

    def __init__(self, superoperator: np.ndarray, name: str) -> None:
        self._superoperator = superoperator
        self._name = name

    def __repr__(self) -> str:
        return self._superoperator.__repr__()

    @property
    def name(self) -> str:
        """Return the name of the superoperator

        Returns:
            str: name of the superoperator
        """
        return self._name

    def apply(self, density_matrix: np.ndarray) -> np.ndarray:
        """
        Apply the superoperator to the density matrix

        Args:
            density_matrix (np.ndarray): density matrix to apply the superoperator to

        Returns:
            np.ndarray: resulting density matrix
        """
        raise NotImplementedError
