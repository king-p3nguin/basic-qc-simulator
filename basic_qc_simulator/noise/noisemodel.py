"""
Module for noise models
"""

import numpy as np

from .quantum_ops import KrausOperators, PauliError


class CustomNoiseModel:
    """
    Class for a custom noise model
    """

    def __init__(self, noise_model: dict) -> None:
        self.noise_model = noise_model

    def apply(self, density_matrix: np.ndarray) -> np.ndarray:
        """
        Apply the noise model to the density matrix

        Args:
            density_matrix (np.ndarray): density matrix to apply the noise model to

        Returns:
            np.ndarray: resulting density matrix
        """
        for noise_op in self.noise_model:
            if noise_op == "kraus_operators":
                kraus_operators = KrausOperators(self.noise_model[noise_op])
                density_matrix = kraus_operators.apply(density_matrix)
            elif noise_op == "pauli_error":
                error = PauliError(self.noise_model[noise_op])
                density_matrix = error.apply(density_matrix)
            else:
                raise ValueError(f"Unknown noise operation: {noise_op}")

        return density_matrix
