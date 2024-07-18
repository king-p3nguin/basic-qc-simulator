"""
Module for the simulator result class.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class SimulatorResultTypes(StrEnum):
    """
    Enum class for simulator result types
    """

    STATE_VECTOR = "state_vector"
    COUNTS_DICT = "counts_dict"
    EXPECTATION_VALUES = "expectation_values"
    PROBABILITIES = "probabilities"
    AMPLITUDES = "amplitudes"
    DENSITY_MATRIX = "density_matrix"
    MATRIX_PRODUCT_STATE = "matrix_product_state"


@dataclass
class SimulatorResult:
    """
    Class for simulator results
    """

    result_type: SimulatorResultTypes
    result: Any
