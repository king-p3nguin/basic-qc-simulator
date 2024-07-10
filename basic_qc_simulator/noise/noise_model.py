"""
Module for noise models
"""

from ..gates import GateTypes
from .quantum_ops import KrausOperators


class CustomNoiseModel:
    """
    Class for a custom noise model
    """

    def __init__(self) -> None:
        self.readout_error = None
        self.gate_errors: dict[GateTypes, KrausOperators] = {}

    def add_noise_to_gate(self, gate_name: GateTypes, value: KrausOperators) -> None:
        """Set the kraus operators for a gate

        Args:
            gate_name (GateTypes): gate type
            value (KrausOperators): kraus operators
        """
        if not gate_name in GateTypes:
            raise TypeError(f"gate_name {gate_name} must be of type GateTypes")
        if not isinstance(value, KrausOperators):
            raise TypeError(f"Value {value} must be of type KrausOperators")
        self.gate_errors[gate_name] = value
