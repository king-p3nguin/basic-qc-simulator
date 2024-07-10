"""
Module for quantum gates and their implementations
"""

from enum import StrEnum
from typing import Any

import numpy as np


class GateTypes(StrEnum):
    """
    Enum class for gate types
    """

    I = "i"
    X = "x"
    Y = "y"
    Z = "z"
    H = "h"
    S = "s"
    T = "t"
    PHASE = "phase"
    RX = "rx"
    RY = "ry"
    RZ = "rz"
    CX = "cx"
    CCX = "ccx"
    SWAP = "swap"

    KRAUS = "kraus"


class Gate:
    """
    Base class for gates

    Attributes:
        name (GateTypes): name of the gate
        matrix (np.ndarray): matrix representation of the gate
        num_qubits (int): number of qubits the gate acts on
    """

    def __init__(self, name: GateTypes, matrix: np.ndarray, num_qubits: int) -> None:
        self._name = name
        self._matrix = matrix
        self._num_qubits = num_qubits

    @property
    def name(self) -> GateTypes:
        """return the name of the gate

        Returns:
            GateTypes: name of the gate
        """
        return self._name

    @property
    def matrix(self) -> np.ndarray:
        """return the matrix representation of the gate

        Returns:
            np.ndarray: matrix representation of the gate
        """
        return self._matrix

    @property
    def num_qubits(self) -> int:
        """return the number of qubits the gate acts on

        Returns:
            int: number of qubits the gate acts on
        """
        return self._num_qubits


class IGate(Gate):
    """
    Identity gate
    """

    def __init__(self) -> None:
        super().__init__(
            name=GateTypes.I, matrix=np.array([[1, 0], [0, 1]]), num_qubits=1
        )


class XGate(Gate):
    """
    Pauli-X gate
    """

    def __init__(self) -> None:
        super().__init__(
            name=GateTypes.X, matrix=np.array([[0, 1], [1, 0]]), num_qubits=1
        )


class YGate(Gate):
    """
    Pauli-Y gate
    """

    def __init__(self) -> None:
        super().__init__(
            name=GateTypes.Y, matrix=np.array([[0, -1j], [1j, 0]]), num_qubits=1
        )


class ZGate(Gate):
    """
    Pauli-Z gate
    """

    def __init__(self) -> None:
        super().__init__(
            name=GateTypes.Z, matrix=np.array([[1, 0], [0, -1]]), num_qubits=1
        )


class HGate(Gate):
    """
    Hadamard gate
    """

    def __init__(self) -> None:
        super().__init__(
            name=GateTypes.H,
            matrix=np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            num_qubits=1,
        )


class SGate(Gate):
    """
    S gate
    """

    def __init__(self) -> None:
        super().__init__(
            name=GateTypes.S, matrix=np.array([[1, 0], [0, 1j]]), num_qubits=1
        )


class TGate(Gate):
    """
    T gate
    """

    def __init__(self) -> None:
        super().__init__(
            name=GateTypes.T,
            matrix=np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
            num_qubits=1,
        )


class PhaseGate(Gate):
    """
    Phase gate
    """

    def __init__(self, phi: float) -> None:
        super().__init__(
            name=GateTypes.PHASE,
            matrix=np.array([[1, 0], [0, np.exp(1j * phi)]]),
            num_qubits=1,
        )
        self.phi = phi


class RXGate(Gate):
    """
    RX gate
    """

    def __init__(self, theta: float) -> None:
        super().__init__(
            name=GateTypes.RX,
            matrix=np.array(
                [
                    [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                    [-1j * np.sin(theta / 2), np.cos(theta / 2)],
                ]
            ),
            num_qubits=1,
        )
        self.theta = theta


class RYGate(Gate):
    """
    RY gate
    """

    def __init__(self, theta: float) -> None:
        super().__init__(
            name=GateTypes.RY,
            matrix=np.array(
                [
                    [np.cos(theta / 2), -np.sin(theta / 2)],
                    [np.sin(theta / 2), np.cos(theta / 2)],
                ]
            ),
            num_qubits=1,
        )
        self.theta = theta


class RZGate(Gate):
    """
    RZ gate
    """

    def __init__(self, theta: float) -> None:
        super().__init__(
            name=GateTypes.RZ,
            matrix=np.array(
                [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]]
            ),
            num_qubits=1,
        )
        self.theta = theta


class CXGate(Gate):
    """
    CNOT gate
    """

    def __init__(self) -> None:
        super().__init__(
            name=GateTypes.CX,
            matrix=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
            num_qubits=2,
        )


class CCXGate(Gate):
    """
    Toffoli gate
    """

    def __init__(self) -> None:
        super().__init__(
            name=GateTypes.CCX,
            matrix=np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                ]
            ),
            num_qubits=3,
        )


class SwapGate(Gate):
    """
    Swap gate
    """

    def __init__(self) -> None:
        super().__init__(
            name=GateTypes.SWAP,
            matrix=np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
            num_qubits=2,
        )


GATETYPES_TO_GATE: dict[GateTypes, Any] = {
    GateTypes.I: IGate,
    GateTypes.X: XGate,
    GateTypes.Y: YGate,
    GateTypes.Z: ZGate,
    GateTypes.H: HGate,
    GateTypes.S: SGate,
    GateTypes.T: TGate,
    GateTypes.PHASE: PhaseGate,
    GateTypes.RX: RXGate,
    GateTypes.RY: RYGate,
    GateTypes.RZ: RZGate,
    GateTypes.CX: CXGate,
    GateTypes.CCX: CCXGate,
    GateTypes.SWAP: SwapGate,
}
