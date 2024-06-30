"""
This module contains the Circuit class which represents a quantum circuit.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import gates
from .simulator_result import SimulatorResultTypes

if TYPE_CHECKING:
    import qiskit


@dataclass
class Instruction:
    """
    Class for circuit instructions

    Attributes:
        gate (gates.Gate): gate of the instruction
        qubits (list): qubits the gate acts on
    """

    gate: gates.Gate
    qubits: list


class Circuit:
    """
    Quantum circuit class

    Attributes:
        num_qubits (int): number of qubits in the circuit
        instructions (list[Instruction]): instructions of the circuit
    """

    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = num_qubits
        self._instructions: list[Instruction] = []
        self._saving_results: dict[int, dict] = {}

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits in the circuit

        Returns:
            int: number of qubits
        """
        return self._num_qubits

    @property
    def instructions(self) -> list[Instruction]:
        """Return the instructions of the circuit

        Returns:
            list[Instruction]: instructions of the circuit
        """
        return self._instructions

    @property
    def saving_results(self) -> dict[int, dict]:
        """Return the saving results of the circuit

        Returns:
            dict[int, dict]: saving results of the circuit
        """
        return self._saving_results

    def add_gate(self, gate: gates.Gate, qubits: list) -> None:
        """Add a gate to the circuit

        Args:
            gate (gates.Gate): gate to add
            qubits (list): qubits the gate acts on
        """
        if len(qubits) != gate.num_qubits:
            raise ValueError(
                f"Number of qubits {len(qubits)} does not match the gate {gate.num_qubits}."
            )
        self._instructions.append(Instruction(gate=gate, qubits=qubits))

    def save_result(self, result_type: SimulatorResultTypes) -> None:
        """Flag where to save the result

        Args:
            result (SimulatorResultTypes): result type
        """
        if result_type not in SimulatorResultTypes._value2member_map_:
            raise ValueError(
                f"Result type {result_type} is not valid. Use one of "
                f"{list(SimulatorResultTypes._value2member_map_.keys())}."
            )
        self._saving_results[len(self._instructions)] = {"result_type": result_type}

    def draw(self, *args, **kwargs) -> Any:
        """Draw the circuit using the qiskit library

        Args:
            *args: positional arguments for the draw method
            **kwargs: keyword arguments for the draw method
        """
        # pylint: disable=cyclic-import
        from .converters.qiskit import to_qiskit

        return to_qiskit(self).draw(*args, **kwargs)

    def to_qiskit(self) -> "qiskit.QuantumCircuit":
        """Convert the circuit to a qiskit circuit

        Returns:
            qiskit.QuantumCircuit: qiskit circuit
        """
        # pylint: disable=cyclic-import
        from .converters.qiskit import to_qiskit

        return to_qiskit(self)

    def x(self, qubit: int) -> None:
        """Add a Pauli-X gate to the circuit

        Args:
            qubit (int): qubit index
        """
        self.add_gate(gates.XGate(), [qubit])

    def y(self, qubit: int) -> None:
        """Add a Pauli-Y gate to the circuit

        Args:
            qubit (int): qubit index
        """
        self.add_gate(gates.YGate(), [qubit])

    def z(self, qubit: int) -> None:
        """Add a Pauli-Z gate to the circuit

        Args:
            qubit (int): qubit index
        """
        self.add_gate(gates.ZGate(), [qubit])

    def h(self, qubit: int) -> None:
        """Add a Hadamard gate to the circuit

        Args:
            qubit (int): qubit index
        """
        self.add_gate(gates.HGate(), [qubit])

    def s(self, qubit: int) -> None:
        """Add a S gate to the circuit

        Args:
            qubit (int): qubit index
        """
        self.add_gate(gates.SGate(), [qubit])

    def t(self, qubit: int) -> None:
        """Add a T gate to the circuit

        Args:
            qubit (int): qubit index
        """
        self.add_gate(gates.TGate(), [qubit])

    def phase(self, qubit: int, phi: float) -> None:
        """Add a phase gate to the circuit

        Args:
            qubit (int): qubit index
            phi (float): phase angle
        """
        self.add_gate(gates.PhaseGate(phi), [qubit])

    def rx(self, qubit: int, theta: float) -> None:
        """Add a RX gate to the circuit

        Args:
            qubit (int): qubit index
            theta (float): rotation angle
        """
        self.add_gate(gates.RXGate(theta), [qubit])

    def ry(self, qubit: int, theta: float) -> None:
        """Add a RY gate to the circuit

        Args:
            qubit (int): qubit index
            theta (float): rotation angle
        """
        self.add_gate(gates.RYGate(theta), [qubit])

    def rz(self, qubit: int, theta: float) -> None:
        """Add a RZ gate to the circuit

        Args:
            qubit (int): qubit index
            theta (float): rotation angle
        """
        self.add_gate(gates.RZGate(theta), [qubit])

    def cx(self, control_qubit: int, target_qubit: int) -> None:
        """Add a CNOT gate to the circuit

        Args:
            control_qubit (int): control qubit index
            target_qubit (int): target qubit index
        """
        self.add_gate(gates.CXGate(), [control_qubit, target_qubit])

    def ccx(self, control_qubit1: int, control_qubit2: int, target_qubit: int) -> None:
        """Add a Toffoli gate to the circuit

        Args:
            control_qubit1 (int): first control qubit index
            control_qubit2 (int): second control qubit index
            target_qubit (int): target qubit index
        """
        self.add_gate(gates.CCXGate(), [control_qubit1, control_qubit2, target_qubit])

    def swap(self, qubit1: int, qubit2: int) -> None:
        """Add a Swap gate to the circuit

        Args:
            qubit1 (int): first qubit index
            qubit2 (int): second qubit index
        """
        self.add_gate(gates.SwapGate(), [qubit1, qubit2])
