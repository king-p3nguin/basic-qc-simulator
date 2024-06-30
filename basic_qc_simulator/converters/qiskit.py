"""
Module for conversion between basic_qc_simulator and qiskit circuits
"""

from typing import TYPE_CHECKING

from .. import gates
from ..circuit import Circuit
from ..simulator_result import SimulatorResultTypes
from ..utils import check_module_installed

if TYPE_CHECKING:
    import qiskit


def to_qiskit(circuit: Circuit) -> "qiskit.QuantumCircuit":
    """
    Convert a basic_qc_simulator circuit to a qiskit circuit

    Args:
        circuit (Circuit): circuit to convert

    Returns:
        qiskit.QuantumCircuit: qiskit circuit
    """
    check_module_installed("qiskit")
    import qiskit

    qc = qiskit.QuantumCircuit(circuit.num_qubits)
    for idx, instruction in enumerate(circuit.instructions):
        if circuit._saving_results.get(idx) is not None:
            result_type = circuit._saving_results.get(idx)["result_type"]  # type: ignore
            _saving_results_in_qiskit(qc, result_type)
        gate = instruction.gate
        if isinstance(gate, gates.XGate):
            qc.x(instruction.qubits[0])
        elif isinstance(gate, gates.YGate):
            qc.y(instruction.qubits[0])
        elif isinstance(gate, gates.ZGate):
            qc.z(instruction.qubits[0])
        elif isinstance(gate, gates.HGate):
            qc.h(instruction.qubits[0])
        elif isinstance(gate, gates.SGate):
            qc.s(instruction.qubits[0])
        elif isinstance(gate, gates.TGate):
            qc.t(instruction.qubits[0])
        elif isinstance(gate, gates.PhaseGate):
            qc.p(gate.phi, instruction.qubits[0])
        elif isinstance(gate, gates.RXGate):
            qc.rx(gate.theta, instruction.qubits[0])
        elif isinstance(gate, gates.RYGate):
            qc.ry(gate.theta, instruction.qubits[0])
        elif isinstance(gate, gates.RZGate):
            qc.rz(gate.theta, instruction.qubits[0])
        elif isinstance(gate, gates.CXGate):
            qc.cx(instruction.qubits[0], instruction.qubits[1])
        elif isinstance(gate, gates.CCXGate):
            qc.ccx(instruction.qubits[0], instruction.qubits[1], instruction.qubits[2])
        elif isinstance(gate, gates.SwapGate):
            qc.swap(instruction.qubits[0], instruction.qubits[1])

    if circuit._saving_results.get(len(circuit.instructions)) is not None:
        result_type = circuit._saving_results.get(len(circuit.instructions))[
            "result_type"
        ]  # type: ignore
        _saving_results_in_qiskit(qc, result_type)

    return qc


def _saving_results_in_qiskit(
    circuit: "qiskit.QuantumCircuit", result_type: SimulatorResultTypes
) -> None:
    """
    Check if the circuit has a result to save and save it if needed

    Args:
        circuit (qiskit.QuantumCircuit): qiskit circuit
        result_type (SimulatorResultTypes): result type to save
    """
    # import qiskit_aer here to add additional methods to qiskit.QuantumCircuit
    import qiskit_aer  # pylint: disable=unused-import

    if result_type == SimulatorResultTypes.STATE_VECTOR:
        circuit.save_statevector()
    else:
        raise ValueError(
            f"Qiskit converter does not support saving {result_type} result."
        )
