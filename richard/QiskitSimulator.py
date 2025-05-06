# modified ChatGPT 4o output

import random
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import *
from qiskit_aer import AerSimulator

class QiskitSimulator:
	"""
	A class to simulate a quantum circuit consisting of Haar-random unitary gates
	applied to subsets of qubits.
	"""

	def __init__(self, n_qubits, targets, gates):
		"""
		Initializes the simulator with a given circuit specification.

		Parameters:
			n_qubits (int): Total number of qubits in the circuit.
			targets (List[List[int]]): A list of length-T lists specifying the qubit indices
									   that each unitary gate acts on.
			gates (List[np.ndarray]): A list of unitary matrices, each of size 2^m × 2^m,
									  corresponding to the operations applied in each step.
		"""
		assert len(targets) == len(gates), "Targets and gates must have the same length"
		self.n_qubits = n_qubits
		self.targets = targets
		self.gates = gates
		self.qc = QuantumCircuit(n_qubits)

		for target, unitary in zip(self.targets, self.gates):
			m = len(target)
			assert unitary.shape == (2**m, 2**m), f"Gate must be of shape {(2**m, 2**m)}"
			gate = UnitaryGate(unitary)
			self.qc.append(gate, target)

		self.qc.save_statevector()

	def simulate(self, device="CPU"):
		"""
		Simulates the built quantum circuit using the statevector simulator.

		Returns:
			np.ndarray: The final statevector of the quantum circuit.
		"""
		simulator = AerSimulator(simulator="statevector", device=device)
		circ = transpile(self.qc, simulator)
		result = simulator.run(circ).result()
		statevector = result.get_statevector(circ)
		return statevector
