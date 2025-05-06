from QuantumCircuitSimulator import QuantumCircuitSimulator
from QiskitSimulator import QiskitSimulator
from gates import *

import random
import torch
from qiskit.quantum_info import random_unitary as qiskit_random_unitary

from timeit import default_timer as timer

def random_unitary(n, dtype=torch.complex64):
	return torch.tensor(qiskit_random_unitary(2**n).data, dtype=dtype)

if __name__ == "__main__":
	use_gpu = True
	device = use_gpu and "cuda" or "cpu"

	check_with_qiskit = False

	n = 24
	m = 7
	steps = 100

	# define circuit
	print("defining circuit")
	targets = [random.sample(range(n), k=m) for i in range(steps)]
	gates = [random_unitary(m, dtype=torch.complex128) for _ in range(steps)]

	# PyTorch implementation
	print("starting pytorch implementation")
	start = timer()
	with torch.no_grad():
		state = torch.zeros([2]*n, dtype=torch.complex128).to(device)
		state[tuple([0]*n)] = 1
		qc = QuantumCircuitSimulator(n, targets, gates).to(device)
		result = qc(state, print_state=False).flatten()
	end = timer()
	pytorch_time = end - start

	# use Qiskit as reference implementation
	if check_with_qiskit:
		print("starting qiskit implementation")
		start = timer()
		simulator = QiskitSimulator(n, targets, gates)
		end = timer()
		qiskit_build_time = end - start

		start = timer()
		reference_result = simulator.simulate(device=(use_gpu and "GPU" or "CPU"))
		end = timer()
		qiskit_time = end - start

	# compare results
	# print(result)
	# print(reference_result)
	print("pytorch:", pytorch_time)

	if check_with_qiskit:
		print("qiskit build:", qiskit_build_time)
		print("qiskit:", qiskit_time)
		print(torch.linalg.norm(result - torch.tensor(reference_result.data, dtype=torch.complex128).to(device)))
