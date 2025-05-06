import torch
import torch.nn as nn

from gates import *

class QuantumCircuitSimulator(nn.Module):
    def __init__(self, num_qubits, targets, gates):
        super().__init__()
        self.num_qubits = num_qubits
        self.targets = targets
        self.gates = nn.ParameterList(gates)

        self.dims = list(range(self.num_qubits))

    def forward(self, state, print_state = True):
        for step, target, gate in zip(range(len(self.targets)), self.targets, self.gates):
            # align with qiskit convention
            # i don't actually know why this works I just asked ChatGPT
            target_axes = [self.num_qubits - 1 - q for q in target]
            target_axes = list(reversed(target_axes))

            permutation = [i for i in self.dims if i not in target_axes] + target_axes
            inv_permutation = [0]*self.num_qubits
            for i, dim in enumerate(permutation):
                inv_permutation[dim] = i

            state = torch.permute(state, permutation)

            state = state.reshape((-1, 2**len(target)))
            state = state @ gate.T
            state = state.reshape([2]*self.num_qubits)

            state = torch.permute(state, inv_permutation)

            if print_state:
                print(step, state.flatten())

        return state
