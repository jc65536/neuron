import qiskit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import *

import torch

DTYPE=torch.complex64

X = torch.tensor(Operator(XGate()).data, dtype=DTYPE)
Y = torch.tensor(Operator(YGate()).data, dtype=DTYPE)
Z = torch.tensor(Operator(ZGate()).data, dtype=DTYPE)
H = torch.tensor(Operator(HGate()).data, dtype=DTYPE)
I = torch.tensor(Operator(IGate()).data, dtype=DTYPE)
S = torch.tensor(Operator(SGate()).data, dtype=DTYPE)
T = torch.tensor(Operator(TGate()).data, dtype=DTYPE)

if __name__ == "__main__":
	print(X)
