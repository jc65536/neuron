import numpy as np
from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

# Generate the permutation info required by the kernel
def get_perm(n, gate_idcs):
    perm = list(range(n))
    locs = list(range(n))
    for i, idx in enumerate(gate_idcs):
        # if perm[i] != idx:
        #     print(f'{i=} {idx=}')
        #     perm[i], perm[idx] = perm[idx], perm[i]
        if perm[i] != idx:
            # Swap idx elem in perm to i in perm
            loc_idx = locs[idx]
            perm[i], perm[loc_idx] = perm[loc_idx], perm[i]
            locs[perm[i]] = i
            locs[perm[loc_idx]] = loc_idx

    movements = []
    for i, idx in enumerate(perm):
        if i != idx:
            # movements.append([idx, i])
            movements.append([i, idx])
    if len(movements) == 0:
        return np.array([[0, 0, 1, 1]], dtype=np.uint32)

    movements_arr = np.array(movements, dtype=np.uint32)
    return np.concatenate((movements_arr,
                           np.left_shift(1, movements_arr)),
                          axis=1)

n = 8
N = 2 ** n
gate_size = 2 # must be at most 7
unitary_size = 2 ** gate_size
ntiles = 2 ** (n - gate_size)

# initial_state: numpy array state vector with 2^n elements
# gates: list of tuples
#            - first element of tuple is an numpy unitary matrix up to size 2^7
#            - second element of tuple is idx permutation info
#              perm: list of src/dst qubit index pairs, shape = (n, 4)
#                    for efficiency, each row has the following numbers:
#                     - src
#                     - idx
#                     - 2^src
#                     - 2^idx
# returns: final state vector
# TODO: orient matrix/vector to to optimize tensor unit?
@nki.jit(mode='simulation')
def run_circuit(initial_state, gates):
    # for now, don't do any permutation
    # i.e. assume the unitary applies to the first 7 qubits
    # also assume only real numbers
    assert initial_state.shape == (N, 1)
    state = nl.ndarray(initial_state.shape, dtype=initial_state.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=state, src=initial_state)
    
    tile_idcs_base_range = nl.arange(unitary_size)[:, None]
    tile_idcs_base = nisa.iota(tile_idcs_base_range, dtype=np.uint32)

    for gate in gates:
        assert type(gate) == tuple and len(gate) == 2
        assert gate[0].shape == (unitary_size, unitary_size)
        assert gate[1].shape[1] == 4
        U_tile = nl.load(gate[0])
        for tile_idx in nl.affine_range(ntiles): # allows for parallel computation
            offset = tile_idx * unitary_size
            #state_tile = nl.load(state[offset:offset+unitary_size, 0])

            tile_idcs = nl.add(tile_idcs_base, offset, dtype=np.uint32)
            # nl.device_print('tile_idcs', tile_idcs)

            y = nl.copy(tile_idcs) # permuted tile idcs
            for i in nl.sequential_range(gate[1].shape[0]):
                a  = nl.load(gate[1][i][0])
                b  = nl.load(gate[1][i][1])
                ma = nl.load(gate[1][i][2]) # 2^a
                mb = nl.load(gate[1][i][3]) # 2^b

                y[:] = nl.bitwise_and(y, nl.invert(mb))
                y[:] = nl.bitwise_or(y, nl.left_shift(nl.bitwise_and(tile_idcs, ma), b - a))
                y[:] = nl.bitwise_or(y, nl.right_shift(nl.bitwise_and(tile_idcs, ma), a - b))

            # nl.device_print('y', y)

            state_tile = nl.load(state[y])
            new_state_tile = nl.matmul(U_tile, state_tile)
            #nl.store(state[offset:offset+unitary_size, 0], value=new_state_tile)
            nl.store(state[y], value=new_state_tile)

    return state

def kron(*args) -> np.ndarray:
    mat = np.array([[1]], dtype=np.float16)
    for arg in args:
        mat = np.kron(mat, arg)
    return mat

I = np.identity(2, dtype=np.float16)
X = np.array([[0, 1], [1, 0]], dtype=np.float16)
H = np.array([[1, 1], [1, -1]], dtype=np.float16) / np.sqrt(2)
CX = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
], dtype=np.float16)

def main():
    initial_state = np.zeros((N, 1), dtype=np.float16)
    initial_state[0, 0] = 1
    # initial_state = np.array([list(range(N))], dtype=np.float16).T

    gates = [(CX, [1, 2])]
    gates = [(kron(I, X), [0, 2])]

    # Max cat state
    gates = [(kron(I, H), [0, 1])]
    for i in range(1, n):
        gates.append((CX, [0, i]))

    for i, gate in enumerate(gates):
        gates[i] = (gate[0], get_perm(n, gate[1]))
    print(gates)
    
    new_state = run_circuit(initial_state, gates)
    print(new_state)

def main_cat():
    assert gate_size == 2

    gates = [(np.kron(I, H), [0, 1])]
    for i in range(1, n):
        gates.append((CX, [0, i]))

    for i, gate in enumerate(gates):
        gates[i] = (gate[0], get_perm(n, gate[1]))

    initial_state = np.zeros((N, 1), dtype=np.float16)
    initial_state[0, 0] = 1

    new_state = run_circuit(initial_state, gates)
    print(new_state)


if __name__ == '__main__':
    main_cat()
