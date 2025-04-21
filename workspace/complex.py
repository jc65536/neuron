# Goal of this module: Following the example code from matmul.py, write a module
# for representing and operating on vectors and matrices over complex numbers
# using only float32 data type.

from typing import TypeVar
import time

import numpy as np
from numpy import float32 as R, complexfloating as C
from numpy.typing import NDArray

from neuronxcc import nki
import neuronxcc.nki.language as nl


# Scalar in C
#   Array of shape 2. Index 0 = real part, index 1 = imag part.
#
# Vector in C^N
#   Array of shape (N, 2)
#
# Matrix over C
#   Array of shape (M, N, 2)
#
# Matrix multiplication
#   A: (M, K, 2)
#   B: (K, N, 2)
#   C = AB: (M, N, 2)
#   where
#   C[i,j,0] = A[i,:,0] dot B[:,j,0] - A[i,:,1] dot B[:,j,1]
#   C[i,j,1] = A[i,:,0] dot B[:,j:1] + A[i,:,1] dot B[:,j,0]


T = TypeVar("T", bound=np.generic)


@nki.jit(mode="simulation")
def matmul_C_tiled(AT: NDArray[T], B: NDArray[T]) -> NDArray[T]:
    """
    NKI kernel to compute product of two matrices over C

    Parameters
    ----------
    AT: NDArray[T]
        Left matrix of shape (K, M, 2). Compared to a normal matmul, the first
        two axes are transposed, so the contraction axis comes first.

        Dimension constraints:
        - 128 divides K
        - 64 divides M

    B: NDArray[T]
        Right matrix of shape (K, N, 2)

        Dimension constraints:
        - 128 divides K
        - 256 divides N

    Returns
    -------
    The product AB of shape [M, N, 2]
    """

    K, M, P = AT.shape
    K_, N, P_ = B.shape

    assert P == 2, f"Third axis must have size 2, got {P} instead"
    assert P_ == 2, f"Third axis must have size 2, got {P} instead"
    assert K == K_, f"Contraction dimension mismatch: {K} != {K_}"

    # AT is broken up into tiles of shape (TILE_K, TILE_M)
    # B is broken up into tiles of shape (TILE_K, TILE_N)
    TILE_M = nl.tile_size.gemm_stationary_fmax // 2  # 64
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax // 2  # 256

    assert M % TILE_M == 0, f"M = {M} is not a multiple of {TILE_M}"
    assert K % TILE_K == 0, f"K = {K} is not a multiple of {TILE_K}"
    assert N % TILE_N == 0, f"N = {N} is not a multiple of {TILE_N}"

    result = nl.ndarray((M, N, 2), dtype=AT.dtype, buffer=nl.shared_hbm)

    for tile_i in nl.affine_range(M // TILE_M):
        for tile_j in nl.affine_range(N // TILE_N):
            res_psum = nl.zeros((TILE_M, TILE_N, 2), AT.dtype, buffer=nl.psum)
            i = tile_i * TILE_M
            j = tile_j * TILE_N

            for tile_k in nl.affine_range(K // TILE_K):
                AT_tile = nl.ndarray(
                    (TILE_K, TILE_M, 2),
                    dtype=AT.dtype,
                    buffer=nl.sbuf,
                )
                B_tile = nl.ndarray(
                    (TILE_K, TILE_N, 2),
                    dtype=B.dtype,
                    buffer=nl.sbuf,
                )

                k = tile_k * TILE_K

                AT_tile = nl.load(AT[k:k + TILE_K, i:i + TILE_M, :])
                B_tile = nl.load(B[k:k + TILE_K, j:j + TILE_N, :])

                # Real part
                res_psum[:, :, 0] += nl.matmul(
                    AT_tile[:, :, 0],
                    B_tile[:, :, 0],
                    transpose_x=True,
                )
                # Neuron will complain about loop dependency if we use -= here
                res_psum[:, :, 0] += -1 * nl.matmul(
                    AT_tile[:, :, 1],
                    B_tile[:, :, 1],
                    transpose_x=True,
                )

                # Imag part
                res_psum[:, :, 1] += nl.matmul(
                    AT_tile[:, :, 0],
                    B_tile[:, :, 1],
                    transpose_x=True,
                )
                res_psum[:, :, 1] += nl.matmul(
                    AT_tile[:, :, 1],
                    B_tile[:, :, 0],
                    transpose_x=True,
                )

            res_sbuf = nl.copy(res_psum, dtype=result.dtype)
            nl.store(result[i:i + TILE_M, j:j + TILE_N, :], value=res_sbuf)

    return result


def to_complex(A: NDArray[R]) -> NDArray[C]:
    """
    Convert an (M, N, 2) matrix of dtype floating into an (M, N) matrix of dtype
    complexfloating
    """
    _, _, P = A.shape
    assert P == 2, f"Third axis must have size 2, got {P} instead"
    return A[..., 0] + A[..., 1] * 1j


def main():
    A = np.random.rand(512, 512, 2).astype(R)
    B = np.random.rand(512, 1024, 2).astype(R)

    A_complex = to_complex(A)
    B_complex = to_complex(B)

    start = time.perf_counter()
    C_complex_true = A_complex @ B_complex
    end = time.perf_counter()

    print(f"Numpy time: {end - start:.3f} s")

    def check_match(nki_func):
        start = time.perf_counter()
        C: NDArray = nki_func(A.transpose(1, 0, 2), B)
        end = time.perf_counter()

        print(f"Time: {end - start:.3f} s")

        C_complex = to_complex(C)

        diff = C_complex_true - C_complex

        abs_error = abs(diff).max()
        print(f"Max abs error: {abs_error}")

        rel_error = abs(diff / C_complex_true).max()
        print(f"Max rel error: {rel_error}")

        allclose = np.allclose(C_complex_true, C_complex, atol=1e-4, rtol=1e-2)
        print(f"np.allclose: {allclose}")

    print("Checking correctness of matmul_C_tiled")
    check_match(matmul_C_tiled)


if __name__ == "__main__":
    main()
