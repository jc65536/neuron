# Example code from
# https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/getting_started.html

from neuronxcc import nki
import neuronxcc.nki.language as nl


@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    """NKI kernel to compute element-wise addition of two input tensors
    """

    # Check all input/output tensor shapes are the same for element-wise operation
    assert a_input.shape == b_input.shape

    # Check size of the first dimension does not exceed on-chip memory tile size limit,
    # so that we don't need to tile the input to keep this example simple
    assert a_input.shape[0] <= nl.tile_size.pmax

    # Load the inputs from device memory to on-chip memory
    a_tile = nl.load(a_input)
    b_tile = nl.load(b_input)

    # Specify the computation (in our case: a + b)
    c_tile = nl.add(a_tile, b_tile)

    # Create a HBM tensor as the kernel output
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

    # Store the result to c_output from on-chip memory to device memory
    nl.store(c_output, value=c_tile)

    # Return kernel output as function output
    return c_output


if __name__ == "__main__":
    import numpy as np

    a = np.ones((4, 3), dtype=np.float16)
    b = np.ones((4, 3), dtype=np.float16)

    # Run NKI kernel on a NeuronDevice
    # c = nki_tensor_add_kernel(a, b)

    # Simulate kernel
    c = nki.simulate_kernel(nki_tensor_add_kernel, a, b)

    print(c)
