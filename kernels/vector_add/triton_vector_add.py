import os
import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr):
    """Kernel to add two vectors"""
    # pid is program ID, which is essentially the block ID
    pid = tl.program_id(axis=0) 
    # starting ptr for block
    block_start = pid * BLOCK_SIZE
    # compute offsets from starting ptr by adding a vector of length BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # create mask to act as a boundary gaurd
    mask = offsets < n_elements
    # load x and y elements at the relevant offsets, respecting the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # add the vectors
    output = x + y
    # store output
    tl.store(output_ptr + offsets, output, mask=mask)


# helper function to launch kernel
def vector_add(x: torch.Tensor, y: torch.Tensor):
    # pre-allocate memory for output vector
    output = torch.empty_like(x)

    n_elements = x.numel()
    # determine launch grid i.e. number of blocks, 
    # can be either Tuple[int] or Callable[meta] -> Tuple[int]
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # launch kernel, indexed with the grid. pass metaparameters as keyword arguments
    # tensors are converted to pointers by default
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # kernel will write results to output ptrs, which we can return
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = vector_add(x, y)
    
    @triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
    def benchmark(size, provider):
        x = torch.rand(size, device=DEVICE, dtype=torch.float32)
        y = torch.rand(size, device=DEVICE, dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: vector_add(x, y), quantiles=quantiles)
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(print_data=True, save_path=os.path.dirname(__file__))
