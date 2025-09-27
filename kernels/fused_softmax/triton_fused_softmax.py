import os
import torch
import triton
import triton.language as tl

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Naive (numerically stable) softmax implementation in PyTorch without any kernel fusion"""
    # get max. for each row: read MN elements, write M elements
    x_max = x.max(dim=-1)[0]
    # subtract max. from each row for numerical stability: read MN + M elements, write MN elements
    x_stable = x - x_max[:, None]
    # compute numerator: read MN elements, write MN elements
    numerator = torch.exp(x_stable)
    # compute denominator: read MN elements, write M elements
    denominator = numerator.sum(dim=-1)
    # compute softmax: read MN + M elements, write MN elements
    softmax = numerator / denominator[:, None]
    # total reads: 5MN + 2M; total writes: 3MN + 2M
    return softmax


@triton.jit
def fused_softmax_kernel(input_ptr, 
                         output_ptr, 
                         input_row_stride, 
                         output_row_stride, 
                         n_rows, 
                         n_cols, 
                         BLOCK_SIZE: tl.constexpr, 
                         num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step  = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        input_start_ptr = input_ptr + row_idx * input_row_stride
        output_start_ptr = output_ptr + row_idx * output_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row = tl.load(input_start_ptr + col_offsets, mask=mask, other=float("-inf"))
        row_minus_max = row - tl.max(row, axis=0)
        num = tl.exp(row_minus_max)
        den = tl.sum(num, axis=0)
        out = num / den
        tl.store(output_start_ptr + col_offsets, out, mask=mask)


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """Helper function to launch fused softmax kernel"""
    n_rows, n_cols = x.shape
    # Triton requires BLOCK_SIZE to be a power of 2 (also better for GPU operations)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # set number of warps (parallelization within a row)
    num_warps = 8
    # set number of software pipelining stages (pipeline parallelization)
    # load these many future items into memory while current item is being processed
    num_stages = 4 if MAX_SMEM>200_000 else 2 # 200 KB
    # pre-allocatote memory for output
    y = torch.empty_like(x)
    # run kernel warmup to determine expected memory usage per thread and program
    kernel = fused_softmax_kernel.warmup(
        x, y,
        x.stride(0), y.stride(0), 
        n_rows, n_cols, 
        BLOCK_SIZE=BLOCK_SIZE, 
        num_stages=num_stages, 
        grid=(1,))
    # get register and sram usage based on warmup
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    # determine maximum possible occupancy i.e. number of programs or blocks the GPU can run at once
    # register-bound or smem-bound
    occupancy = MAX_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, MAX_SMEM // size_smem)
    num_programs = occupancy * NUM_SMS
    # do not need more programs than rows (would be a waste)
    num_programs = min(num_programs, n_rows)
    # launch kernel
    fused_softmax_kernel[(num_programs, 1, 1)](x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages)
    return y


if __name__ == "__main__":
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    DEVICE_INDEX = torch.cuda.current_device()

    # get device properties
    device_props = triton.runtime.driver.active.utils.get_device_properties(DEVICE_INDEX)
    MAX_SMEM = device_props["max_shared_mem"]
    MAX_REGS = device_props["max_num_regs"]
    NUM_SMS  = device_props["multiprocessor_count"]
    WARP_SIZE = device_props["warpSize"]
    
    # unit test
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=DEVICE)
    y_triton = fused_softmax(x)
    y_torch  = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), "Test failed!"

    # benchmark
    @triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'naive_softmax'],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="fused-softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
    def benchmark(M, N, provider):
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        stream = getattr(torch, DEVICE.type).Stream()
        getattr(torch, DEVICE.type).set_stream(stream)
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
        if provider == 'triton':
            ms = triton.testing.do_bench(lambda: fused_softmax(x))
        if provider == 'naive_softmax':
            ms = triton.testing.do_bench(lambda: naive_softmax(x))
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(print_data=True, save_path=os.path.dirname(__file__))
