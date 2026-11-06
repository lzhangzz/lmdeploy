#!/usr/bin/env python3
import argparse
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import make_fake_tensor


@cute.kernel
def rmsnorm_kernel(x: cute.Tensor, weight: cute.Tensor, y: cute.Tensor, eps: cutlass.Float32):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    width = x.shape[1]
    sum_sq = cutlass.Float32(0.0)
    for col in range(lane, width, 32):
        value = x[row, col].to(cutlass.Float32)
        sum_sq += value * value
    sum_sq = cute.arch.warp_reduction_sum(sum_sq)
    inv_rms = cute.math.rsqrt(sum_sq / cutlass.Float32(width) + eps)
    for col in range(lane, width, 32):
        value = x[row, col].to(cutlass.Float32)
        scale = weight[col].to(cutlass.Float32)
        y[row, col] = (value * inv_rms * scale).to(y.element_type)


@cute.jit
def rmsnorm(x: cute.Tensor, weight: cute.Tensor, y: cute.Tensor, eps: cutlass.Float32, stream: cuda.CUstream):
    rmsnorm_kernel(x, weight, y, eps).launch(grid=(x.shape[0], 1, 1), block=(32, 1, 1), stream=stream)


def main():
    parser = argparse.ArgumentParser(description='Export CuTe DSL RMSNorm kernels for C++ dynamic loading')
    parser.add_argument('--build-dir', type=Path, default=Path('build/cute_dsl_rmsnorm'),
                        help='Kernel artifact directory')
    parser.add_argument('--gpu-arch', default='sm_120')
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    print(f'CuTe DSL {cutlass.__version__}, CUDA {cutlass.CUDA_VERSION}, target {args.gpu_arch}', flush=True)
    for name, dtype in [('f32', cutlass.Float32), ('f16', cutlass.Float16), ('bf16', cutlass.BFloat16)]:
        x = make_fake_tensor(dtype, (cute.sym_int32(), cute.sym_int32()), (cute.sym_int64(), 1), assumed_align=16)
        weight = make_fake_tensor(dtype, (cute.sym_int32(),), (1,), assumed_align=16)
        y = make_fake_tensor(dtype, (cute.sym_int32(), cute.sym_int32()), (cute.sym_int64(), 1), assumed_align=16)
        compiled = cute.compile(rmsnorm, x, weight, y, cutlass.Float32(1e-6), cuda.CUstream(0),
                                options=f'--gpu-arch {args.gpu_arch}')
        compiled.export_to_c(file_path=str(build_dir), file_name=f'rmsnorm_{name}', function_prefix=f'rmsnorm_{name}')
        print(f'Exported rmsnorm_{name}.o and .h', flush=True)


if __name__ == '__main__':
    main()
