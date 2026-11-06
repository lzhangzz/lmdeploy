#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import tilelang
import tilelang.language as T
import tvm_ffi
from tilelang import tvm

DTYPES = [('f32', 'float32'), ('f16', 'float16'), ('bf16', 'bfloat16')]
WIDTHS = [33, 128, 4096]


def make_rmsnorm(width, dtype, symbol):
    rows = T.dynamic('rows')
    x_stride = T.dynamic('x_stride', 'int64')
    y_stride = T.dynamic('y_stride', 'int64')
    block_width = tilelang.math.next_power_of_2(width)

    @T.prim_func
    def rmsnorm(x: T.StridedTensor((rows, width), (x_stride, 1), dtype),
                weight: T.Tensor((width,), dtype),
                y: T.StridedTensor((rows, width), (y_stride, 1), dtype),
                eps: T.float32):
        with T.Kernel(rows, threads=128) as row:
            values = T.alloc_fragment((1, block_width), 'float32')
            squares = T.alloc_fragment((1, block_width), 'float32')
            sum_sq = T.alloc_fragment((1,), 'float32')
            for col in T.Parallel(block_width):
                values[0, col] = T.if_then_else(col < width, T.cast(x[row, col], 'float32'), 0.0)
                squares[0, col] = values[0, col] * values[0, col]
            T.reduce_sum(squares, sum_sq, dim=1)
            for col in T.Parallel(block_width):
                if col < width:
                    y[row, col] = T.cast(values[0, col] * T.rsqrt(sum_sq[0] / width + eps)
                                         * T.cast(weight[col], 'float32'), dtype)

    return rmsnorm.with_attr('global_symbol', symbol)


def main():
    parser = argparse.ArgumentParser(description='Export TileLang RMSNorm kernels for C++ dynamic loading')
    parser.add_argument('--build-dir', type=Path, default=Path('build/tilelang_rmsnorm'),
                        help='Kernel artifact directory')
    parser.add_argument('--gpu-arch', default='sm_120')
    parser.add_argument('--host-compiler', help='NVCC host compiler used for kernel export')
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which(args.host_compiler) if args.host_compiler else shutil.which('g++-15') or shutil.which('g++')
    if compiler is None:
        parser.error('NVCC host compiler was not found; supply --host-compiler')
    tilelang.disable_cache()
    print(f'TileLang {tilelang.__version__}, TVM FFI {tvm_ffi.__version__}, target {args.gpu_arch}', flush=True)
    print(f'NVCC host compiler: {compiler}', flush=True)
    target = tvm.target.Target(f'cuda -arch={args.gpu_arch}')
    for name, dtype in DTYPES:
        for width in WIDTHS:
            symbol = f'rmsnorm_{name}_h{width}'
            kernel = tilelang.compile(make_rmsnorm(width, dtype, symbol), out_idx=[],
                                      execution_backend='tvm_ffi', target=target, compile_flags=[f'-ccbin={compiler}'])
            kernel.export_library(str(build_dir / f'{symbol}.so'))
            kernel.export_sources(kernel_path=str(build_dir / f'{symbol}.cu'),
                                  host_path=str(build_dir / f'{symbol}_host.cc'))
            print(f'Exported {symbol}.so', flush=True)


if __name__ == '__main__':
    main()
