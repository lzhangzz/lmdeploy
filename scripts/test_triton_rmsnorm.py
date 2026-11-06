#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import triton
import triton.language as tl
from triton.tools.compile import CompileArgs, compile_kernel

DTYPES = [('f32', 'fp32'), ('f16', 'fp16'), ('bf16', 'bf16')]
BLOCK_SIZE = 8192


@triton.jit
def rmsnorm_kernel(x, weight, y, rows, cols, x_stride, y_stride, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    values = tl.load(x + row * x_stride + col, mask=col < cols, other=0).to(tl.float32)
    scales = tl.load(weight + col, mask=col < cols, other=0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(values * values, axis=0) / cols + eps.to(tl.float32))
    tl.store(y + row * y_stride + col, values * inv_rms * scales, mask=col < cols)


def main():
    parser = argparse.ArgumentParser(description='Export Triton RMSNorm kernels for C++ dynamic loading')
    parser.add_argument('--build-dir', type=Path, default=Path('build/triton_rmsnorm'),
                        help='Kernel artifact directory')
    parser.add_argument('--epsilon-type', choices=['fp32', 'fp64'], default='fp64',
                        help='Device epsilon ABI; fp32 reproduces the Triton 3.7.0 AOT scalar mismatch')
    args = parser.parse_args()
    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TRITON_CACHE_DIR', str(build_dir / 'cache'))

    target = triton.runtime.driver.active.get_current_target()
    if target.backend != 'cuda':
        parser.error('This verification requires a CUDA device')
    print(f'Triton {triton.__version__}, target {target}, epsilon ABI {args.epsilon_type}', flush=True)

    symbols = []
    sources = []
    source_path = Path(__file__).resolve()
    for name, dtype in DTYPES:
        signature = (f'*{dtype}:16, *{dtype}:16, *{dtype}:16, i32, i32, i64, i64, '
                     f'{args.epsilon_type}, {BLOCK_SIZE}')
        symbol, files = compile_kernel(CompileArgs(path=str(source_path), kernel_name='rmsnorm_kernel',
                                                 signature=signature, grid='rows,1,1', num_warps=4, num_stages=1,
                                                 out_name=f'rmsnorm_{name}', out_path=build_dir / f'rmsnorm_{name}'))
        symbols.append(f'#define RMSNORM_{name.upper()}_SYMBOL "{symbol}"')
        sources.extend(file.name for file in files if file.suffix == '.c')
        print(f'Exported {symbol}', flush=True)
    (build_dir / 'rmsnorm_symbols.h').write_text('#pragma once\n' + '\n'.join(symbols) + '\n')
    cmake_sources = '\n'.join(f'    "{source}"' for source in sources)
    (build_dir / 'rmsnorm_sources.cmake').write_text(f'set(TRITON_RMSNORM_SOURCES\n{cmake_sources}\n)\n')


if __name__ == '__main__':
    main()
