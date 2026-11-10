# Installation

LMDeploy is a python library for compressing, deploying, and serving Large Language Models(LLMs) and Vision-Language Models(VLMs).
Its inference engine, TurboMind, is developed in C++ and CUDA and strives for ultimate optimization of inference performance.

It supports LLMs and VLMs deployment on Linux. Furthermore, it is compatible with the following NVIDIA GPUs:

- Volta(sm70): V100
- Turing(sm75): 20 series, T4
- Ampere(sm80,sm86): 30 series, A10, A16, A30, A100
- Ada Lovelace(sm89): 40 series
- Hopper(sm90): H100, H200
- Blackwell(sm100, sm120): B200, 50 series

## Install with pip (Recommend)

It is recommended installing lmdeploy using pip in a conda environment (python 3.10 - 3.13):

```shell
conda create -n lmdeploy python=3.12 -y
conda activate lmdeploy
pip install lmdeploy
```

## Install from source

LMDeploy builds the TurboMind engine with NVIDIA CUDA support. A source installation always compiles the native extension. TurboMind requires C++20 for both host C++ and CUDA compilation, CMake 3.25.2 or newer, and CUDA Toolkit 12.0 or newer. Use a host compiler with C++20 `std::format` support that your CUDA Toolkit also supports, such as GCC 13 or newer.

Once the CUDA toolkit is successfully set up, you can build and install LMDeploy with a single command:

```shell
pip install git+https://github.com/InternLM/lmdeploy.git
```

If you prefer a specific version instead of the `main` branch of LMDeploy, you can specify it in your command:

```shell
pip install https://github.com/InternLM/lmdeploy/archive/refs/tags/v0.13.0.zip
```

LMDeploy only supports NVIDIA CUDA GPUs. Ascend, Cambricon, MACA and AMD ROCm devices are not supported.
