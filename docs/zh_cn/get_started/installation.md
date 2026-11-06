# 安装

LMDeploy 是一个用于大型语言模型（LLMs）和视觉-语言模型（VLMs）压缩、部署和服务的 Python 库。
其推理引擎 TurboMind 由 C++ 和 CUDA 开发，致力于推理性能的优化。

LMDeploy 支持在 Linux 和 Windows 平台上部署 LLMs 和 VLMs。此外，它还与以下 NVIDIA GPU 兼容：

Volta(sm70): V100
Turing(sm75): 20 系列，T4
Ampere(sm80,sm86): 30 系列，A10, A16, A30, A100
Ada Lovelace(sm89): 40 系列

## 使用 pip 安装（推荐）

我们推荐在一个干净的conda环境下（python3.10 - 3.13），安装 lmdeploy：

```shell
conda create -n lmdeploy python=3.12 -y
conda activate lmdeploy
pip install lmdeploy
```

## 从源码安装

LMDeploy 面向 NVIDIA CUDA 环境编译 TurboMind 引擎，从源码安装时总会编译原生扩展。TurboMind 的主机 C++ 和 CUDA 代码均要求使用 C++20，并需要 CMake 3.25.2 或更新版本以及 CUDA 工具包 12.0 或更新版本。主机编译器必须支持 C++20，且受所选 CUDA 工具包支持，例如 Linux 上的 GCC 10 或更新版本，或 Windows 上的 Visual Studio 2022。

成功安装 CUDA 工具包后，您可以使用以下单行命令构建并安装 LMDeploy：

```shell
pip install git+https://github.com/InternLM/lmdeploy.git
```

如果您希望使用特定版本，而不是 LMDeploy 的 `main` 分支，可以在命令行中指定：

```shell
pip install https://github.com/InternLM/lmdeploy/archive/refs/tags/v0.11.0.zip
```

LMDeploy 仅支持 NVIDIA CUDA GPU，不支持昇腾、寒武纪、沐曦以及 AMD ROCm 设备。
