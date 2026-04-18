# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

Build in the `build` folder:
- Configure (if not already configured): run `sh ../my_generate.sh` from the `build` folder
- Build: run `ninja` from the `build` folder (or specify individual targets)

## Environment setup

For the lmdeploy Python code and `_turbomind` Python extension to be found by Python, set:

```
PYTHONPATH=${workspace_dir}/lmdeploy:${workspace_dir}/build/lib
```

Replace `${workspace_dir}` with the current project directory.

## Using locally cached models

Query the model-server MCP tool (`list_models`) for models available locally. Models are stored on different cache directories. You need:

```python
import huggingface_hub.constants as hf_constants
hf_constants.HF_HUB_OFFLINE = 1
hf_constants.HF_HUB_CACHE = "cache_dir returned by `get_model_cache_path` tool"
```

Set these **before** loading models in lmdeploy.

## Testing

See `build/toy_example.py` for sample test code.

**You MUST verify the response every time you test a model.** The model must respond with meaningful human words relevant to your test prompt. Gibberish responses indicate a bug.

## Debugging

Iterate until the bug is fixed:

```
while bugs:
    modify the code
    evaluate the outcome
```

Do not stop with active bugs.

When needed, use the model-server MCP tools (`get_model_config`, `get_checkpoint_info`) to inspect model dimensions and weight shapes for debugging. Do not write code to obtain such information yourself.

## GPU usage

**Always** check `get_gpu_usage` for empty GPUs before running anything on GPU.

When OOM is encountered during a model test, check if the GPU is occupied by another process.

## NVIDIA L20Y

**NVIDIA L20Y is a SM90 GPU (alias of the H800).** We know it for sure. `nvidia-smi` will wrongly report it as SM89.

Both `torch` and CUDA API `cudaDeviceGetAttribute` will report SM90 correctly.

## Hard constraints

We use an in-tree development style. Installing lmdeploy via setup scripts brings multiple `_turbomind` extension `.so` files into the workspace, which causes chaos. Therefore:

- NEVER install lmdeploy as a pip package
- NEVER run the `setup.py` script
- DON'T even think about it
