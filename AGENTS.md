# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General guidelines

- Perfer "right design shape" over "minimal changes"
- DO NOT use worktree unless asked **explicitly**

## Documentation

- Do not re-flow or re-wrap Markdown/prose docs to match a line-length (column)
  limit. Preserve a file's existing line breaks; never reformat wrapping just to
  satisfy a width. Edit content, not wrapping.
- TurboMind async execution, scheduler, cache management, and module-level
  `BatchOp` changes must preserve the normative contract in
  `src/turbomind/engine/README.md`. Reference items in that document by
  `<section>.<leaf>`, for example `contracts.batchop-schedule` or
  `checklist.cache-memory`.

## Planning and design

When planning or designing something:

1. Include concrete code snippets instead of vague descriptions of the design.
2. In those snippets, expand anything new rather than hiding it behind placeholders,
   omitted helper calls, ellipses, or unexplained abstractions.

## Build

Build in the `build` folder:
- Configure (if not already configured): run `sh ../my_generate.sh` from the `build` folder
- Build: run `ninja` from the `build` folder (or specify individual targets)

## Using locally cached models

Query the model-server MCP tool (`list_models`) for models available locally. Models are stored on different cache directories. You need:

```python
import huggingface_hub.constants as hf_constants
hf_constants.HF_HUB_OFFLINE = 1
hf_constants.HF_HUB_CACHE = "cache_dir returned by `get_model_cache_path` tool"
```

Set these **before** loading models in lmdeploy.

Notice the following method will not work because the values are cached by HF hub on it's first import.
```
os.environ['HF_HUB_OFFLINE'] = 1
os.environ['HF_HUB_CACHE'] = '...'
```

## Testing

Verify TurboMind with `scripts/test_turbomind_model.py`

**You MUST verify the response every time you test a model.** The model must respond with meaningful human words relevant to your test prompt. Gibberish responses indicate a bug. Also the requested response length should be **at least 128 tokens** for testing a model.

- DO NOT batch the testing by wrapping the test script in bash for loop
- DO NOT modify the test script, the script MUST be used AS IS

## Debugging

Iterate until the bug is fixed:

```
while bugs:
    modify the code
    evaluate the outcome
```

Do not stop with active bugs.

When needed, use the model-server MCP tools (`get_model_config`, `get_weight_info`) to inspect model dimensions and weight shapes for debugging. Do not write code to obtain such information yourself.

## GPU usage

**Always** check `get_gpu_usage` for empty GPUs before running anything on GPU.

GPU commands (model tests, anything that touches CUDA) **must run outside of the sandbox**. The sandbox has no access to the NVIDIA driver, so these commands fail fast inside it. Run them with unrestricted (non-sandboxed) permissions.

When OOM is encountered during a model test, check if the GPU is occupied by another process.


## Hard constraints

We use an in-tree development style. Installing lmdeploy via setup scripts brings multiple `_turbomind` extension `.so` files into the workspace, which causes chaos. Therefore:

- NEVER install lmdeploy as a pip package
- NEVER run the `setup.py` script
- DO NOT even think about it
