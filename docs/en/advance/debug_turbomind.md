# How to debug Turbomind

Turbomind is implemented in C++, which is not as easy to debug as Python. This document provides basic methods for debugging Turbomind.

## Prerequisite

First, complete the local compilation according to the commands in [Install from source](../get_started/installation.md).

## Configure Python debug environment

Turbomind is loaded into the process of a Python interpreter. Without that interpreter's debug symbols, every frame inside CPython shows up in `gdb` as a bare address, which makes it hard to tell whether a hang or a crash originates in Python, in the extension, or in a kernel.

Install `gdb` and the symbols through your distribution's package manager. The exact package names differ, and the symbols must match the interpreter that actually runs lmdeploy:

- **Debian / Ubuntu**: `sudo apt install gdb python3-dbg libc6-dbg`
- **RHEL / Fedora / CentOS Stream / Rocky / Alma**: enable the matching debuginfo repository (e.g. `sudo dnf install dnf-plugins-core && sudo dnf debuginfo-install python3 glibc`), which resolves the exact builds that are installed
- **Arch**: install `gdb`, then the `python` and `glibc` debug packages from the `debug` repository

Note that the distribution's `python3-dbg` does not cover an interpreter installed some other way, such as a conda environment. If lmdeploy runs under such an interpreter, either debug with the distribution's `python3` or accept a less informative stack.

A recent `gdb` (version 10 or later) is recommended, since it copes better with the C++20 symbols produced by the current build.

### Verification

```bash
gdb python3
```

The banner should not contain `(no debugging symbols found)`, and an interpreter function should resolve to a source location:

```
(gdb) info line PyEval_EvalFrameDefault
Line 1234 of "Python/ceval.c" starts at address 0x... and ends at 0x....
```

For background on debugging CPython itself, see [DebuggingWithGdb](https://wiki.python.org/moin/DebuggingWithGdb).

## Set up symbolic links

After setting up symbolic links, there is no need to install it locally with `pip` every time.

```bash
# Change directory to lmdeploy, e.g.
cd /workdir/lmdeploy

# Since it has been built in the build directory
# Link the lib directory
cd lmdeploy && ln -s ../build/lib . && cd ..
# (Optional) Link compile_commands.json for clangd index
ln -s build/compile_commands.json .
```

## Start debugging

````bash
# Use gdb to start the API server with Llama-2-13b-chat-hf, e.g.
gdb --args python3 -m lmdeploy serve api_server /workdir/Llama-2-13b-chat-hf

# Set directories in gdb
Reading symbols from python3...
(gdb) set directories /workdir/lmdeploy

# Set a breakpoint using the relative path, e.g.
(gdb) b src/turbomind/models/language_model.cc:175

# Or better, break on a function name so the breakpoint survives edits
# that shift line numbers
(gdb) b LanguageModel::Impl::PostEmbedding

# When it shows
# ```
# No source file named src/turbomind/models/language_model.cc.
# Make breakpoint pending on future shared library load? (y or [n])
# ```
# Just type `y` and press enter

# Run
(gdb) r

# (Optional) Use https://github.com/InternLM/lmdeploy/blob/main/benchmark/profile_restful_api.py to send a request

python3 profile_restful_api.py --backend lmdeploy --dataset-path /workdir/ShareGPT_V3_unfiltered_cleaned_split.json --num_prompts 1
````

## Using GDB

Refer to [GDB Execution Commands](https://lldb.llvm.org/use/map.html) and happy debugging.
