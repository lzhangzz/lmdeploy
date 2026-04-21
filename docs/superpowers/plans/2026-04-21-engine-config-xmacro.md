# Engine Config X-Macro Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the YAML string transport for engine config with an x-macro struct passed directly through pybind, eliminating ~50 lines of YAML parsing and the yaml-cpp dependency.

**Architecture:** New `EngineConfig` x-macro struct (standalone, like `RopeConfig`) is auto-bound to Python via `bind_struct`. Python constructs it from the existing `TurbomindEngineConfig` dataclass and passes it directly to `TurboMind.create`. C++ reads fields directly instead of parsing YAML.

**Tech Stack:** C++ x-macros, pybind11, Python dataclasses

---

### Task 1: Create EngineConfig x-macro struct

**Files:**
- Create: `src/turbomind/engine/engine_config.h`

- [ ] **Step 1: Create the header file**

```cpp
// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <string>
#include <vector>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/module.h"  // TM_MEMBER, TM_FOR_EACH

namespace turbomind {

struct EngineConfig {
    #define ENGINE_FIELDS(X)                          \
        X(DataType,       data_type)                  \
        X(int,            cache_block_seq_len, 0)     \
        X(int,            quant_policy, 0)            \
        X(int,            tune_layer_num, 1)          \
        X(int,            max_batch_size, 0)          \
        X(int,            max_prefill_token_num, 0)   \
        X(int,            max_context_token_num, 0)   \
        X(int,            session_len, 0)             \
        X(float,          cache_max_block_count, 0)   \
        X(int,            cache_chunk_size, 0)        \
        X(bool,           enable_prefix_caching, false)\
        X(bool,           enable_metrics, false)      \
        X(int,            num_tokens_per_iter, 0)     \
        X(int,            max_prefill_iters, 1)       \
        X(int,            async_, 0)                  \
        X(int,            outer_dp_size)              \
        X(int,            attn_dp_size)               \
        X(int,            attn_tp_size)               \
        X(int,            attn_cp_size)               \
        X(int,            mlp_tp_size)                \
        X(std::vector<int>, devices)                  \
        X(int,            nnodes)                     \
        X(int,            node_rank)                  \
        X(std::string,    communicator)

    ENGINE_FIELDS(TM_MEMBER)
    TM_FOR_EACH(EngineConfig, ENGINE_FIELDS)

    #undef ENGINE_FIELDS
};

}  // namespace turbomind
```

- [ ] **Step 2: Verify it compiles**

Run: `cd /data/lmdeploy-modeling/build && ninja turbomind 2>&1 | tail -20`
Expected: Compile succeeds (the header is not yet included anywhere, so this just validates syntax).

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/engine/engine_config.h
git commit -m "feat: add EngineConfig x-macro struct"
```

---

### Task 2: Add EngineConfig to pybind bindings

**Files:**
- Modify: `src/turbomind/python/bind.cpp:317-324` (add `bind_struct` call)
- Modify: `src/turbomind/python/bind.cpp:1-20` (add include)

- [ ] **Step 1: Add include for EngineConfig**

In `src/turbomind/python/bind.cpp`, add after the existing includes (e.g., after line 20):

```cpp
#include "src/turbomind/engine/engine_config.h"
```

- [ ] **Step 2: Add bind_struct call for EngineConfig**

In `src/turbomind/python/bind.cpp`, after the existing `bind_struct<turbomind::core::RopeConfig>(m, "RopeConfig");` line (around line 455), add:

```cpp
    bind_struct<turbomind::EngineConfig>(m, "EngineConfig");
```

- [ ] **Step 3: Build and verify**

Run: `cd /data/lmdeploy-modeling/build && ninja _turbomind 2>&1 | tail -20`
Expected: Build succeeds.

- [ ] **Step 4: Verify Python can import and use EngineConfig**

Run:
```bash
cd /data/lmdeploy-modeling && python -c "
import _turbomind as _tm
ec = _tm.EngineConfig()
ec.data_type = _tm.DataType.TYPE_BF16
ec.max_batch_size = 32
ec.session_len = 4096
ec.devices = [0]
ec.nnodes = 1
ec.node_rank = 0
ec.communicator = 'nccl'
ec.attn_tp_size = 1
ec.attn_dp_size = 1
ec.attn_cp_size = 1
ec.mlp_tp_size = 1
ec.outer_dp_size = 1
print(f'data_type={ec.data_type}, max_batch_size={ec.max_batch_size}, devices={ec.devices}')
print('EngineConfig binding works!')
"
```
Expected: `data_type=14, max_batch_size=32, devices=[0]` and "EngineConfig binding works!" (14 is the encoded value of TYPE_BF16).

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/python/bind.cpp
git commit -m "feat: bind EngineConfig struct to Python"
```

---

### Task 3: Update TurboMind C++ to accept EngineConfig

**Files:**
- Modify: `src/turbomind/turbomind.h:23` (constructor signature)
- Modify: `src/turbomind/turbomind.cc:40-65` (remove `data_type_from_string`)
- Modify: `src/turbomind/turbomind.cc:99` (Impl constructor signature)
- Modify: `src/turbomind/turbomind.cc:178-260` (replace YAML parsing with direct field access)
- Modify: `src/turbomind/turbomind.cc:510-513` (outer constructor signature)
- Modify: `src/turbomind/turbomind.cc:29` (remove yaml-cpp include)
- Modify: `src/turbomind/CMakeLists.txt:39` (remove yaml-cpp dependency)

- [ ] **Step 1: Update turbomind.h constructor signature**

Change line 23 from:
```cpp
TurboMind(std::string model_dir, std::string config, FFICtxFactory ffi_ctx_factory);
```
to:
```cpp
TurboMind(std::string model_dir, EngineConfig config, FFICtxFactory ffi_ctx_factory);
```

Add include after line 10:
```cpp
#include "src/turbomind/engine/engine_config.h"
```

- [ ] **Step 2: Remove yaml-cpp include and data_type_from_string from turbomind.cc**

Remove line 29:
```cpp
#include <yaml-cpp/yaml.h>
```

Remove the static function `data_type_from_string` (lines 40-65).

Add include for engine_config.h (if not already transitively included):
```cpp
#include "src/turbomind/engine/engine_config.h"
```

- [ ] **Step 3: Update Impl constructor declaration**

Change line 99 from:
```cpp
Impl(string model_dir, string config, FFICtxFactory ffi_ctx_factory);
```
to:
```cpp
Impl(string model_dir, EngineConfig config, FFICtxFactory ffi_ctx_factory);
```

- [ ] **Step 4: Replace YAML parsing in Impl constructor body**

Replace lines 178-260 (the full constructor body) with:

```cpp
TurboMind::Impl::Impl(string model_dir, EngineConfig config, FFICtxFactory ffi_ctx_factory):
    data_type_{}, engine_param_{}, ffi_ctx_factory_{ffi_ctx_factory}
{
    data_type_ = config.data_type;
    TM_CHECK(data_type_ == kBfloat16 || data_type_ == kHalf);

    engine_param_.cache_block_seq_len = config.cache_block_seq_len;
    engine_param_.quant_policy        = config.quant_policy;
    engine_param_.tune_layer_num      = config.tune_layer_num;

    engine_param_.max_batch_size = config.max_batch_size;
    auto max_forward_token_num   = config.max_prefill_token_num;
    max_forward_token_num       += engine_param_.max_batch_size;

    engine_param_.max_context_token_num = config.max_context_token_num;
    engine_param_.session_len           = config.session_len;

    engine_param_.cache_max_block_count = config.cache_max_block_count;
    engine_param_.cache_chunk_size      = config.cache_chunk_size;
    engine_param_.enable_prefix_caching = config.enable_prefix_caching;
    engine_param_.enable_metrics        = config.enable_metrics;

    engine_param_.num_tokens_per_iter = config.num_tokens_per_iter;
    engine_param_.max_prefill_iters   = config.max_prefill_iters;

    phases_ = config.async_ ? 2 : 1;

    engine_param_.outer_dp_size = config.outer_dp_size;

    engine_param_.attn_dp_size = config.attn_dp_size;
    engine_param_.attn_tp_size = config.attn_tp_size;
    engine_param_.attn_cp_size = config.attn_cp_size;

    engine_param_.mlp_tp_size = config.mlp_tp_size;

    engine_param_.devices = std::move(config.devices);

    // multi-node information
    engine_param_.nnodes    = config.nnodes;
    engine_param_.node_rank = config.node_rank;

    {
        auto sp                             = engine_param_.attn_tp_size * engine_param_.attn_cp_size;
        engine_param_.max_forward_token_num = ((size_t)max_forward_token_num + sp - 1) / sp * sp;
    }

    comm_size_ = engine_param_.attn_dp_size * engine_param_.attn_tp_size * engine_param_.attn_cp_size;
    FT_CHECK(engine_param_.mlp_tp_size == comm_size_);

    communicator_type_ = std::move(config.communicator);

    HandleMissingParams();

    weights_.resize(engine_param_.devices.size());
    engines_.resize(engine_param_.devices.size());
    contexts_.resize(engine_param_.devices.size());

    // NOTE: This runs on Python main thread
    group_id_ = comm::CreateHostGroupId((engine_param_.nnodes == 1) ? "" : "hybrid");
    group_id_->Initialize();

    const int devices = engine_param_.devices.size();

    for (int i = 0; i < devices; ++i) {
        global_rank_.push_back(engine_param_.node_rank * devices + i);
    }

    queue_id_.resize(devices);
    engine_params_.resize(devices, engine_param_);
}
```

- [ ] **Step 5: Update outer TurboMind constructor**

Change lines 510-513 from:
```cpp
TurboMind::TurboMind(string model_dir, string config, FFICtxFactory ffi_ctx_factory):
    impl_{std::make_unique<Impl>(model_dir, config, ffi_ctx_factory)}
{
}
```
to:
```cpp
TurboMind::TurboMind(string model_dir, EngineConfig config, FFICtxFactory ffi_ctx_factory):
    impl_{std::make_unique<Impl>(model_dir, std::move(config), ffi_ctx_factory)}
{
}
```

- [ ] **Step 6: Build**

Run: `cd /data/lmdeploy-modeling/build && ninja turbomind _turbomind 2>&1 | tail -30`
Expected: Build succeeds.

- [ ] **Step 7: Remove yaml-cpp from CMakeLists.txt**

In `src/turbomind/CMakeLists.txt`, change line 39 from:
```
        yaml-cpp::yaml-cpp)
```
to:
```
        )
```

- [ ] **Step 8: Build again to verify yaml-cpp removal**

Run: `cd /data/lmdeploy-modeling/build && ninja turbomind _turbomind 2>&1 | tail -30`
Expected: Build succeeds without yaml-cpp.

- [ ] **Step 9: Commit**

```bash
git add src/turbomind/turbomind.h src/turbomind/turbomind.cc src/turbomind/CMakeLists.txt
git commit -m "refactor: replace YAML parsing with EngineConfig struct in TurboMind"
```

---

### Task 4: Update pybind TurboMind.create binding

**Files:**
- Modify: `src/turbomind/python/bind.cpp:677-694` (update create lambda)

- [ ] **Step 1: Update the TurboMind.create pybind binding**

Change the `.def_static("create", ...)` block (lines 678-694) from:

```cpp
        .def_static(
            "create",
            [](std::string model_dir, std::string config) -> std::shared_ptr<TurboMind> {
                auto gil_factory = [] {  //
                    // erase the type
                    return std::static_pointer_cast<void>(std::make_shared<ScopedGIL>());
                };
                auto no_gil_deleter = [](TurboMind* ptr) {
                    pybind11::gil_scoped_release release;
                    delete ptr;
                };

                std::shared_ptr<TurboMind> model(new TurboMind(model_dir, config, gil_factory), no_gil_deleter);
                return model;
            },
            "model_dir"_a,
            "config"_a = "")
```

to:

```cpp
        .def_static(
            "create",
            [](std::string model_dir, turbomind::EngineConfig config) -> std::shared_ptr<TurboMind> {
                auto gil_factory = [] {  //
                    // erase the type
                    return std::static_pointer_cast<void>(std::make_shared<ScopedGIL>());
                };
                auto no_gil_deleter = [](TurboMind* ptr) {
                    pybind11::gil_scoped_release release;
                    delete ptr;
                };

                std::shared_ptr<TurboMind> model(new TurboMind(model_dir, std::move(config), gil_factory), no_gil_deleter);
                return model;
            },
            "model_dir"_a,
            "engine_config"_a)
```

Key changes: `std::string config` → `turbomind::EngineConfig config`, `"config"_a = ""` → `"engine_config"_a`, added `std::move(config)`.

- [ ] **Step 2: Build**

Run: `cd /data/lmdeploy-modeling/build && ninja _turbomind 2>&1 | tail -20`
Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/python/bind.cpp
git commit -m "refactor: update TurboMind.create to accept EngineConfig"
```

---

### Task 5: Update Python _from_hf to construct EngineConfig

**Files:**
- Modify: `lmdeploy/turbomind/turbomind.py:208-235` (_from_hf method)
- Modify: `lmdeploy/turbomind/turbomind.py:12` (remove `asdict` import if no longer needed)

- [ ] **Step 1: Add _engine_config_to_tm helper and update _from_hf**

Replace the `_from_hf` method (lines 208-235) with:

```python
def _from_hf(self, model_path: str, engine_config: TurbomindEngineConfig):
    """Load model which is in hf format."""
    assert is_supported(model_path), (
        f'turbomind does not support {model_path}. '
        'Plz try pytorch engine instead.')

    from .deploy.converter import get_tm_config
    from .deploy.target_model.base import OUTPUT_MODELS

    spec, model_path = get_tm_config(model_path, engine_config)

    self._vocab_size = spec._vocab_size
    self.engine_config = engine_config

    dtype_map = {
        'bfloat16': _tm.DataType.TYPE_BF16,
        'float16': _tm.DataType.TYPE_FP16,
    }
    ec = _tm.EngineConfig()
    ec.data_type = dtype_map[engine_config.dtype]
    ec.cache_block_seq_len = engine_config.cache_block_seq_len
    ec.quant_policy = engine_config.quant_policy
    ec.max_batch_size = engine_config.max_batch_size
    ec.max_prefill_token_num = engine_config.max_prefill_token_num
    ec.session_len = engine_config.session_len
    ec.cache_max_block_count = engine_config.cache_max_entry_count
    ec.cache_chunk_size = engine_config.cache_chunk_size
    ec.enable_prefix_caching = engine_config.enable_prefix_caching
    ec.enable_metrics = engine_config.enable_metrics
    ec.num_tokens_per_iter = engine_config.num_tokens_per_iter
    ec.max_prefill_iters = engine_config.max_prefill_iters
    ec.async_ = engine_config.async_
    ec.outer_dp_size = engine_config.outer_dp_size
    ec.attn_dp_size = engine_config.attn_dp_size
    ec.attn_tp_size = engine_config.attn_tp_size
    ec.attn_cp_size = engine_config.attn_cp_size
    ec.mlp_tp_size = engine_config.mlp_tp_size
    ec.devices = engine_config.devices
    ec.nnodes = engine_config.nnodes
    ec.node_rank = engine_config.node_rank
    ec.communicator = engine_config.communicator

    logger.info(f'turbomind engine config:\n\n'
                f'dtype={engine_config.dtype}, session_len={engine_config.session_len}, '
                f'max_batch_size={engine_config.max_batch_size}, '
                f'devices={engine_config.devices}, '
                f'tp={engine_config.attn_tp_size}, '
                f'dp={engine_config.attn_dp_size}, '
                f'cp={engine_config.attn_cp_size}')

    model_comm = _tm.TurboMind.create(model_dir='', engine_config=ec)
    self._create_weight(model_comm)

    self._tm_model = OUTPUT_MODELS.get('tm')(
        spec=spec,
        model_comm=model_comm,
        gpu_count=self.gpu_count,
        model_path=model_path)
    return model_comm
```

- [ ] **Step 2: Remove unused imports**

Check if `asdict` (from dataclasses) and `yaml` are used elsewhere in the file. If `asdict` is no longer used anywhere in `turbomind.py`, remove line 12:
```python
from dataclasses import asdict
```

If `yaml` is no longer used anywhere in `turbomind.py`, remove line 20:
```python
import yaml
```

- [ ] **Step 3: Test with a model**

Run the test script to verify the full pipeline works:

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py <model_name>
```

Use any locally cached model from `list_models`. Verify the model responds with meaningful text (at least 128 tokens).

Expected: Model loads successfully and generates coherent text.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/turbomind.py
git commit -m "refactor: replace YAML serialization with EngineConfig in TurboMind._from_hf"
```

---

### Task 6: Verify end-to-end and clean up

**Files:**
- Verify build and test with at least one model

- [ ] **Step 1: Full rebuild**

Run: `cd /data/lmdeploy-modeling/build && ninja 2>&1 | tail -20`
Expected: Full build succeeds with no warnings related to the changes.

- [ ] **Step 2: Test with a model**

Run the test script with a model, generating at least 128 tokens. Verify meaningful output:

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py <model_name>
```

Expected: Model loads, generates coherent text, no errors.

- [ ] **Step 3: Verify yaml-cpp is gone from the build**

Run: `cd /data/lmdeploy-modeling/build && ninja -t clean && ninja _turbomind 2>&1 | grep -i yaml`
Expected: No output (yaml-cpp is no longer part of the build).

- [ ] **Step 4: Final commit (if any remaining cleanup)**

```bash
git add -A
git commit -m "chore: engine config x-macro migration cleanup"
```
(Only if there are uncommitted changes.)
