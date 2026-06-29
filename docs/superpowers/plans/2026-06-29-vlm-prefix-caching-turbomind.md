# VLM Prefix Caching (native Qwen3.5 ViT) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable TurboMind prefix caching for the native Qwen3.5 C++ ViT VLM path by folding a per-image content fingerprint into the prefix-trie identity, so KV (and the vision encoder forward) are correctly reused across requests that share an image.

**Architecture:** A 256-bit opaque `Fingerprint` (empty = never-equal sentinel) flows from `Qwen3_5VitItem` → an engine-visible `Sequence::multimodal_spans` projection → the scheduler, which folds each image's fingerprint into the cumulative `PrefixKey` and stores it on the *first* `LogicalBlock` of the image; `PrefixTrie::Find` adds an exact fingerprint compare. ViT-skip for fully-cached images is automatic via the existing window-intersection filter once `history_len` advances. Fingerprint *generation* is out of scope (later PR); everything ships dormant (empty fingerprints).

**Tech Stack:** C++17 (TurboMind engine + Qwen3.5 ViT module), pybind11 bindings, Python serving glue (lmdeploy), CMake/Ninja build, CUDA.

---

## How this plan handles testing (read first)

The project has a **Catch2 unit-test harness** (`BUILD_TEST=ON` in `my_generate.sh`; existing examples `src/turbomind/core/test_logger.cc`, `test_core.cc`), but **no tests under `src/turbomind/engine/`** yet. It also mandates model-level verification through `scripts/test_turbomind_model.py` (used AS IS) plus a GPU. This plan uses three gates:

- **Compile gate (every C++ task):** `cd build && ninja _turbomind`. A clean build is the immediate correctness gate for type/signature changes.
- **C++ unit gate (Task 2B):** a new Catch2 test (`src/turbomind/engine/test_prefix_trie.cc`) for `Fingerprint` equality and `PrefixTrie::Find` honoring `image_fps`. Runs on CPU (no GPU), so it can run in the sandbox.
- **Behavioral gate (Task 10):** the text regression run (`test_turbomind_model.py`) + a new VLM harness (`scripts/vlm_prefix_cache_check.py`) that self-validates with a **greedy text-equality oracle** and **log-signal assertions** (scheduler `matched`/`resume` + a new ViT-skip log).
- **GPU rule:** model runs (Task 10) MUST run outside the sandbox (`required_permissions: ["all"]`); the sandbox has no NVIDIA driver. Builds and the C++ unit test run normally. Always check an empty GPU with `nvidia-smi` first.
- **Model/index:** the model-server MCP is unavailable. Test model `Qwen/Qwen3.5-27B`, cache dir `/mnt_cfs/huggingface_hub/hub/` (from `/data/models.json`).
- **Logs need an env var:** the C++ logger level is read from `TM_LOG_LEVEL` (`pipeline.py:57` only `setdefault`s it). The harness sets `os.environ['TM_LOG_LEVEL']='INFO'` explicitly so the INFO lines it asserts on are emitted.

Spec: `docs/superpowers/specs/2026-06-29-vlm-prefix-caching-turbomind-design.md`.

## File map (what each touched file is responsible for)

- Create `src/turbomind/engine/fingerprint.h` — the `Fingerprint` value type + equality semantics.
- Modify `src/turbomind/engine/prefix_key.h` — `HashCombine(Fingerprint)` + `ExtendPrefixKey(..., fps)` overload.
- Modify `src/turbomind/engine/block.h` — `LogicalBlock::image_fps` storage.
- Modify `src/turbomind/engine/prefix_trie.h` — `Find`/`Search` exact fingerprint compare.
- Modify `src/turbomind/engine/request.h` — `MultiModalSpan` projection + `Sequence::multimodal_spans`.
- Modify `src/turbomind/engine/scheduler.cc` — fold/compare fingerprints in `MatchPrompt`/`CreateMissingBlocks`/`SetupForks`; `AcceptState::next_fp`; `PrefixEligible` comment.
- Create `src/turbomind/engine/test_prefix_trie.cc` + modify `src/turbomind/engine/CMakeLists.txt` — Catch2 unit test for fingerprint identity.
- Modify `src/turbomind/models/qwen3_5vit/qwen3_5vit_input.h` — `Qwen3_5VitItem::fingerprint`.
- Modify `src/turbomind/python/bind.cpp` — `Qwen3_5VitItem` fingerprint init arg + property.
- Modify `lmdeploy/turbomind/models/qwen3_5.py` — pass `fingerprint` to the item.
- Modify `src/turbomind/models/qwen3_5vit/qwen3_5vit.cc` — fill `multimodal_spans` in `Add`; add ViT-skip INFO log in `Setup`.
- Modify `lmdeploy/serve/core/vl_async_engine.py` — relax prefix-cache force-disable for native-vision TurboMind.
- Modify `src/turbomind/engine/README.md` — `contracts.prefix-identity` / `contracts.prefix-prepare` updates.
- Create `scripts/vlm_prefix_cache_check.py` — manual VLM prefix-cache harness (image reuse / no-false-hit / dormant).

---

## Task 1: `Fingerprint` value type + hash fold

**Files:**
- Create: `src/turbomind/engine/fingerprint.h`
- Modify: `src/turbomind/engine/prefix_key.h`

- [ ] **Step 1: Create the `Fingerprint` header**

Create `src/turbomind/engine/fingerprint.h`:

```cpp
#pragma once
#include <array>
#include <cstdint>

namespace turbomind {

// 256-bit (SHA-256) opaque multimodal content identity, stored as four 64-bit
// words. All-zero is the reserved "empty" sentinel; an empty fingerprint never
// compares equal to anything -- including another empty fingerprint.
struct Fingerprint {
    std::array<uint64_t, 4> words{};

    bool empty() const noexcept { return words == std::array<uint64_t, 4>{}; }

    friend bool operator==(const Fingerprint& a, const Fingerprint& b) noexcept
    {
        if (a.empty() || b.empty()) {
            return false;
        }
        return a.words == b.words;
    }
    friend bool operator!=(const Fingerprint& a, const Fingerprint& b) noexcept { return !(a == b); }
};

}  // namespace turbomind
```

- [ ] **Step 2: Add a `HashCombine` overload for `Fingerprint`**

In `src/turbomind/engine/prefix_key.h`, add the include and a new overload directly **after** the existing `inline size_t HashCombine(size_t seed, size_t value)` (lines 35–38).

Add near the top includes:

```cpp
#include "src/turbomind/engine/fingerprint.h"
```

Add after the existing `HashCombine`:

```cpp
inline size_t HashCombine(size_t seed, const Fingerprint& fp) noexcept
{
    for (uint64_t w : fp.words) {
        seed = HashCombine(seed, static_cast<size_t>(w));
    }
    return seed;
}
```

- [ ] **Step 3: Build to verify it compiles**

Run: `cd build && ninja _turbomind`
Expected: build SUCCEEDS (the new header/overload are unused so far; this just proves they parse and the include path resolves).

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/engine/fingerprint.h src/turbomind/engine/prefix_key.h
git commit -m "feat(engine): add Fingerprint type and hash fold for VLM prefix caching"
```

---

## Task 2: Prefix-key fold + trie identity (fingerprint-aware lookup)

**Files:**
- Modify: `src/turbomind/engine/prefix_key.h`
- Modify: `src/turbomind/engine/block.h`
- Modify: `src/turbomind/engine/prefix_trie.h`

Defaulted `fps` params keep existing token-only callers compiling and behaviorally unchanged (a non-fingerprinted block has empty `image_fps`, which compares equal to the empty default).

- [ ] **Step 1: Add the `ExtendPrefixKey` fingerprint overload**

In `src/turbomind/engine/prefix_key.h`, add after the existing `ExtendPrefixKey(PrefixKey, TokenSpan)` (lines 64–72):

```cpp
inline PrefixKey ExtendPrefixKey(PrefixKey key, TokenSpan tokens, const std::vector<Fingerprint>& fps)
{
    key = ExtendPrefixKey(key, tokens);  // existing token fold
    for (const Fingerprint& fp : fps) {
        key.hash = HashCombine(key.hash, fp);
    }
    return key;
}
```

- [ ] **Step 2: Add `image_fps` storage to `LogicalBlock`**

In `src/turbomind/engine/block.h`, add the include near the top (after the existing `#include "src/turbomind/engine/prefix_key.h"` at line 10):

```cpp
#include "src/turbomind/engine/fingerprint.h"
```

Then add the field to `LogicalBlock` right after `std::vector<int> tokens;` (line 141):

```cpp
    std::vector<int>         tokens;
    std::vector<Fingerprint> image_fps;  // start-fingerprints of images beginning in this block (usually empty)
```

- [ ] **Step 3: Make `PrefixTrie::Find` and `Search` fingerprint-aware**

In `src/turbomind/engine/prefix_trie.h`, the include of `block.h` already transitively provides `Fingerprint`. Replace the existing `Find` (lines 20–30) and `Search` (lines 34–53) with:

```cpp
    // Exact full lookup: hash, parent, length, token identity, and start-
    // fingerprints must match. `fps` are the start-fingerprints of images whose
    // start token falls inside this block (empty for ordinary blocks).
    LogicalBlock* Find(const LogicalBlock* parent, const PrefixKey& key, TokenSpan tokens,
                       const std::vector<Fingerprint>& fps = {}) const
    {
        if (auto it = index_.find(key); it != index_.end()) {
            LogicalBlock* b = it->second;
            if (b->parent == parent && b->size == tokens.size
                && std::equal(tokens.begin(), tokens.end(), b->tokens.begin())
                && b->image_fps == fps) {  // vector==; empty Fingerprint never equal
                return b;
            }
        }
        return nullptr;
    }

    // Longest partial match within one block (never the full block). `fps`/`fp_pos`
    // describe images whose start token falls inside this block: fp_pos[k] is the
    // block-relative start position of fps[k] (ascending). On a hit, `key` is
    // replaced with the matched node's key.
    LogicalBlock* Search(const LogicalBlock* parent, PrefixKey& key, TokenSpan tokens,
                         const std::vector<Fingerprint>& fps    = {},
                         const std::vector<int>&         fp_pos = {}) const
    {
        std::vector<PrefixKey> prefixes;  // token-only cumulative keys
        PrefixKey              k = key;
        for (const int* it = tokens.begin(); it != tokens.end(); ++it) {
            k.hash = HashCombine(k.hash, static_cast<size_t>(*it));
            ++k.length;
            prefixes.push_back(k);
        }
        if (static_cast<int>(prefixes.size()) == block_size_) {
            prefixes.pop_back();  // enforce a partial match
        }
        for (int i = static_cast<int>(prefixes.size()); i > 0; --i) {
            std::vector<Fingerprint> sub;  // images that begin within [0, i)
            for (size_t j = 0; j < fps.size() && fp_pos[j] < i; ++j) {
                sub.push_back(fps[j]);
            }
            PrefixKey ki = prefixes[i - 1];
            for (const Fingerprint& fp : sub) {
                ki.hash = HashCombine(ki.hash, fp);
            }
            if (LogicalBlock* b = Find(parent, ki, TokenSpan{tokens.begin(), i}, sub)) {
                key = ki;
                return b;
            }
        }
        return nullptr;
    }
```

- [ ] **Step 4: Build to verify it compiles (no behavior change yet)**

Run: `cd build && ninja _turbomind`
Expected: build SUCCEEDS. Existing scheduler callers still pass no `fps` (default empty), so behavior is unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/engine/prefix_key.h src/turbomind/engine/block.h src/turbomind/engine/prefix_trie.h
git commit -m "feat(engine): fingerprint-aware PrefixKey fold and PrefixTrie identity"
```

---

## Task 2B: C++ unit test for fingerprint identity (Catch2)

**Files:**
- Create: `src/turbomind/engine/test_prefix_trie.cc`
- Modify: `src/turbomind/engine/CMakeLists.txt`

This locks the two correctness invariants in isolation (no GPU, no model): `Fingerprint` equality semantics and `PrefixTrie::Find` honoring `image_fps`. The header-only trie lets us build `LogicalBlock`s on the stack — no pool/allocator needed.

- [ ] **Step 1: Write the failing test**

Create `src/turbomind/engine/test_prefix_trie.cc`:

```cpp
// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/engine/fingerprint.h"
#include "src/turbomind/engine/prefix_key.h"
#include "src/turbomind/engine/prefix_trie.h"

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace turbomind;

namespace {
Fingerprint FP(uint64_t a)
{
    return Fingerprint{{a, a + 1, a + 2, a + 3}};
}
}  // namespace

TEST_CASE("Fingerprint: empty never equals; distinct differ; identical match", "[fingerprint]")
{
    Fingerprint empty{};
    REQUIRE(empty.empty());
    REQUIRE_FALSE(empty == empty);  // empty never equals anything -- including itself
    REQUIRE(empty != empty);

    const Fingerprint a = FP(100), b = FP(200), a2 = FP(100);
    REQUIRE(a == a2);
    REQUIRE_FALSE(a == b);
    REQUIRE_FALSE(a == empty);
    REQUIRE_FALSE(empty == a);
}

TEST_CASE("PrefixTrie::Find honors image_fps", "[prefix_trie]")
{
    const int  bs = 4;
    PrefixTrie trie{bs};

    std::vector<int>  toks = {1, 2, 3, 4};
    const Fingerprint fpA = FP(1), fpB = FP(2);

    LogicalBlock blkA{};
    blkA.parent    = nullptr;
    blkA.size      = bs;
    blkA.tokens    = toks;
    blkA.image_fps = {fpA};
    blkA.key       = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(toks), {fpA});
    REQUIRE(trie.Insert(blkA));

    // Same tokens + same fingerprint -> hit.
    {
        const auto key = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(toks), {fpA});
        REQUIRE(trie.Find(nullptr, key, MakeTokenSpan(toks), {fpA}) == &blkA);
    }
    // Same tokens, DIFFERENT fingerprint -> miss (no false hit).
    {
        const auto key = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(toks), {fpB});
        REQUIRE(trie.Find(nullptr, key, MakeTokenSpan(toks), {fpB}) == nullptr);
    }
    // Same tokens, EMPTY fingerprint -> miss (empty never equals).
    {
        const std::vector<Fingerprint> empty_fps = {Fingerprint{}};
        const auto key = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(toks), empty_fps);
        REQUIRE(trie.Find(nullptr, key, MakeTokenSpan(toks), empty_fps) == nullptr);
    }
}

TEST_CASE("PrefixTrie::Find: plain text block matches with empty fps", "[prefix_trie]")
{
    const int  bs = 4;
    PrefixTrie trie{bs};

    std::vector<int> toks = {5, 6, 7, 8};
    LogicalBlock     blk{};
    blk.parent = nullptr;
    blk.size   = bs;
    blk.tokens = toks;  // no image_fps -> empty
    blk.key    = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(toks));
    REQUIRE(trie.Insert(blk));

    const auto key = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(toks));
    REQUIRE(trie.Find(nullptr, key, MakeTokenSpan(toks)) == &blk);        // default fps = {}
    REQUIRE(trie.Find(nullptr, key, MakeTokenSpan(toks), {}) == &blk);
}
```

- [ ] **Step 2: Register the test executable**

Append to `src/turbomind/engine/CMakeLists.txt` (after the existing `set_property(... CUDA_RESOLVE_DEVICE_SYMBOLS ON)` line):

```cmake
if (BUILD_TEST)
    add_executable(test_prefix_trie test_prefix_trie.cc)
    target_link_libraries(test_prefix_trie PRIVATE core Catch2::Catch2WithMain)
endif ()
```

> If the linker reports undefined `LogicalBlock`/`BlockHandle` symbols (it should not — the test only uses header-inline code), add `engine` before `core` in `target_link_libraries`.

- [ ] **Step 3: Build the test**

Run: `cd build && ninja test_prefix_trie`
Expected: build SUCCEEDS; the binary is emitted at `build/bin/test_prefix_trie`.

- [ ] **Step 4: Run the test**

Run: `cd build && ./bin/test_prefix_trie`
Expected: `All tests passed` (3 test cases). This runs on CPU; no GPU/driver needed.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/engine/test_prefix_trie.cc src/turbomind/engine/CMakeLists.txt
git commit -m "test(engine): unit-test Fingerprint identity and PrefixTrie image_fps match"
```

---

## Task 3: Engine-visible multimodal projection

**Files:**
- Modify: `src/turbomind/engine/request.h`

`MultiModalData` stays opaque/forward-declared in the engine; we add a lightweight `(interval, fingerprint)` projection the scheduler can read.

- [ ] **Step 1: Add the `MultiModalSpan` type and the `Sequence` field**

In `src/turbomind/engine/request.h`, add the include after the existing `#include "src/turbomind/core/interval.h"` (line 17):

```cpp
#include "src/turbomind/engine/fingerprint.h"
```

Add the `MultiModalSpan` definition right after `struct MultiModalData;` (line 141):

```cpp
struct MultiModalData;  // defined in models/vision_model.h

// The prefix-cache projection of one multimodal input: its token span and
// content identity. The engine never sees MultiModalData / pixels.
struct MultiModalSpan {
    Interval    interval;     // absolute token span [begin, end)
    Fingerprint fingerprint;  // empty until the generation PR
};
```

Add the vector to `Sequence`, immediately before `multimodal_inputs` (line 248):

```cpp
    // persistent per-sequence vision features (qwen3.5-vit, W1)
    std::vector<MultiModalSpan>                  multimodal_spans;   // engine-visible projection; consumed by scheduler
    std::vector<std::shared_ptr<MultiModalData>> multimodal_inputs;  // opaque (unchanged)
```

- [ ] **Step 2: Build to verify it compiles**

Run: `cd build && ninja _turbomind`
Expected: build SUCCEEDS (new field is default-empty and unused so far).

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/engine/request.h
git commit -m "feat(engine): add MultiModalSpan projection to Sequence"
```

---

## Task 4: Scheduler integration (fold + compare fingerprints)

**Files:**
- Modify: `src/turbomind/engine/scheduler.cc`

- [ ] **Step 1: Add the cursor to `AcceptState`**

In `src/turbomind/engine/scheduler.cc`, extend `Scheduler::AcceptState` (lines 281–288):

```cpp
struct Scheduler::AcceptState {
    const LogicalBlock* parent{};  // trie node reached so far (nullptr = root)
    PrefixKey           key{};

    int                 miss{};         // first block index not matched in the trie
    const LogicalBlock* miss_parent{};  // trie position at the miss, for fork_from
    PrefixKey           miss_key{};

    size_t next_fp = 0;  // monotonic cursor into Sequence::multimodal_spans
};
```

- [ ] **Step 2: Add a start-fingerprint collector helper**

Add this free function inside the existing anonymous namespace in `scheduler.cc` (e.g. right after `ResumeSourceName`, near line 158). It is used only by the rare boundary-knob path in `SetupForks`; the hot path uses the monotonic cursor inline.

```cpp
// Collect start-fingerprints of images whose start token lies in [lo, hi), with
// their block-relative start positions. multimodal_spans is prompt-ordered
// ascending by interval.begin().
void CollectStartFps(const Sequence&           s,
                     int                       lo,
                     int                       hi,
                     std::vector<Fingerprint>& fps,
                     std::vector<int>*         pos = nullptr)
{
    for (const auto& sp : s.multimodal_spans) {
        const int b = sp.interval.begin();
        if (b < lo) {
            continue;
        }
        if (b >= hi) {
            break;
        }
        fps.push_back(sp.fingerprint);
        if (pos) {
            pos->push_back(b - lo);
        }
    }
}
```

- [ ] **Step 3: Rewrite `MatchPrompt` to fold + compare start-fingerprints**

Replace `Scheduler::MatchPrompt` (lines 304–326) with:

```cpp
void Scheduler::MatchPrompt(Sequence& s, AcceptState& st)
{
    const int bs          = logical_.block_size();
    const int full_blocks = s.prompt_len / bs;

    int i = 0;
    for (; i < full_blocks; ++i) {
        const int                offset = i * bs;
        size_t                   cur    = st.next_fp;  // working copy; do not commit on a miss
        std::vector<Fingerprint> fps;
        while (cur < s.multimodal_spans.size() && s.multimodal_spans[cur].interval.begin() < offset + bs) {
            fps.push_back(s.multimodal_spans[cur].fingerprint);
            ++cur;
        }
        const auto tokens = TokenSegment(s, offset, bs);
        const auto next   = ExtendPrefixKey(st.key, tokens, fps);
        if (LogicalBlock* b = trie_.Find(st.parent, next, tokens, fps)) {
            s.block_ids.emplace_back(b);  // retain via BlockHandle copy
            st.parent  = b;
            st.key     = next;
            st.next_fp = cur;  // commit advance only on a match
        }
        else {
            break;  // cursor still at the miss block's first span
        }
    }

    st.miss        = i;
    st.miss_parent = st.parent;
    st.miss_key    = st.key;
}
```

- [ ] **Step 4: Rewrite `CreateMissingBlocks` to fold + store start-fingerprints**

Replace `Scheduler::CreateMissingBlocks` (lines 328–364) with:

```cpp
void Scheduler::CreateMissingBlocks(Sequence& s, AcceptState& st)
{
    const int bs     = logical_.block_size();
    const int prompt = s.prompt_len;

    const int all_blocks = (prompt + bs - 1) / bs;

    for (int i = st.miss; i < all_blocks; ++i) {
        const int                offset = i * bs;
        const int                size   = std::min(prompt - offset, bs);
        std::vector<Fingerprint> fps;
        while (st.next_fp < s.multimodal_spans.size()
               && s.multimodal_spans[st.next_fp].interval.begin() < offset + size) {
            fps.push_back(s.multimodal_spans[st.next_fp].fingerprint);
            ++st.next_fp;
        }
        const auto    tokens = TokenSegment(s, offset, size);
        BlockHandle   h      = logical_.Create(i);
        LogicalBlock& x      = *h;
        x.prefix_id          = cache_.Create(registry_.prefix().object_id(), h.get());
        if (size == bs) {
            const auto next = ExtendPrefixKey(st.key, tokens, fps);
            x.parent        = st.parent;
            x.key           = next;
            x.size          = size;
            x.tokens.assign(tokens.begin(), tokens.end());
            x.image_fps     = fps;  // usually empty
            if (!trie_.Insert(x)) {
                LogCollision(s, CollisionSite::kAccept, offset, offset + size);
                // Stays un-indexed; treated as a private block from here on.
                x.parent = nullptr;
                x.key    = {};
                x.size   = 0;
                x.tokens.clear();
                x.image_fps.clear();
            }
            else {
                st.parent = h.get();
                st.key    = next;
            }
        }
        // The partial last block stays private; parent/key do not advance.
        s.block_ids.push_back(std::move(h));  // request ref
    }
}
```

- [ ] **Step 5: Make `SetupForks` fold + compare start-fingerprints (boundary-knob path)**

In `Scheduler::SetupForks`, update the two boundary-knob branches. Replace the `fork_from` Search block (lines 385–393) with:

```cpp
    // Partial match for the first missed position (fork_from)
    if (fork_match && st.miss < all_blocks) {
        LogicalBlock& x      = *s.block_ids[st.miss];
        const int     offset = st.miss * bs;
        const int     size   = std::min(prompt - offset, bs);
        PrefixKey     k      = st.miss_key;

        std::vector<Fingerprint> fps;
        std::vector<int>         fp_pos;
        CollectStartFps(s, offset, offset + size, fps, &fp_pos);

        if (LogicalBlock* v = trie_.Search(st.miss_parent, k, TokenSegment(s, offset, size), fps, fp_pos)) {
            x.fork_from = BlockHandle{v};  // edge ref
        }
    }
```

Replace the `fork_to` publish block's node setup (lines 407–426) with the fingerprint-aware version (gather start-fingerprints for the node range `[last*bs, last*bs + node_size)` and store them):

```cpp
            if (node_size >= 1) {
                LogicalBlock& x      = *s.block_ids.back();
                const auto    tokens = TokenSegment(s, last * bs, node_size);

                std::vector<Fingerprint> fps;
                CollectStartFps(s, last * bs, last * bs + node_size, fps);

                const auto    next = ExtendPrefixKey(st.key, tokens, fps);
                BlockHandle   vh   = logical_.Create(last);
                LogicalBlock& y    = *vh;
                y.parent           = st.parent;
                y.key              = next;
                y.size             = node_size;
                y.tokens.assign(tokens.begin(), tokens.end());
                y.image_fps        = fps;
                y.prefix_id        = cache_.Create(registry_.prefix().object_id(), vh.get());
                if (trie_.Insert(y)) {
                    x.fork_to   = std::move(vh);  // edge holds the only ref
                    have_target = true;
                }
                else {
                    LogCollision(s, CollisionSite::kPromptBoundary, last * bs, last * bs + node_size);
                    // undiscoverable: vh drops at scope end -> recycle
                }
            }
```

- [ ] **Step 6: Add the `PrefixEligible` clarifying comment (no functional change)**

Replace `Scheduler::PrefixEligible` (lines 211–215) with:

```cpp
bool Scheduler::PrefixEligible(const Sequence& s) const noexcept
{
    // Native VLM (multimodal_spans) is eligible: image identity is carried by the
    // per-image fingerprint folded into the prefix key. The legacy Python-embedding
    // path (input_embeds) stays excluded -- out of scope for this change.
    return enable_prefix_caching_ && !is_warm_up_ && s.input_embeds.empty() && s.input_embeds_offsets.empty()
           && s.token_ids != nullptr;
}
```

- [ ] **Step 7: Build to verify it compiles**

Run: `cd build && ninja _turbomind`
Expected: build SUCCEEDS.

- [ ] **Step 8: Commit**

```bash
git add src/turbomind/engine/scheduler.cc
git commit -m "feat(engine): fold per-image fingerprints into prefix matching/indexing"
```

---

## Task 5: Item plumbing — `Qwen3_5VitItem.fingerprint`, binding, converter

**Files:**
- Modify: `src/turbomind/models/qwen3_5vit/qwen3_5vit_input.h`
- Modify: `src/turbomind/python/bind.cpp`
- Modify: `lmdeploy/turbomind/models/qwen3_5.py`

- [ ] **Step 1: Add the `fingerprint` field to `Qwen3_5VitItem`**

In `src/turbomind/models/qwen3_5vit/qwen3_5vit_input.h`, add the include after `#include "src/turbomind/engine/multimodal_input.h"` (line 6):

```cpp
#include "src/turbomind/engine/fingerprint.h"
```

Replace the `Qwen3_5VitItem` struct (lines 15–28) with:

```cpp
struct Qwen3_5VitItem {
    Modality           modality;
    Tensor             data;
    int                token_begin;
    int                token_end;
    std::array<int, 3> grid_thw;
    Fingerprint        fingerprint{};  // empty until the generation PR

    Qwen3_5VitItem() = default;

    Qwen3_5VitItem(Modality           modality,
                   Tensor             data,
                   int                token_begin,
                   int                token_end,
                   std::array<int, 3> grid_thw,
                   Fingerprint        fingerprint = {}):
        modality{modality},
        data{std::move(data)},
        token_begin{token_begin},
        token_end{token_end},
        grid_thw{grid_thw},
        fingerprint{fingerprint}
    {
    }
};
```

- [ ] **Step 2: Bind `fingerprint` as `py::bytes` (0 or 32 bytes) in `bind.cpp`**

In `src/turbomind/python/bind.cpp`, ensure these includes exist near the top (add any missing):

```cpp
#include <cstring>
#include "src/turbomind/engine/fingerprint.h"
```

Replace the `Qwen3_5VitItem` binding (lines 347–368) with the version below. It adds a defaulted `fingerprint` init arg and a validated read/write property:

```cpp
    auto fp_from_bytes = [](const py::bytes& b) -> ft::Fingerprint {
        // py::bytes -> raw buffer via the CPython API (py::bytes::operator std::string
        // is *explicit*, so `std::string s = b;` would not compile; and this also
        // rejects non-bytes inputs cleanly).
        ft::Fingerprint fp{};
        char*           buf = nullptr;
        Py_ssize_t      len = 0;
        if (PyBytes_AsStringAndSize(b.ptr(), &buf, &len) != 0) {
            throw py::error_already_set();
        }
        if (len == 0) {
            return fp;  // empty sentinel
        }
        if (len != 32) {
            throw std::invalid_argument("Qwen3_5VitItem.fingerprint must be 0 or 32 bytes (SHA-256)");
        }
        std::memcpy(fp.words.data(), buf, 32);
        return fp;
    };
    auto fp_to_bytes = [](const ft::Fingerprint& fp) -> py::bytes {
        return py::bytes(reinterpret_cast<const char*>(fp.words.data()), 32);
    };
    py::class_<QwenVitItem>(multimodal, "Qwen3_5VitItem")
        .def(py::init<>())
        .def(py::init([fp_from_bytes](MMModality              modality,
                                      std::shared_ptr<Tensor> data,
                                      int                     token_begin,
                                      int                     token_end,
                                      std::array<int, 3>      grid_thw,
                                      py::bytes               fingerprint) {
                 return QwenVitItem{modality, *data, token_begin, token_end, grid_thw, fp_from_bytes(fingerprint)};
             }),
             "modality"_a,
             "data"_a,
             "token_begin"_a,
             "token_end"_a,
             "grid_thw"_a,
             "fingerprint"_a = py::bytes())
        .def_readwrite("modality", &QwenVitItem::modality)
        .def_property(
            "data",
            [](const QwenVitItem& self) { return std::make_shared<Tensor>(self.data); },
            [](QwenVitItem& self, std::shared_ptr<Tensor> data) { self.data = *data; })
        .def_readwrite("token_begin", &QwenVitItem::token_begin)
        .def_readwrite("token_end", &QwenVitItem::token_end)
        .def_readwrite("grid_thw", &QwenVitItem::grid_thw)
        .def_property(
            "fingerprint",
            [fp_to_bytes](const QwenVitItem& self) { return fp_to_bytes(self.fingerprint); },
            [fp_from_bytes](QwenVitItem& self, py::bytes b) { self.fingerprint = fp_from_bytes(b); });
```

> Note: `ft` is the existing alias for the `turbomind` namespace in `bind.cpp`. If the file uses a different alias near these lines, match it.

- [ ] **Step 3: Pass `fingerprint` through the Python converter**

In `lmdeploy/turbomind/models/qwen3_5.py`, update the `Qwen3_5VitItem(...)` construction in `to_turbomind_multimodal` (lines 404–411):

```python
            token_begin, token_end = self._offset_pair(input_mm['offset'])
            items.append(
                _tm.multimodal.Qwen3_5VitItem(
                    modality=tm_modality,
                    data=data,
                    token_begin=token_begin,
                    token_end=token_end,
                    grid_thw=grid_thw,
                    fingerprint=input_mm.get('fingerprint', b''),
                ))
```

- [ ] **Step 4: Build to verify it compiles**

Run: `cd build && ninja _turbomind`
Expected: build SUCCEEDS.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/models/qwen3_5vit/qwen3_5vit_input.h src/turbomind/python/bind.cpp lmdeploy/turbomind/models/qwen3_5.py
git commit -m "feat(qwen3_5): plumb image fingerprint from Python into Qwen3_5VitItem"
```

---

## Task 6: Vision module — fill the projection + ViT-skip log

**Files:**
- Modify: `src/turbomind/models/qwen3_5vit/qwen3_5vit.cc`

- [ ] **Step 1: Populate `multimodal_spans` alongside `multimodal_inputs` in `Add`**

In `src/turbomind/models/qwen3_5vit/qwen3_5vit.cc`, replace the body of the per-item loop in `int Add(Sequence& s)` (lines 222–229) with:

```cpp
                const int tokens = item.token_end - item.token_begin;
                if (tokens <= 0) {
                    return Request::kInvalid;
                }

                const Interval interval{item.token_begin, Interval::Size{tokens}};
                s.multimodal_inputs.push_back(
                    std::make_shared<MultiModalData>(MultiModalData{item.data, interval, item.grid_thw}));
                s.multimodal_spans.push_back(MultiModalSpan{interval, item.fingerprint});
```

- [ ] **Step 2: Add the ViT-skip observability log in `Setup`**

In `Qwen3_5Vit::Setup`, count multimodal prefill sequences and total images, then log a single greppable line. Replace the request-collection block — from the `// collect image/video pixel values, grid_thws and embeds_coords` comment through the end of the request `for` loop (lines 383–410) — with the instrumented version below. The surrounding code is unchanged: `input_ids_offsets`, `image_embeds_offsets`, `d.Clear()`, and `std::vector<Tensor> pixel_values` are still declared just above (lines 378–381), and the `// copy pixel values to batch input` block still follows (line 412+).

```cpp
        // collect image/video pixel values, grid_thws and embeds_coords
        Buffer_<Sequence*> rc = env.at("requests").buffer();
        int                mm_prefill_seqs = 0;  // prefill sequences carrying multimodal inputs
        int                images_total    = 0;  // total images across those sequences
        for (int i = 0; i < rc.size(); ++i) {
            const auto& s = *rc[i];

            if ((not s.autoregres) && (not s.multimodal_inputs.empty())) {
                ++mm_prefill_seqs;
                images_total += (int)s.multimodal_inputs.size();
                Interval text{s.history_len + s.inflight_input_len, Interval::Size{s.input_len}};
                for (const auto& mm : s.multimodal_inputs) {
                    auto o = mm->interval & text;
                    if (auto size = (int)o.size()) {
                        pixel_values.push_back(mm->data);
                        d.batch_size += mm->data.shape(0);

                        const int text_offset  = input_ids_offsets + o.begin() - text.begin();
                        const int image_offset = image_embeds_offsets + o.begin() - mm->interval.begin();
                        d.input_embeds_coords.emplace_back(size, text_offset);
                        d.image_embeds_coords.emplace_back(size, image_offset);

                        auto& grid_thw = mm->grid_thw;
                        d.grid_thws_host.emplace_back(grid_thw);
                        auto prod = std::accumulate(grid_thw.begin(), grid_thw.end(), 1, std::multiplies<int>());
                        image_embeds_offsets += (prod / cfg.spatial_merge_size / cfg.spatial_merge_size);
                    }
                }
            }

            input_ids_offsets += s.autoregres ? 1 : s.input_len;
        }

        // Prefix-cache observability: on a fully-cached image, the window filter
        // above batches 0 images (ViT skipped). Only logged for multimodal
        // prefill passes so decode steps stay quiet.
        if (mm_prefill_seqs > 0) {
            const int images_batched = (int)pixel_values.size();
            TM_LOG_INFO("Qwen3.5 ViT setup: mm_seqs={} images_batched={} images_skipped={} patches={}",
                        mm_prefill_seqs,
                        images_batched,
                        images_total - images_batched,
                        d.batch_size);
        }
```

> `qwen3_5vit.cc` already includes `src/turbomind/core/logger.h` (line 5), and `TM_LOG_INFO` uses `fmt`-style `{}` placeholders (same as `scheduler.cc`), so no new include is needed.

- [ ] **Step 3: Build to verify it compiles**

Run: `cd build && ninja _turbomind`
Expected: build SUCCEEDS.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/qwen3_5vit/qwen3_5vit.cc
git commit -m "feat(qwen3_5vit): fill multimodal_spans and log ViT-skip on cache hit"
```

---

## Task 7: Python enablement — relax the VLM prefix-cache force-disable

**Files:**
- Modify: `lmdeploy/serve/core/vl_async_engine.py`

- [ ] **Step 1: Permit the native-vision TurboMind path**

In `lmdeploy/serve/core/vl_async_engine.py`, replace the force-disable block (lines 33–38) with:

```python
        if backend_config and backend_config.enable_prefix_caching:
            native_tm_vision = (backend == 'turbomind'
                                and getattr(self.vl_encoder.model, '_turbomind_native_vision', False))
            pytorch_new_preprocess = (backend == 'pytorch'
                                      and getattr(self.vl_encoder, '_uses_new_preprocess', False))
            if not (native_tm_vision or pytorch_new_preprocess):
                backend_config.enable_prefix_caching = False
                logger.warning('Prefix caching is disabled for this VL model path. '
                               'Supported: TurboMind native-vision models and PyTorch new-preprocess '
                               'multimodal inputs.')
```

- [ ] **Step 2: Confirm the native-vision flag is set on the model**

Run: `rg -n "_turbomind_native_vision" lmdeploy/`
Expected: `lmdeploy/vl/model/qwen3_5.py` sets `_turbomind_native_vision = True` and `lmdeploy/vl/model/base.py` defaults it to `False`. The flag already exists, so no model-class change is needed here — this step only confirms the `getattr(self.vl_encoder.model, '_turbomind_native_vision', False)` lookup added in Step 1 will resolve to `True` for Qwen3.5.

- [ ] **Step 3: Syntax check**

Run: `python -c "import ast; ast.parse(open('lmdeploy/serve/core/vl_async_engine.py').read())"`
Expected: no output, exit 0.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/serve/core/vl_async_engine.py
git commit -m "feat(serve): enable prefix caching for native-vision TurboMind VLM"
```

---

## Task 8: Normative contract update

**Files:**
- Modify: `src/turbomind/engine/README.md`

- [ ] **Step 1: Update `contracts.prefix-identity`**

In `src/turbomind/engine/README.md`, find the `contracts.prefix-identity` item and update it to read (preserve the file's existing line breaks/wrapping style; edit content only):

```markdown
Prefix identity is token identity, per-image content identity, plus parent
identity. Index lookup must use cumulative `PrefixKey`, exact parent identity,
exact segment-token comparison, and exact comparison of the block's
start-fingerprints (`LogicalBlock::image_fps`). A fingerprint is the image's
opaque 256-bit content identity; an empty fingerprint never compares equal to
anything, including another empty fingerprint. Blocks interior to an image carry
no fingerprint of their own -- their identity is carried by the cumulative key
and the parent chain, since the image's first block exact-compares the
fingerprint. Hash equality alone is never identity.
```

- [ ] **Step 2: Update `contracts.prefix-prepare`**

Find the `contracts.prefix-prepare` item and append:

```markdown
`Accept` folds each image's fingerprint into the cumulative key at the block
where the image starts (from `Sequence::multimodal_spans`) and stores it on that
`LogicalBlock`; partial-block `Search` applies the same folding when the boundary
knobs are enabled.
```

- [ ] **Step 3: Add a one-line concepts note (optional but recommended)**

In the `concepts` section, add:

```markdown
`Sequence::multimodal_spans` is the engine-visible `(token span, fingerprint)`
projection of multimodal inputs; `multimodal_inputs` (pixels) stays opaque.
```

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/engine/README.md
git commit -m "docs(engine): document fingerprint in prefix-identity/prefix-prepare contracts"
```

---

## Task 9: VLM prefix-cache verification harness

**Files:**
- Create: `scripts/vlm_prefix_cache_check.py`

This harness drives the native Qwen3.5 VL path with images and a stand-in content fingerprint (sha256 of preprocessed pixels) injected at the converter-input dict. It captures the engine's INFO logs (C++ `TM_LOG` writes to fd 2) via fd redirection and self-validates with a greedy text-equality oracle plus log-signal assertions.

- [ ] **Step 1: Write the harness**

Create `scripts/vlm_prefix_cache_check.py`:

```python
#!/usr/bin/env python3
"""Manual harness: verify TurboMind native-Qwen3.5 ViT prefix caching for images.

Scenarios (greedy decoding, sequential requests in one pipeline so the 2nd sees
the 1st's published blocks):

  reuse    same image + same prompt twice, stand-in fingerprint injected.
           Expect: warm request reuses the image span -> ViT logs images_batched=0
           (skip); warm text == cold text.
  distinct two DIFFERENT images of equal token length, fingerprints injected.
           Expect: NO false hit -> the 2nd image is re-encoded (no ViT line with
           images_batched=0); both outputs non-empty.
  dormant  same image twice, NO fingerprint injected (empty fp).
           Expect: image reuse stays dormant -> 2nd image re-encoded (no
           images_batched=0 line); outputs non-empty and equal (recompute).

Run OUTSIDE the sandbox (needs a GPU):

  python scripts/vlm_prefix_cache_check.py \
      --model-id Qwen/Qwen3.5-27B --cache-dir /mnt_cfs/huggingface_hub/hub/ \
      --tp 1 --gpus 0 --scenario reuse
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import re
import sys
import tempfile


def _set_hf_cache(cache_dir: str) -> None:
    import huggingface_hub.constants as hf_constants
    hf_constants.HF_HUB_OFFLINE = 1
    hf_constants.HF_HUB_CACHE = cache_dir


def make_image(seed: int, size=(448, 448)):
    """Deterministic RGB image; same seed -> identical pixels, different seed ->
    different content. Fixed size -> identical Qwen grid_thw (equal token len)."""
    import random
    from PIL import Image, ImageDraw
    rng = random.Random(seed)
    img = Image.new('RGB', size, (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    d = ImageDraw.Draw(img)
    for _ in range(10):
        x0, y0 = rng.randint(0, size[0] - 1), rng.randint(0, size[1] - 1)
        x1, y1 = rng.randint(0, size[0] - 1), rng.randint(0, size[1] - 1)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=color)
    return img


def install_fingerprint_patch() -> None:
    """Inject sha256(pixels) at the converter input dict; exercises the real
    `input_mm.get('fingerprint')` path. Mirrors the future generator."""
    from lmdeploy.turbomind.models.qwen3_5 import Qwen3_5VisionModel
    if getattr(Qwen3_5VisionModel, '_fp_patched', False):
        return
    _orig = Qwen3_5VisionModel.to_turbomind_multimodal

    def _patched(self, multimodal):
        for mm in multimodal:
            pv = mm.get('pixel_values', mm.get('pixel_values_videos'))
            mm['fingerprint'] = hashlib.sha256(pv.contiguous().cpu().numpy().tobytes()).digest()
        return _orig(self, multimodal)

    Qwen3_5VisionModel.to_turbomind_multimodal = _patched
    Qwen3_5VisionModel._fp_patched = True


@contextlib.contextmanager
def capture_low_level_output():
    """Redirect fds 1 & 2 to a temp file so C++ TM_LOG output is captured."""
    f = tempfile.NamedTemporaryFile('w+', suffix='.log', delete=False)
    saved_out, saved_err = os.dup(1), os.dup(2)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)
    try:
        yield f.name
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        f.close()


VIT_RE = re.compile(r'Qwen3\.5 ViT setup: mm_seqs=(\d+) images_batched=(\d+) images_skipped=(\d+) patches=(\d+)')
MATCH_RE = re.compile(r'matched \[0,(\d+)\) \((\d+) blk')
RESUME_RE = re.compile(r'resume \[0,(\d+)\).*source=(\w+)')


def parse_log(path: str):
    vit, matched, resume = [], [], []
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            if (m := VIT_RE.search(line)):
                vit.append(tuple(int(x) for x in m.groups()))
            if (m := MATCH_RE.search(line)):
                matched.append(tuple(int(x) for x in m.groups()))
            if (m := RESUME_RE.search(line)):
                resume.append((int(m.group(1)), m.group(2)))
    return vit, matched, resume


def run(args) -> int:
    _set_hf_cache(args.cache_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # The C++ TurboMind logger reads its level from TM_LOG_LEVEL; pipeline(...) only
    # setdefault()s it, so set it explicitly to guarantee the INFO lines we assert on.
    os.environ['TM_LOG_LEVEL'] = 'INFO'

    inject = args.scenario in ('reuse', 'distinct')
    if inject:
        install_fingerprint_patch()

    if args.scenario == 'distinct':
        images = [make_image(1), make_image(2)]  # different content, equal size
    else:
        img = make_image(1)
        images = [img, img]  # same image twice

    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

    engine_config = TurbomindEngineConfig(
        tp=args.tp,
        session_len=8192,
        cache_max_entry_count=0.5,
        enable_prefix_caching=True,
    )
    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False)
    prompt = args.prompt

    texts = []
    with capture_low_level_output() as log_path:
        with pipeline(args.model_id, backend_config=engine_config, log_level='INFO',
                      trust_remote_code=True) as pipe:
            for image in images:
                out = pipe((prompt, image), gen_config=gen_config)
                texts.append(out.text if hasattr(out, 'text') else str(out))

    vit, matched, resume = parse_log(log_path)

    # --- report (now that fds are restored) ---
    print(f'=== scenario: {args.scenario} (inject_fingerprint={inject}) ===')
    print(f'log: {log_path}')
    for i, t in enumerate(texts):
        print(f'--- response {i} ({len(t)} chars) ---')
        print(t.strip()[:400])
    print(f'ViT setup lines (mm_seqs, images_batched, images_skipped, patches): {vit}')
    print(f'matched (M, blk): {matched}')
    print(f'resume (history, source): {resume}')

    # --- assertions ---
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(('PASS: ' if cond else 'FAIL: ') + msg)
        ok = ok and cond

    check(all(t.strip() for t in texts), 'both responses are non-empty')
    check(len(vit) >= 1, 'ViT setup was logged at least once')
    skipped_image = any(b == 0 and s >= 1 for (_, b, s, _) in vit)

    if args.scenario == 'reuse':
        check(skipped_image, 'warm request skipped its image (images_batched=0) -> ViT skip')
        check(texts[0].strip() == texts[1].strip(), 'warm text == cold text (greedy oracle)')
    elif args.scenario == 'distinct':
        check(not skipped_image, 'no false hit: every image was re-encoded (no images_batched=0)')
    elif args.scenario == 'dormant':
        check(not skipped_image, 'image reuse dormant: image re-encoded (no images_batched=0)')
        check(texts[0].strip() == texts[1].strip(), 'recompute is deterministic (texts equal)')

    print('RESULT:', 'OK' if ok else 'FAILED')
    return 0 if ok else 1


def main(argv) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-id', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--tp', type=int, required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--scenario', choices=['reuse', 'distinct', 'dormant'], required=True)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--prompt', default='Describe this image in detail.')
    return run(p.parse_args(argv[1:]))


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
```

- [ ] **Step 2: Syntax check**

Run: `python -c "import ast; ast.parse(open('scripts/vlm_prefix_cache_check.py').read())"`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/vlm_prefix_cache_check.py
git commit -m "test(scripts): add native-Qwen3.5 VLM prefix-cache verification harness"
```

---

## Task 10: End-to-end verification (GPU, outside sandbox)

**Files:** none (verification only).

All commands here touch CUDA and MUST run outside the sandbox (`required_permissions: ["all"]`). First confirm an empty GPU.

- [ ] **Step 1: Full build**

Run: `cd build && ninja`
Expected: build SUCCEEDS (all targets, including `_turbomind`).

- [ ] **Step 2: Check for an empty GPU**

Run: `nvidia-smi`
Expected: identify a GPU with ~0% utilization and enough free memory (a 27B fp16 model needs ~54 GB; choose `--tp`/`--gpus` accordingly — e.g. a single 80 GB card with `--tp 1 --gpus 0`, otherwise `--tp 2 --gpus 0,1`).

- [ ] **Step 3: Text-path regression (model script AS IS)**

Run:
```bash
python scripts/test_turbomind_model.py \
  --model-id Qwen/Qwen3.5-27B \
  --cache-dir /mnt_cfs/huggingface_hub/hub/ \
  --tp 1 --gpus 0 \
  --enable-prefix-caching \
  --max-new-tokens 128
```
Expected: a coherent, on-topic response of >=128 tokens (gibberish = bug). This guards the shared `PrefixTrie`/`Scheduler` paths against regressions.

- [ ] **Step 4: Scenario `reuse` (image KV reuse + ViT skip)**

Run:
```bash
python scripts/vlm_prefix_cache_check.py \
  --model-id Qwen/Qwen3.5-27B --cache-dir /mnt_cfs/huggingface_hub/hub/ \
  --tp 1 --gpus 0 --scenario reuse
```
Expected: `RESULT: OK`. Specifically: a ViT line with `images_batched=0` (warm image skipped), warm text == cold text, both responses coherent. If `RESULT: FAILED`, debug before continuing (see Debugging Loop below).

- [ ] **Step 5: Scenario `distinct` (no false hit)**

Run:
```bash
python scripts/vlm_prefix_cache_check.py \
  --model-id Qwen/Qwen3.5-27B --cache-dir /mnt_cfs/huggingface_hub/hub/ \
  --tp 1 --gpus 0 --scenario distinct
```
Expected: `RESULT: OK`. No ViT line with `images_batched=0` (the second, different image is re-encoded — proves different fingerprints do not collide).

- [ ] **Step 6: Scenario `dormant` (empty fingerprint = no image reuse)**

Run:
```bash
python scripts/vlm_prefix_cache_check.py \
  --model-id Qwen/Qwen3.5-27B --cache-dir /mnt_cfs/huggingface_hub/hub/ \
  --tp 1 --gpus 0 --scenario dormant
```
Expected: `RESULT: OK`. No `images_batched=0` line (empty fingerprints never match, so the image is always re-encoded), outputs coherent and equal.

- [ ] **Step 7: Record results**

Confirm Steps 3–6 all passed. If all green, the feature works end-to-end: text caching unaffected, image KV + ViT reused only when fingerprints match, and dormant by default.

### Debugging Loop (if any scenario fails)

Iterate until fixed (do not stop with active bugs):

1. Re-run the failing scenario and read the printed `log:` file fully (it has the full engine INFO trace).
2. If `reuse` shows no `images_batched=0`: the image block did not match. Check `matched`/`resume source` lines — if `matched` stops before the image, the fingerprint fold/compare disagrees between `MatchPrompt` (Task 4 Step 3) and `CreateMissingBlocks` (Task 4 Step 4) — confirm both use `ExtendPrefixKey(..., fps)` with the same cursor semantics and that `qwen3_5.py` actually forwards the digest (Task 5 Step 3) and the patch is installed.
3. If `distinct` shows `images_batched=0` (false hit): `PrefixTrie::Find` is not comparing `image_fps` (Task 2 Step 3) or `CreateMissingBlocks` did not store `x.image_fps` (Task 4 Step 4).
4. If gibberish output on the warm run: KV reuse is returning wrong blocks — verify the fingerprint is folded at the image's *first* block only and that continuation blocks keep empty `image_fps`.
5. Rebuild (`cd build && ninja`) and repeat.

---

## Self-review notes (author checklist, already applied)

- **Spec coverage:** §2 type/plumbing → Tasks 1,5; §3 key/trie → Tasks 2,2B,4; §4 projection/cursor → Tasks 3,4,6; §5 eligibility/Python → Tasks 4,7; §6 contract/testing → Tasks 8,9,10; §7 ViT skip log + verification → Tasks 6,10.
- **Type consistency:** `Fingerprint`, `MultiModalSpan`, `LogicalBlock::image_fps`, `AcceptState::next_fp`, `ExtendPrefixKey(key, tokens, fps)`, `Find(parent, key, tokens, fps)`, `Search(parent, key, tokens, fps, fp_pos)`, `CollectStartFps(...)`, `Qwen3_5VitItem.fingerprint` are used identically across tasks.
- **Defaulted params** keep the build green between Task 2 and Task 4 with no behavior change for text-only requests (empty `image_fps` == empty default).
- **Verified against live code (deep review):** `Interval` API (`begin()`/`Size`/`operator&`); `ft = turbomind` alias in `bind.cpp`; `py::bytes` conversion via `PyBytes_AsStringAndSize` (the implicit `std::string` form does not compile); `multimodal_spans` are prompt-ordered (`preprocess_utils.py:274` sorts items by `offset[0]`); `pipe((prompt, PIL.Image))` is an accepted input (`processors/multimodal.py:282-287`); mRoPE stays correct under ViT-skip because `SetupMrope` walks the persistent `s.multimodal_inputs` (all images) independent of the windowed pixel batch; the C++ logger level comes from `TM_LOG_LEVEL` (harness sets it); ViT-skip log uses `fmt`-style `{}` (file already includes `logger.h`); a Catch2 harness exists with `BUILD_TEST=ON`, so Task 2B adds a real engine unit test (binaries land in `build/bin/`).
