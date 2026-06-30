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
