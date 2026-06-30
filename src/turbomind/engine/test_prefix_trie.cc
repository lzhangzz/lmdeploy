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

TEST_CASE("PrefixTrie::Search finds a partial block and sub-selects fingerprints", "[prefix_trie]")
{
    const int  bs = 4;
    PrefixTrie trie{bs};

    // Insert a PARTIAL block of length 2 whose first token carries an image start.
    std::vector<int>  part = {1, 2};
    const Fingerprint fpA  = FP(1);
    LogicalBlock      blkA{};
    blkA.parent    = nullptr;
    blkA.size      = (int)part.size();
    blkA.tokens    = part;
    blkA.image_fps = {fpA};
    blkA.key       = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(part), {fpA});
    REQUIRE(trie.Insert(blkA));

    // Search a full 4-token span that shares the {1,2} prefix; the image starts at
    // relative position 0. Search must enforce a partial match and land on blkA.
    std::vector<int> full = {1, 2, 3, 4};
    PrefixKey        key{};
    LogicalBlock*    hit = trie.Search(nullptr, key, MakeTokenSpan(full), {fpA}, /*fp_pos=*/{0});
    REQUIRE(hit == &blkA);
    REQUIRE(key == blkA.key);  // on a hit, key is replaced with the matched node's key
}

TEST_CASE("PrefixTrie::Search excludes an image that starts beyond the matched prefix", "[prefix_trie]")
{
    const int  bs = 4;
    PrefixTrie trie{bs};

    // Insert a PARTIAL length-2 block with NO image in its first two tokens.
    std::vector<int> part = {1, 2};
    LogicalBlock     blk{};
    blk.parent = nullptr;
    blk.size   = (int)part.size();
    blk.tokens = part;  // empty image_fps
    blk.key    = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(part));
    REQUIRE(trie.Insert(blk));

    // Image starts at relative position 2 (token index 2). For the length-2 prefix the
    // sub-selection (fp_pos < 2) is empty, so it must match the empty-fps block.
    std::vector<int> full = {1, 2, 3, 4};
    const Fingerprint fpA = FP(7);
    PrefixKey        key{};
    LogicalBlock*    hit = trie.Search(nullptr, key, MakeTokenSpan(full), {fpA}, /*fp_pos=*/{2});
    REQUIRE(hit == &blk);
}

TEST_CASE("PrefixTrie::Search is bounded when fp_pos is shorter than fps (no OOB)", "[prefix_trie]")
{
    const int  bs = 4;
    PrefixTrie trie{bs};

    // Partial length-2 block whose first token carries one image start (fpA).
    std::vector<int>  part = {1, 2};
    const Fingerprint fpA  = FP(1);
    LogicalBlock      blkA{};
    blkA.parent    = nullptr;
    blkA.size      = (int)part.size();
    blkA.tokens    = part;
    blkA.image_fps = {fpA};
    blkA.key       = ExtendPrefixKey(PrefixKey{}, MakeTokenSpan(part), {fpA});
    REQUIRE(trie.Insert(blkA));

    // Regression for the `j < fp_pos.size()` bound: fps has 2 entries but fp_pos only 1.
    // The loop must stop at fp_pos.size() and never read fp_pos[1].
    std::vector<int>  full = {1, 2, 3, 4};
    const Fingerprint fpB  = FP(2);
    PrefixKey         key{};
    LogicalBlock*     hit = trie.Search(nullptr, key, MakeTokenSpan(full), {fpA, fpB}, /*fp_pos=*/{0});
    REQUIRE(hit == &blkA);
}
