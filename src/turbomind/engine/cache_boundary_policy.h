#pragma once

#include <memory>
#include <string>

namespace turbomind {

struct Sequence;
struct EngineConfig;

// Per-sequence decision for whether the scheduler publishes a *partial-block*
// cache boundary node. Full-block prefix/checkpoint behavior is always-on and
// is NOT governed by this policy.
//
// Determinism contract: each predicate MUST be a pure function of cross-rank-
// identical sequence attributes (Request fields, gen_cfg, prompt geometry).
// Each local rank builds its own policy instance and consults it on its own
// Sequence; divergent per-rank decisions corrupt the shared logical view.
class CacheBoundaryPolicy {
public:
    virtual ~CacheBoundaryPolicy() = default;

    // Producer side of the prompt boundary at B (Sequence::prompt_boundary_pos
    // = prompt_len - cache_prompt_boundary_skip): admit populating the boundary
    // (a partial fork_to node when B is mid-block, else the block-aligned
    // checkpoint).
    virtual bool PublishPromptBoundary(const Sequence& s) const = 0;

    // Producer side of the generation boundary: index the terminal partial
    // generated block + adopt the terminal recurrent frontier (PublishGeneration).
    virtual bool PublishGenerationBoundary(const Sequence& s) const = 0;
};

// Built-in policy that returns the two global-boolean knobs unchanged.
class DefaultCacheBoundaryPolicy: public CacheBoundaryPolicy {
public:
    DefaultCacheBoundaryPolicy(bool prompt_boundary, bool generation_boundary):
        prompt_boundary_{prompt_boundary}, generation_boundary_{generation_boundary}
    {
    }

    bool PublishPromptBoundary(const Sequence&) const override
    {
        return prompt_boundary_;
    }
    bool PublishGenerationBoundary(const Sequence&) const override
    {
        return generation_boundary_;
    }

private:
    bool prompt_boundary_;
    bool generation_boundary_;
};

class AutoCacheBoundaryPolicy: public CacheBoundaryPolicy {
public:
    AutoCacheBoundaryPolicy(int min_interval, bool prompt_boundary, bool generation_boundary);

    bool PublishPromptBoundary(const Sequence& s) const override;
    bool PublishGenerationBoundary(const Sequence& s) const override;

private:
    int  threshold_;
    bool prompt_boundary_;
    bool generation_boundary_;
};

// Build the policy named by cfg.cache_boundary_policy ("" or "default" => the
// built-in DefaultCacheBoundaryPolicy). Aborts via TM_CHECK on an unknown name.
std::unique_ptr<CacheBoundaryPolicy> CreateCacheBoundaryPolicy(const EngineConfig& cfg);

}  // namespace turbomind
