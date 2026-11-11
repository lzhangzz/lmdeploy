#include "src/turbomind/kernels/linear_attn/kernel/flashqla/host.h"
#include "src/turbomind/kernels/linear_attn/kernel/plan.h"
#include "src/turbomind/kernels/linear_attn/registrar.h"

#include <limits>
#include <stdexcept>

namespace turbomind::linear_attn::delta_rule {
namespace {

template<DataType StateType>
class FlashQlaGdrKernel final: public GdrKernel {
public:
    const GdrKernelSpec& spec() const noexcept override
    {
        static constexpr GdrKernelSpec value{"flashqla", GdrMode::kChunked, kBfloat16, StateType, 128, 32};
        return value;
    }

    const char* name() const noexcept override
    {
        return StateType == kFloat32 ? "flashqla_chunk32_f32_state" : "flashqla_chunk32_bf16_state";
    }

    bool Match(const Operation& operation, const PlanningContext& context) const override
    {
        if (!detail::MatchesGdrSpec(spec(), operation, context) || context.arch != 1200 || context.physical_batch != 1
            || operation.cp_level != ContextParallelLevel::kOff || context.has_speculative_rows || context.hq <= 0
            || context.hv <= 0 || context.hv % context.hq || context.heads_per_block <= 0
            || context.num_head_groups < (context.hv - 1) / context.heads_per_block + 1
            || context.gate_stride != context.hv || context.beta_stride != context.hv || context.q_offsets.size() < 2
            || context.q_offsets.front() < 0 || context.q_offsets.back() > context.token_slots)
            return false;
        for (size_t index = 1; index < context.q_offsets.size(); ++index) {
            if (context.q_offsets[index] <= context.q_offsets[index - 1])
                return false;
        }
        return flashqla::HasKernels(context.hq, context.hv, StateType);
    }

    bool Plan(const Operation& operation, const PlanningContext& context, delta_rule::Plan* plan) const override
    {
        plan->problem           = detail::BuildProblem(context, spec());
        plan->cp                = detail::BuildDisabledContextParallelPlan(plan->problem, operation.cp_level);
        const size_t   gates    = size_t(context.token_slots) * context.hv;
        constexpr auto max_size = size_t(std::numeric_limits<core::ssize_t>::max());
        if (gates > max_size / 128 || gates > (max_size - 255) / (sizeof(float) + 32 * sizeof(uint16_t)))
            throw std::invalid_argument("FlashQLA tensor/workspace size overflow");
        plan->g_cumsum        = {core::Layout{{1, context.token_slots, context.hv}}, kFloat32, gates};
        plan->resolvent       = {core::Layout{{1, context.token_slots, context.hv, 32}}, kBfloat16, gates * 32};
        plan->out             = {core::Layout{{1, context.token_slots, context.hv, 128}}, kBfloat16, gates * 128};
        plan->workspace_bytes = detail::AlignUp(gates * sizeof(float), 256) + gates * 32 * sizeof(uint16_t);
        plan->workspace       = {
            core::Layout{{static_cast<core::ssize_t>(plan->workspace_bytes)}}, kUint8, plan->workspace_bytes};
        return true;
    }

    void Run(const Arguments& args, const delta_rule::Plan& plan, cudaStream_t stream) const override
    {
        flashqla::Run(args, plan, stream);
    }
};

Registrar reg([](Collector& collector) {
    collector.add<FlashQlaGdrKernel<kFloat32>>();
    collector.add<FlashQlaGdrKernel<kBfloat16>>();
});

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
