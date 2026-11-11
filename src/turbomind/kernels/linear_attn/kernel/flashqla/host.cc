#include "src/turbomind/kernels/linear_attn/kernel/flashqla/host.h"
#include "src/turbomind/kernels/linear_attn/kernel/plan.h"
#include "src/turbomind/kernels/tilelang/library.h"
#include "src/turbomind/kernels/tilelang/stream.h"
#include "src/turbomind/kernels/tilelang/tensor.h"

#include <array>
#include <optional>
#include <stdexcept>
#include <string>

namespace turbomind::linear_attn::delta_rule::flashqla {
namespace {
namespace ffi = tvm::ffi;

// Exported symbol names; must match lmdeploy/turbomind/kernels/tilelang/flashqla/__init__.py. The `v2` prefix is
// the positional ABI version.
std::optional<std::array<std::string, 3>> Symbols(int hq, int hv, DataType state_dtype)
{
    if (state_dtype != kFloat32 && state_dtype != kBfloat16)
        return std::nullopt;
    const auto heads = "_h" + std::to_string(hv);
    const auto pair  = heads + "_q" + std::to_string(hq);
    const auto state = state_dtype == kFloat32 ? "_f32" : "_bf16";
    return std::array<std::string, 3>{
        "flashqla_v2_cumsum" + heads, "flashqla_v2_kkt" + pair, "flashqla_v2_fused" + pair + state};
}

// FlashQLA's own kernel registry. Intentionally leaked: loaded CUDA modules must not be torn down during
// static destruction, after the CUDA context may already be gone.
tilelang::Library& Library()
{
    static auto* library = new tilelang::Library;
    return *library;
}

struct Kernels {
    const ffi::Function *cumsum, *kkt, *fused;
};

std::optional<Kernels> FindKernels(int hq, int hv, DataType state_dtype)
{
    const auto symbols = Symbols(hq, hv, state_dtype);
    if (!symbols)
        return std::nullopt;
    auto&   library = Library();
    Kernels kernels{library.Find((*symbols)[0]), library.Find((*symbols)[1]), library.Find((*symbols)[2])};
    if (!kernels.cumsum || !kernels.kkt || !kernels.fused)
        return std::nullopt;
    return kernels;
}

void validate_qkv(const core::Tensor& tensor, int tokens, int heads)
{
    if (tensor.dtype() != kBfloat16 || tensor.ndim() != 4 || tensor.shape(0) != 1 || tensor.shape(1) != tokens
        || tensor.shape(2) != heads || tensor.shape(3) != 128 || tensor.stride(2) != 128 || tensor.stride(3) != 1
        || tensor.stride(1) % 8 != 0 || reinterpret_cast<uintptr_t>(tensor.raw_data()) % 16 != 0)
        throw std::invalid_argument("FlashQLA requires aligned bf16 Q/K/V/output with contiguous head cells");
}

}  // namespace

void AddKernel(const std::string& symbol, const std::string& path)
{
    Library().Add(symbol, path);
}

bool Registered(int hq, int hv, DataType state_dtype)
{
    const auto symbols = Symbols(hq, hv, state_dtype);
    if (!symbols)
        return false;
    auto& library = Library();
    for (const auto& symbol : *symbols) {
        if (!library.Contains(symbol))
            return false;
    }
    return true;
}

bool HasKernels(int hq, int hv, DataType state_dtype)
{
    return FindKernels(hq, hv, state_dtype).has_value();
}

void Run(const Arguments& args, const delta_rule::Plan& plan, cudaStream_t stream)
{
    if (!args.workspace || !args.out || !args.state_ptrs || !args.finished || !args.q_offsets)
        throw std::invalid_argument(
            "FlashQLA requires caller-provided workspace, checkpoint pointers, offsets, mask, and output");
    const auto& workspace = *args.workspace;
    if (!workspace || workspace.dtype() != kUint8 || workspace.device().type != kDEVICE || !workspace.is_contiguous()
        || size_t(workspace.byte_size()) < plan.workspace_bytes
        || reinterpret_cast<uintptr_t>(workspace.raw_data()) % 256)
        throw std::invalid_argument("FlashQLA requires the caller-provided CUDA workspace described by its plan");
    const auto& problem = plan.problem;
    validate_qkv(args.q, problem.token_num, problem.hq);
    validate_qkv(args.k, problem.token_num, problem.hq);
    validate_qkv(args.v, problem.token_num, problem.hv);
    validate_qkv(*args.out, problem.token_num, problem.hv);
    for (const auto* tensor : {&args.g, &args.beta}) {
        if (tensor->dtype() != kFloat32 || tensor->ndim() != 3 || tensor->shape(0) != 1
            || tensor->shape(1) != problem.token_num || tensor->shape(2) != problem.hv
            || tensor->stride(1) != problem.hv || tensor->stride(2) != 1)
            throw std::invalid_argument("FlashQLA requires compact fp32 gates");
    }
    if (args.q_offsets.dtype() != kInt32 || args.q_offsets.size() != problem.sequence_num + 1
        || !args.q_offsets.is_contiguous() || args.finished.dtype() != kBool
        || args.finished.size() != problem.sequence_num || !args.finished.is_contiguous()
        || !args.state_ptrs.is_contiguous()
        || (args.state_ptrs.ndim() == 1 ?
                args.state_ptrs.size() != problem.sequence_num || problem.num_head_groups != 1 :
                args.state_ptrs.ndim() != 2 || args.state_ptrs.shape(0) != problem.sequence_num
                    || args.state_ptrs.shape(1) < problem.num_head_groups))
        throw std::invalid_argument("FlashQLA checkpoint pointer/offset/mask shape mismatch");
    for (const auto* tensor : std::array<const core::Tensor*, 9>{&args.k,
                                                                 &args.v,
                                                                 &args.g,
                                                                 &args.beta,
                                                                 args.out,
                                                                 args.workspace,
                                                                 &args.q_offsets,
                                                                 &args.state_ptrs,
                                                                 &args.finished}) {
        if (tensor->device() != args.q.device())
            throw std::invalid_argument("FlashQLA arguments must be on the same CUDA device");
    }
    auto*        storage          = static_cast<uint8_t*>(const_cast<void*>(workspace.raw_data()));
    const size_t resolvent_offset = detail::AlignUp(plan.g_cumsum.storage_size * sizeof(float), 256);
    using tilelang::ToDLTensor;
    auto q = ToDLTensor(args.q), k = ToDLTensor(args.k), v = ToDLTensor(args.v);
    auto g = ToDLTensor(args.g), beta = ToDLTensor(args.beta), out = ToDLTensor(*args.out);
    auto pointers = ToDLTensor(args.state_ptrs), offsets = ToDLTensor(args.q_offsets);
    auto finished = ToDLTensor(args.finished);
    // TMA operands: the singleton batch stride may be arbitrary (e.g. 0) in borrowed views.
    int64_t q_strides[4], k_strides[4], v_strides[4], out_strides[4];
    tilelang::CollapseBatchStride(q, q_strides);
    tilelang::CollapseBatchStride(k, k_strides);
    tilelang::CollapseBatchStride(v, v_strides);
    tilelang::CollapseBatchStride(out, out_strides);
    int64_t  gc_shape[]{1, problem.token_num, problem.hv};
    int64_t  gc_strides[]{gc_shape[1] * gc_shape[2], gc_shape[2], 1};
    int64_t  a_shape[]{1, problem.token_num, problem.hv, 32};
    int64_t  a_strides[]{gc_strides[0] * 32, gc_strides[1] * 32, 32, 1};
    DLTensor gc{storage, q.device, 3, tilelang::ToDLDataType(kFloat32), gc_shape, gc_strides, 0};
    DLTensor a{storage + resolvent_offset, q.device, 4, tilelang::ToDLDataType(kBfloat16), a_shape, a_strides, 0};
    int64_t  pointer_shape[]{args.state_ptrs.shape(0), 1};
    int64_t  pointer_strides[]{1, 1};
    if (args.state_ptrs.ndim() == 1) {
        pointers.ndim    = 2;
        pointers.shape   = pointer_shape;
        pointers.strides = pointer_strides;
    }
    const auto functions = FindKernels(problem.hq, problem.hv, problem.state_dtype);
    if (!functions)
        throw std::invalid_argument("FlashQLA kernels for Hq=" + std::to_string(problem.hq)
                                    + " Hv=" + std::to_string(problem.hv) + " are not available");
    tilelang::StreamScope scope(args.q.device().id, stream);
    (*functions->cumsum)(&g, &offsets, &gc, problem.total_chunks);
    (*functions->kkt)(&k, &beta, &offsets, &a, problem.total_chunks);
    (*functions->fused)(&q,
                        &k,
                        &v,
                        &a,
                        &gc,
                        &beta,
                        &pointers,
                        &offsets,
                        &finished,
                        &out,
                        args.state_layer_offset,
                        problem.heads_per_block);
}

}  // namespace turbomind::linear_attn::delta_rule::flashqla
