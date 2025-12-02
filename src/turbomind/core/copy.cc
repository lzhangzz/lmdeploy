
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/check.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <type_traits>
#include <variant>

namespace turbomind::core {

// picked from "cudaTypedefs.h"
typedef CUresult(CUDAAPI* PFN_cuMemcpyBatchAsync_v12080)(CUdeviceptr_v2*        dsts,
                                                         CUdeviceptr_v2*        srcs,
                                                         size_t*                sizes,
                                                         size_t                 count,
                                                         CUmemcpyAttributes_v1* attrs,
                                                         size_t*                attrIdxs,
                                                         size_t                 numAttrs,
                                                         size_t*                failIdx,
                                                         CUstream               hStream);

namespace {

const auto& GetCopyAPI()
{
    static auto inst = []() -> std::variant<std::monostate, PFN_cuMemcpyBatchAsync_v12080> {
        const auto                      symbol = "cuMemcpyBatchAsync";
        cudaDriverEntryPointQueryResult status{};
        void*                           fpn{};
        TM_CHECK_EQ(cudaGetDriverEntryPoint(symbol, &fpn, cudaEnableDefault, &status), 0);
        if (fpn && status == cudaDriverEntryPointSuccess) {
            return (PFN_cuMemcpyBatchAsync_v12080)fpn;
        }
        else {
            return {};
        }
    }();
    return inst;
}

}  // namespace

BatchCopyV2::~BatchCopyV2() = default;

BatchCopyV2::BatchCopyV2(): self_{this}
{
    Reset();
}

void BatchCopyV2::Run()
{
    std::visit(
        [&](auto&& copy) {
            using T = std::decay_t<decltype(copy)>;
            if constexpr (std::is_same_v<T, PFN_cuMemcpyBatchAsync_v12080>) {
                CUmemcpyAttributes_v1 attr{};
                attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
                attr.flags          = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
                std::vector<size_t> ais(src_.size(), 0);
                size_t              fail_idx{SIZE_MAX};
                copy((CUdeviceptr_v2*)dst_.data(),
                     (CUdeviceptr_v2*)src_.data(),
                     size_.data(),
                     src_.size(),
                     &attr,
                     ais.data(),
                     1,
                     &fail_idx,
                     core::Context::stream().handle());
                if (auto i = fail_idx; i != SIZE_MAX) {
                    TM_CHECK(0) << (void*)src_[i] << " " << size_[i] << " " << (void*)dst_[i];
                }
            }
            else {
                for (unsigned i = 0; i < src_.size(); ++i) {
                    core::Copy(src_[i], size_[i], dst_[i]);
                }
            }
        },
        GetCopyAPI());

    Reset();
}

}  // namespace turbomind::core