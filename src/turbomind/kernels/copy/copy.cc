#include "src/turbomind/kernels/copy/copy.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace turbomind::core {

// Forward declarations — defined in copy.cu and transpose.cu
void VectorizedCopy(const void* data_a, void* data_b,
                    const Layout& a, const Layout& b,
                    int rank, DataType dtype, cudaStream_t stream);

void TransposeCopy(const void* data_a, void* data_b,
                   const Layout& a, const Layout& b,
                   DataType dtype, cudaStream_t stream);

// ============================================================================
// GenericCopy: layout normalization + dispatch
// ============================================================================
void GenericCopy(const Tensor& src, Tensor& dst, cudaStream_t stream)
{
    auto a = src.layout();
    auto b = dst.layout();

    TM_CHECK_EQ(a.size(), b.size()) << "GenericCopy: src and dst must have the same number of elements";

    // Sort strides ascending so innermost (fastest-varying) dim is first
    std::vector<int> idxs(a.rank());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {
        return a.stride()[i] < a.stride()[j];
    });

    a = a.permute(idxs);
    b = b.permute(idxs);

    a = a.coalesce();
    b = b.coalesce();

    int rank = std::max(a.rank(), b.rank());

    if (a.rank() < rank) {
        a = a.view(b.shape());
    }
    else if (b.rank() < rank) {
        b = b.view(a.shape());
    }

    const DataType dtype = src.dtype();

    // --- 2D transpose detection ---
    constexpr int kTileDim = 32;
    bool is_2d_transpose = (rank == 2) &&
        (a.stride(0) == 1) && (b.stride(1) == 1) &&
        (a.stride(1) > 1) && (b.stride(0) > 1);

    if (is_2d_transpose &&
        a.shape(0) % kTileDim == 0 && a.shape(1) % kTileDim == 0)
    {
        TransposeCopy(src.raw_data(), dst.raw_data(), a, b, dtype, stream);
        return;
    }

    // --- Vectorized / scalar copy ---
    VectorizedCopy(src.raw_data(), dst.raw_data(), a, b, rank, dtype, stream);
}

}  // namespace turbomind::core
