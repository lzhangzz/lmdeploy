#pragma once

#include <string>

#include "src/turbomind/kernels/linear_attn/delta_rule.h"

namespace turbomind::linear_attn::delta_rule::flashqla {

// Registers one exported module with FlashQLA's own kernel registry (Python: `_tm.add_flashqla_kernel`).
void AddKernel(const std::string& symbol, const std::string& path);
// True when the Python kernel provider registered all three kernels for the specialization, which it
// does only when FlashQLA was requested (the opt-in). Does not load them.
bool Registered(int hq, int hv, DataType state_dtype);
// Registered, and all three load; loads them on first use (memoized).
bool HasKernels(int hq, int hv, DataType state_dtype);
void Run(const Arguments&, const delta_rule::Plan&, cudaStream_t);

}  // namespace turbomind::linear_attn::delta_rule::flashqla
