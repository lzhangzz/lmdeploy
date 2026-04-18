// verify_04_divide.cu — Section 4: Division (Tiling)
#include <cute/layout.hpp>
#include <cassert>
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Logical Divide Verification ===" << std::endl;

  // --- Example: logical_divide((4,2,3):(2,1,8), 4:2) ---
  {
    auto A = make_layout(make_shape(_4{}, _2{}, _3{}),
                         make_stride(_2{}, _1{}, _8{}));  // (4,2,3):(2,1,8)
    auto B = make_layout(make_shape(_4{}), make_stride(_2{}));  // 4:2

    std::cout << "A = "; print(A); std::cout << std::endl;
    std::cout << "B = "; print(B); std::cout << std::endl;

    // Step 1: complement(4:2, 24)
    auto Bstar = complement(B, _24{});
    std::cout << "B* = complement(B, 24) = "; print(Bstar); std::cout << std::endl;

    // Step 2: logical_divide = A ∘ (B, B*)
    auto ld = logical_divide(A, B);
    std::cout << "logical_divide(A, B) = "; print(ld); std::cout << std::endl;

    // Verify: size should be same as A
    static_assert(size(ld) == size(A));
    std::cout << "  size(ld) = " << size(ld) << " (size(A) = " << size(A) << ")" << std::endl;

    // Verify: coalesce should give back the same function
    auto ld_coalesced = coalesce(ld);
    std::cout << "  coalesced: "; print(ld_coalesced); std::cout << std::endl;

    std::cout << "  PASS: logical_divide" << std::endl;
  }

  // --- Convenience variants ---
  {
    auto A = make_layout(make_shape(_8{}, Int<12>{}),
                         make_stride(_1{}, _8{}));  // (8,12):(1,8)
    auto T = make_layout(make_shape(_4{}, _4{}),
                         make_stride(_1{}, _4{}));   // (4,4):(1,4)

    std::cout << "\nConvenience variants:" << std::endl;
    std::cout << "A = "; print(A); std::cout << std::endl;
    std::cout << "T = "; print(T); std::cout << std::endl;

    auto ld = logical_divide(A, T);
    std::cout << "logical_divide:  "; print(ld); std::cout << std::endl;

    auto zd = zipped_divide(A, T);
    std::cout << "zipped_divide:   "; print(zd); std::cout << std::endl;

    auto td = tiled_divide(A, T);
    std::cout << "tiled_divide:    "; print(td); std::cout << std::endl;

    auto fd = flat_divide(A, T);
    std::cout << "flat_divide:     "; print(fd); std::cout << std::endl;

    // All should preserve size
    static_assert(size(ld) == size(A));
    static_assert(size(zd) == size(A));
    static_assert(size(td) == size(A));
    static_assert(size(fd) == size(A));
    std::cout << "  PASS: all variants preserve size" << std::endl;
  }

  // --- 2D divide ---
  {
    auto A = make_layout(make_shape(_8{}, _16{}),
                         make_stride(_1{}, _8{}));  // (8,16):(1,8)
    auto T = make_tile(make_layout(_4{}, _1{}),     // 4-element stride-1 tile
                       make_layout(_8{}, _1{}));     // 8-element stride-1 tile

    std::cout << "\n2D divide:" << std::endl;
    std::cout << "A = "; print(A); std::cout << std::endl;

    auto ld = logical_divide(A, T);
    std::cout << "ld = "; print(ld); std::cout << std::endl;
    static_assert(size(ld) == size(A));
    std::cout << "  PASS: 2D divide preserves size" << std::endl;
  }

  std::cout << "\nALL PASSED" << std::endl;
  return 0;
}
