// verify_02_composition.cu — Section 2: Composition
#include <cute/layout.hpp>
#include <cassert>
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Composition Verification ===" << std::endl;

  // --- Example 1: 20:2 ∘ (5,4):(4,1) = (5,4):(8,2) ---
  {
    auto A = make_layout(make_shape(Int<20>{}), make_stride(_2{}));
    auto B = make_layout(make_shape(_5{}, _4{}), make_stride(_4{}, _1{}));
    auto R = composition(A, B);

    std::cout << "Example 1:" << std::endl;
    std::cout << "  A = "; print(A); std::cout << std::endl;
    std::cout << "  B = "; print(B); std::cout << std::endl;
    std::cout << "  R = A ∘ B = "; print(R); std::cout << std::endl;

    // Verify: R(c) = A(B(c))
    static_assert(R(_0{},_0{}) == A(B(_0{},_0{})));  // A(0) = 0
    static_assert(R(_1{},_0{}) == A(B(_1{},_0{})));  // A(4) = 8
    static_assert(R(_0{},_1{}) == A(B(_0{},_1{})));  // A(1) = 2
    static_assert(R(_4{},_3{}) == A(B(_4{},_3{})));  // A(19) = 38
    std::cout << "  PASS: R(c) = A(B(c)) verified" << std::endl;
  }

  // --- Example 2: (10,2):(16,4) ∘ (5,4):(1,5) ---
  {
    auto A = make_layout(make_shape(Int<10>{}, _2{}),
                         make_stride(Int<16>{}, _4{}));
    auto B = make_layout(make_shape(_5{}, _4{}),
                         make_stride(_1{}, _5{}));
    auto R = composition(A, B);

    std::cout << "\nExample 2:" << std::endl;
    std::cout << "  A = "; print(A); std::cout << std::endl;
    std::cout << "  B = "; print(B); std::cout << std::endl;
    std::cout << "  R = A ∘ B = "; print(R); std::cout << std::endl;

    // Verify: R(c) = A(B(c))
    static_assert(R(_0{},_0{}) == A(B(_0{},_0{})));
    static_assert(R(_1{},_0{}) == A(B(_1{},_0{})));
    static_assert(R(_0{},_1{}) == A(B(_0{},_1{})));
    static_assert(R(_4{},_3{}) == A(B(_4{},_3{})));
    std::cout << "  PASS: R(c) = A(B(c)) verified" << std::endl;

    // Print some specific values
    std::cout << "  R(0,0) = " << R(_0{},_0{}) << std::endl;
    std::cout << "  R(1,0) = " << R(_1{},_0{}) << std::endl;
    std::cout << "  R(0,1) = " << R(_0{},_1{}) << std::endl;
  }

  // --- By-mode composition (Tiler) ---
  {
    auto A = make_layout(make_shape(make_shape(_4{}, _8{}), make_shape(_6{}, _3{})),
                         make_stride(make_stride(_1{}, Int<16>{}),
                                     make_stride(Int<32>{}, Int<192>{})));

    // Tiler: 2 layouts, one per mode of A
    auto tiler = make_tile(make_layout(make_shape(_2{}, _4{}), make_stride(_1{}, _2{})),
                           make_layout(make_shape(_3{}, _2{}), make_stride(_1{}, _3{})));

    std::cout << "\nBy-mode composition:" << std::endl;
    std::cout << "  A = "; print(A); std::cout << std::endl;

    // By-mode: compose each mode independently
    auto r0 = composition(get<0>(A), get<0>(tiler));
    auto r1 = composition(get<1>(A), get<1>(tiler));
    std::cout << "  mode0: A[0] ∘ tiler[0] = "; print(r0); std::cout << std::endl;
    std::cout << "  mode1: A[1] ∘ tiler[1] = "; print(r1); std::cout << std::endl;

    // Verify mode0: (4,8):(1,16) ∘ (2,4):(1,2)
    static_assert(r0(_0{},_0{}) == get<0>(A)(get<0>(tiler)(_0{},_0{})));
    static_assert(r0(_1{},_0{}) == get<0>(A)(get<0>(tiler)(_1{},_0{})));
    static_assert(r0(_0{},_3{}) == get<0>(A)(get<0>(tiler)(_0{},_3{})));
    std::cout << "  PASS: by-mode composition" << std::endl;
  }

  std::cout << "\nALL PASSED" << std::endl;
  return 0;
}
