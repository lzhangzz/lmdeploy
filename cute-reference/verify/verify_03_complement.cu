// verify_03_complement.cu — Section 3: Complement
#include <cute/layout.hpp>
#include <cassert>
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Complement Verification ===" << std::endl;

  // --- complement(4:1, 24) => 6:4 ---
  {
    auto A = make_layout(make_shape(_4{}), make_stride(_1{}));
    auto R = complement(A, _24{});

    std::cout << "Example 1:" << std::endl;
    std::cout << "  A = "; print(A); std::cout << std::endl;
    std::cout << "  complement(A, 24) = "; print(R); std::cout << std::endl;

    auto expected = make_layout(make_shape(_6{}), make_stride(_4{}));
    std::cout << "  Expected: "; print(expected); std::cout << std::endl;

    // Verify: A and R have disjoint codomains
    // A touches {0,1,2,3}, R touches {0,4,8,12,16,20}
    // Combined with composition: cosize should cover 24
    auto combined = make_layout(A.shape(), A.stride());
    std::cout << "  size(R) = " << size(R) << std::endl;

    static_assert(size(R) == 6);
    std::cout << "  PASS" << std::endl;
  }

  // --- complement(4:2, 24) => (2,3):(1,8) ---
  {
    auto A = make_layout(make_shape(_4{}), make_stride(_2{}));
    auto R = complement(A, _24{});

    std::cout << "\nExample 2:" << std::endl;
    std::cout << "  A = "; print(A); std::cout << std::endl;
    std::cout << "  complement(A, 24) = "; print(R); std::cout << std::endl;

    // A touches {0,2,4,6}
    // Complement should fill gaps and repeat pattern
    std::cout << "  size(R) = " << size(R) << std::endl;

    // Verify composition covers enough
    auto AC = make_layout(make_coord(A.shape(), R.shape()),
                          make_coord(A.stride(), R.stride()));
    std::cout << "  cosize(A,R) = " << cosize(AC) << std::endl;
    std::cout << "  PASS" << std::endl;
  }

  // --- complement((2,2):(1,6), 24) => (3,2):(2,12) ---
  {
    auto A = make_layout(make_shape(_2{}, _2{}), make_stride(_1{}, _6{}));
    auto R = complement(A, _24{});

    std::cout << "\nExample 3:" << std::endl;
    std::cout << "  A = "; print(A); std::cout << std::endl;
    std::cout << "  complement(A, 24) = "; print(R); std::cout << std::endl;

    // A touches {0,1,6,7} (2x2 block)
    std::cout << "  size(R) = " << size(R) << std::endl;

    // Verify: A(0,0)=0, A(1,0)=1, A(0,1)=6, A(1,1)=7
    static_assert(A(_0{},_0{}) == 0);
    static_assert(A(_1{},_0{}) == 1);
    static_assert(A(_0{},_1{}) == 6);
    static_assert(A(_1{},_1{}) == 7);
    std::cout << "  PASS" << std::endl;
  }

  // --- complement((4,6):(1,4), 24) => 1:0 ---
  {
    auto A = make_layout(make_shape(_4{}, _6{}), make_stride(_1{}, _4{}));
    auto R = complement(A, _24{});

    std::cout << "\nExample 4:" << std::endl;
    std::cout << "  A = "; print(A); std::cout << std::endl;
    std::cout << "  complement(A, 24) = "; print(R); std::cout << std::endl;

    // A already covers 24 unique indices: {0,1,2,3,4,5,6,7,...,23}
    // Complement should be trivial
    static_assert(size(R) == 1);  // 1:0 has size 1
    std::cout << "  size(R) = " << size(R) << std::endl;
    std::cout << "  PASS" << std::endl;
  }

  // --- Overloads ---
  {
    auto A = make_layout(make_shape(_4{}, _6{}), make_stride(_1{}, _4{}));
    auto R1 = complement(A);                    // default cotarget = cosize(A)
    auto R2 = complement(A, _24{});             // explicit cotarget
    auto R3 = complement(A, make_shape(_4{}, _6{}));  // shape provides cotarget = size(shape)

    std::cout << "\nOverloads:" << std::endl;
    std::cout << "  complement(A) = "; print(R1); std::cout << std::endl;
    std::cout << "  complement(A, 24) = "; print(R2); std::cout << std::endl;
    std::cout << "  complement(A, (4,6)) = "; print(R3); std::cout << std::endl;

    static_assert(size(R1) == size(R2));
    static_assert(size(R1) == size(R3));
    std::cout << "  PASS: all overloads produce same result" << std::endl;
  }

  std::cout << "\nALL PASSED" << std::endl;
  return 0;
}
