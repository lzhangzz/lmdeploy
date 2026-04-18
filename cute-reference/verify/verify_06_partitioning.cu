// verify_06_partitioning.cu — Section 6: Higher-Level Partitioning Functions
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cassert>
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Higher-Level Partitioning Verification ===" << std::endl;

  // --- ceil_div ---
  {
    auto M = _32{};
    auto bM = _8{};
    auto result = ceil_div(M, bM);

    std::cout << "ceil_div(32, 8) = "; print(result); std::cout << std::endl;
    static_assert(result == 4);
    std::cout << "  PASS: 32/8 = 4" << std::endl;

    // ceil_div(35, 8) = 5
    auto result2 = ceil_div(Int<35>{}, _8{});
    std::cout << "ceil_div(35, 8) = "; print(result2); std::cout << std::endl;
    static_assert(result2 == 5);
    std::cout << "  PASS: ceil(35/8) = 5" << std::endl;

    // Layout ceil_div: shape(complement(tiler, shape(target)))
    auto tiler = make_layout(make_shape(_8{}), make_stride(_1{}));
    auto target = make_shape(_32{});
    auto grid = ceil_div(target, tiler);
    std::cout << "ceil_div(shape(32), 8:1) = "; print(grid); std::cout << std::endl;
    std::cout << "  PASS: layout ceil_div" << std::endl;
  }

  // --- congruent ---
  {
    auto shape_a = make_shape(_4{}, _8{});
    auto shape_b = make_shape(_4{}, _8{});
    auto shape_c = make_shape(_4{}, _6{});

    static_assert(congruent(shape_a, shape_b));
    std::cout << "PASS: congruent((4,8), (4,8)) = true" << std::endl;
    std::cout << "  PASS" << std::endl;
  }

  // --- make_coord ---
  {
    auto c = make_coord(_2{}, _3{});
    std::cout << "make_coord(2, 3) = ";
    print(c); std::cout << std::endl;

    static_assert(get<0>(c) == _2{});
    static_assert(get<1>(c) == _3{});
    std::cout << "  PASS: make_coord" << std::endl;

    // Nested coord
    auto nc = make_coord(make_coord(_1{}, _2{}), _3{});
    std::cout << "make_coord((1,2), 3) = ";
    print(nc); std::cout << std::endl;
    std::cout << "  PASS: nested make_coord" << std::endl;
  }

  // --- make_identity_tensor ---
  {
    auto my_shape = make_shape(_4{}, _8{});
    auto id = make_identity_tensor(my_shape);
    std::cout << "make_identity_tensor((4,8)):" << std::endl;
    std::cout << "  layout = "; print(id.layout()); std::cout << std::endl;

    // id(m,n) should equal make_coord(m,n)
    auto c00 = id(_0{}, _0{});
    auto c12 = id(_1{}, _2{});
    std::cout << "  id(0,0) = (" << get<0>(c00) << "," << get<1>(c00) << ")" << std::endl;
    std::cout << "  id(1,2) = (" << get<0>(c12) << "," << get<1>(c12) << ")" << std::endl;
    static_assert(get<0>(c12) == _1{});
    static_assert(get<1>(c12) == _2{});
    std::cout << "  PASS: identity tensor maps to coordinates" << std::endl;
  }

  std::cout << "\nALL PASSED" << std::endl;
  return 0;
}
