// verify_08_utilities.cu — Section 8: Construction & Validation Utilities
#include <cute/layout.hpp>
#include <cassert>
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Construction & Validation Utilities ===" << std::endl;

  // --- make_ordered_layout ---
  {
    auto layout = make_ordered_layout(make_shape(_8{}, _4{}), LayoutRight{});
    std::cout << "make_ordered_layout((8,4), LayoutRight) = ";
    print(layout); std::cout << std::endl;

    // LayoutRight = row-major = last dimension stride 1
    static_assert(stride<0>(layout) == _4{});
    static_assert(stride<1>(layout) == _1{});
    std::cout << "  PASS: LayoutRight = (4,1)" << std::endl;

    auto layout_left = make_ordered_layout(make_shape(_8{}, _4{}), LayoutLeft{});
    std::cout << "make_ordered_layout((8,4), LayoutLeft) = ";
    print(layout_left); std::cout << std::endl;

    // LayoutLeft = column-major = first dimension stride 1
    static_assert(stride<0>(layout_left) == _1{});
    static_assert(stride<1>(layout_left) == _8{});
    std::cout << "  PASS: LayoutLeft = (1,8)" << std::endl;
  }

  // --- make_shape, make_stride ---
  {
    auto s = make_shape(_4{}, make_shape(_2{}, _3{}));
    auto d = make_stride(_1{}, make_stride(_4{}, _8{}));

    std::cout << "make_shape(4, (2,3)) = "; print(s); std::cout << std::endl;
    std::cout << "make_stride(1, (4,8)) = "; print(d); std::cout << std::endl;

    static_assert(rank(s) == 2);
    static_assert(size(s) == 24);  // 4 * 2 * 3
    static_assert(depth(s) == 2);
    std::cout << "  PASS: make_shape/make_stride" << std::endl;
  }

  // --- LayoutLeft and LayoutRight stride patterns ---
  {
    auto left = Layout<Shape<_8, _4>, Stride<_1, _8>>{};
    std::cout << "Layout<(_8,_4),(_1,_8)> = "; print(left); std::cout << std::endl;

    auto right = Layout<Shape<_8, _4>, Stride<_4, _1>>{};
    std::cout << "Layout<(_8,_4),(_4,_1)> = "; print(right); std::cout << std::endl;

    static_assert(left(_0{},_0{}) == 0);
    static_assert(left(_1{},_0{}) == 1);
    static_assert(left(_0{},_1{}) == 8);
    std::cout << "  PASS: column-major stride pattern" << std::endl;

    static_assert(right(_0{},_0{}) == 0);
    static_assert(right(_0{},_1{}) == 1);
    static_assert(right(_1{},_0{}) == 4);
    std::cout << "  PASS: row-major stride pattern" << std::endl;
  }

  // --- Layout construction ---
  {
    auto layout = make_layout(make_shape(_4{}, _8{}), make_stride(_1{}, _4{}));
    std::cout << "make_layout((4,8), (1,4)) = "; print(layout); std::cout << std::endl;

    // Verify it's a valid layout with correct properties
    static_assert(size(layout) == 32);
    static_assert(cosize(layout) == 32);
    static_assert(layout(_0{},_0{}) == 0);
    static_assert(layout(_3{},_7{}) == 3*1 + 7*4);  // 31
    std::cout << "  PASS: Layout construction and access" << std::endl;
  }

  std::cout << "\nALL PASSED" << std::endl;
  return 0;
}
