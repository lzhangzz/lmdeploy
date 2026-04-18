// verify_00_inttuple.cu — Section 0: IntTuple & Layout foundations
#include <cute/layout.hpp>
#include <cassert>
#include <iostream>

using namespace cute;

int main() {
  int failures = 0;

  // --- IntTuple basics ---
  static_assert(rank(make_shape(_4{}, _8{})) == 2);
  static_assert(size(make_shape(_4{}, _8{})) == 32);
  static_assert(depth(make_shape(_4{}, make_shape(_2{}, _3{}))) == 2);

  // --- Layout: (4,8):(1,4) ---
  auto layout = make_layout(make_shape(_4{}, _8{}), make_stride(_1{}, _4{}));

  std::cout << "Layout: "; print(layout); std::cout << std::endl;

  // layout(2, 3) = 2*1 + 3*4 = 14
  static_assert(layout(_2{}, _3{}) == 14);
  std::cout << "PASS: layout(2,3) == 14" << std::endl;

  // size(layout) = 4*8 = 32
  static_assert(size(layout) == 32);
  std::cout << "PASS: size(layout) == 32" << std::endl;

  // cosize(layout) = max_index + 1
  // max index = 3*1 + 7*4 = 31, cosize = 32
  // NOTE: doc claims 29 — this is likely a bug in the doc
  static_assert(cosize(layout) == 32);
  std::cout << "cosize(layout) = " << cosize(layout) << std::endl;
  if (cosize(layout) != 29) {
    std::cout << "DOC BUG: doc claims cosize=29, actual=" << cosize(layout) << std::endl;
  }

  // shape(layout) == (4,8)
  static_assert(shape(layout) == make_shape(_4{}, _8{}));
  std::cout << "PASS: shape(layout) == (4,8)" << std::endl;

  // stride(layout) == (1,4)
  static_assert(stride(layout) == make_stride(_1{}, _4{}));
  std::cout << "PASS: stride(layout) == (1,4)" << std::endl;

  // --- Hierarchical IntTuple ---
  auto nested_shape = make_shape(make_shape(_2{}, _1{}), _3{});
  static_assert(rank(nested_shape) == 2);    // 2 top-level elements
  static_assert(size(nested_shape) == 6);    // 2*1*3
  static_assert(depth(nested_shape) == 2);   // (shape,int) depth=2

  std::cout << "PASS: IntTuple rank/size/depth" << std::endl;

  if (failures > 0) {
    std::cout << "FAILURES: " << failures << std::endl;
    return 1;
  }
  std::cout << "ALL PASSED" << std::endl;
  return 0;
}
