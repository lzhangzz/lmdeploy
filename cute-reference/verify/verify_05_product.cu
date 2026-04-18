// verify_05_product.cu — Section 5: Product (Replication)
#include <cute/layout.hpp>
#include <cassert>
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Logical Product Verification ===" << std::endl;

  // --- Example: logical_product((2,2):(4,1), 6:1) ---
  {
    auto A = make_layout(make_shape(_2{}, _2{}),
                         make_stride(_4{}, _1{}));  // (2,2):(4,1) — indices {0,1,4,5}
    auto B = make_layout(make_shape(_6{}), make_stride(_1{}));  // 6:1

    std::cout << "A = "; print(A); std::cout << std::endl;
    std::cout << "B = "; print(B); std::cout << std::endl;

    // Verify A's index pattern
    static_assert(A(_0{},_0{}) == 0);
    static_assert(A(_1{},_0{}) == 4);
    static_assert(A(_0{},_1{}) == 1);
    static_assert(A(_1{},_1{}) == 5);
    std::cout << "  A indices: {" << A(_0{},_0{}) << ","
              << A(_1{},_0{}) << "," << A(_0{},_1{}) << ","
              << A(_1{},_1{}) << "}" << std::endl;

    auto lp = logical_product(A, B);
    std::cout << "logical_product(A, B) = "; print(lp); std::cout << std::endl;

    static_assert(size(lp) == size(A) * size(B));
    std::cout << "  size(lp) = " << size(lp) << " (expect " << (size(A)*size(B)) << ")" << std::endl;
    std::cout << "  PASS: logical_product" << std::endl;
  }

  // --- blocked_product ---
  {
    auto A = make_layout(make_shape(_2{}, _4{}),
                         make_stride(_1{}, _2{}));  // (2,4):(1,2)
    auto B = make_layout(make_shape(_3{}, _2{}),
                         make_stride(_1{}, _3{}));  // (3,2):(1,3)

    std::cout << "\nblocked_product:" << std::endl;
    std::cout << "A = "; print(A); std::cout << std::endl;
    std::cout << "B = "; print(B); std::cout << std::endl;

    auto bp = blocked_product(A, B);
    std::cout << "blocked_product(A, B) = "; print(bp); std::cout << std::endl;

    static_assert(size(bp) == size(A) * size(B));
    std::cout << "  size(bp) = " << size(bp) << std::endl;
    std::cout << "  PASS" << std::endl;
  }

  // --- raked_product ---
  {
    auto A = make_layout(make_shape(_2{}, _4{}),
                         make_stride(_1{}, _2{}));  // (2,4):(1,2)
    auto B = make_layout(make_shape(_3{}, _2{}),
                         make_stride(_1{}, _3{}));  // (3,2):(1,3)

    std::cout << "\nraked_product:" << std::endl;
    std::cout << "A = "; print(A); std::cout << std::endl;
    std::cout << "B = "; print(B); std::cout << std::endl;

    auto rp = raked_product(A, B);
    std::cout << "raked_product(A, B) = "; print(rp); std::cout << std::endl;

    static_assert(size(rp) == size(A) * size(B));
    std::cout << "  size(rp) = " << size(rp) << std::endl;
    std::cout << "  PASS" << std::endl;
  }

  // --- tile_to_shape ---
  {
    auto atom = make_layout(make_shape(_2{}, _4{}),
                            make_stride(_1{}, _2{}));
    auto target = make_shape(_8{}, Int<16>{});

    std::cout << "\ntile_to_shape:" << std::endl;
    std::cout << "atom = "; print(atom); std::cout << std::endl;

    auto tiled = tile_to_shape(atom, target);
    std::cout << "tile_to_shape(atom, (8,16)) = "; print(tiled); std::cout << std::endl;
    std::cout << "  shape = "; print(shape(tiled)); std::cout << std::endl;
    std::cout << "  PASS" << std::endl;
  }

  // --- blocked_product vs raked_product difference ---
  {
    auto A = make_layout(make_shape(_2{}, _2{}), make_stride(_1{}, _2{}));
    auto B = make_layout(make_shape(_2{}, _2{}), make_stride(_1{}, _2{}));

    auto bp = blocked_product(A, B);
    auto rp = raked_product(A, B);

    std::cout << "\nblocked_product vs raked_product:" << std::endl;
    std::cout << "  blocked: "; print(bp); std::cout << std::endl;
    std::cout << "  raked:   "; print(rp); std::cout << std::endl;

    static_assert(size(bp) == size(rp));
    std::cout << "  PASS: both have same size" << std::endl;
  }

  std::cout << "\nALL PASSED" << std::endl;
  return 0;
}
