// verify_07_mode_select.cu — Section 7: Mode Selection & Manipulation
#include <cute/layout.hpp>
#include <cassert>
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Mode Selection & Manipulation Verification ===" << std::endl;

  // --- select<I...> ---
  {
    auto s = make_shape(_4{}, _8{}, _16{});  // (4,8,16)
    auto s02 = select<0,2>(s);                // (4,16)

    std::cout << "select<0,2>((4,8,16)) = ";
    print(s02); std::cout << std::endl;

    static_assert(get<0>(s02) == _4{});
    static_assert(get<1>(s02) == _16{});
    static_assert(rank(s02) == 2);
    std::cout << "  PASS: select<0,2>" << std::endl;

    // select on layouts
    auto layout = make_layout(make_shape(_4{}, _8{}, _16{}),
                              make_stride(_1{}, _4{}, _32{}));
    auto l02 = select<0,2>(layout);
    std::cout << "select<0,2>(layout) = "; print(l02); std::cout << std::endl;
    static_assert(size(l02) == 64);
    std::cout << "  PASS: select on layout" << std::endl;
  }

  // --- take<B,E> ---
  {
    auto s = make_shape(_4{}, _8{}, _16{}, _2{});  // (4,8,16,2)
    auto t = take<0,2>(s);                           // (4,8)

    std::cout << "take<0,2>((4,8,16,2)) = ";
    print(t); std::cout << std::endl;

    static_assert(rank(t) == 2);
    static_assert(get<0>(t) == _4{});
    static_assert(get<1>(t) == _8{});
    std::cout << "  PASS: take<0,2>" << std::endl;

    // take on layouts
    auto layout = make_layout(s, make_stride(_1{}, _4{}, _32{}, _64{}));
    auto lt = take<0,2>(layout);
    std::cout << "take<0,2>(layout) = "; print(lt); std::cout << std::endl;
    std::cout << "  PASS: take on layout" << std::endl;
  }

  // --- append<N> ---
  {
    auto s = make_shape(_4{}, _8{});  // (4,8)
    auto s3 = append<3>(s, _1{});     // (4,8,1)

    std::cout << "append<3>((4,8), 1) = ";
    print(s3); std::cout << std::endl;

    static_assert(rank(s3) == 3);
    static_assert(get<0>(s3) == _4{});
    static_assert(get<1>(s3) == _8{});
    static_assert(get<2>(s3) == _1{});
    std::cout << "  PASS: append<3>" << std::endl;
  }

  // --- replace<I> ---
  {
    auto s = make_shape(_4{}, _8{}, _16{});  // (4,8,16)
    auto s2 = replace<2>(s, _2{});            // (4,8,2)

    std::cout << "replace<2>((4,8,16), 2) = ";
    print(s2); std::cout << std::endl;

    static_assert(get<0>(s2) == _4{});
    static_assert(get<1>(s2) == _8{});
    static_assert(get<2>(s2) == _2{});
    std::cout << "  PASS: replace<2>" << std::endl;

    // replace on layouts
    auto layout = make_layout(make_shape(_4{}, _8{}, _16{}),
                              make_stride(_1{}, _4{}, _32{}));
    auto l2 = replace<2>(layout, make_layout(_2{}, _1{}));
    std::cout << "replace<2>(layout, (2,1)) = "; print(l2); std::cout << std::endl;
    static_assert(size(l2) == 64);  // 4*8*2 = 64
    std::cout << "  PASS: replace on layout" << std::endl;
  }

  // --- flatten ---
  {
    auto s = make_shape(make_shape(_2{}, _4{}), _8{});  // ((2,4),8)
    auto f = flatten(s);                                   // (2,4,8)

    std::cout << "flatten(((2,4),8)) = ";
    print(f); std::cout << std::endl;

    static_assert(rank(f) == 3);
    static_assert(size(f) == 64);
    std::cout << "  PASS: flatten" << std::endl;

    // flatten on layouts
    auto layout = make_layout(make_shape(make_shape(_2{}, _4{}), _8{}),
                              make_stride(make_stride(_1{}, _2{}), _8{}));
    auto fl = flatten(layout);
    std::cout << "flatten(layout) = "; print(fl); std::cout << std::endl;
    static_assert(size(fl) == 64);
    std::cout << "  PASS: flatten on layout" << std::endl;
  }

  // --- make_tile ---
  {
    auto tiler = make_tile(make_layout(_4{}, _1{}),
                           make_layout(_8{}, _2{}));

    std::cout << "make_tile(4:1, 8:2) = (";
    print(get<0>(tiler)); std::cout << ", ";
    print(get<1>(tiler)); std::cout << ")" << std::endl;

    static_assert(size(get<0>(tiler)) == 4);
    static_assert(size(get<1>(tiler)) == 8);
    std::cout << "  PASS: make_tile" << std::endl;
  }

  std::cout << "\nALL PASSED" << std::endl;
  return 0;
}
