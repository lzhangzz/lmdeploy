# Reusable helpers for calling exported TileLang (TVM FFI) kernels from TurboMind.
# Only headers and libtvm_ffi are needed at build time; kernels are found at runtime.
if(NOT TVM_FFI_ROOT)
    message(FATAL_ERROR "ENABLE_TILELANG requires TVM_FFI_ROOT (the tvm_ffi Python package directory)")
endif()
find_path(TILELANG_TVM_FFI_INCLUDE tvm/ffi/function.h PATHS "${TVM_FFI_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(TILELANG_TVM_FFI_LIBRARY tvm_ffi PATHS "${TVM_FFI_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_library(tilelang_utils STATIC library.cc)
set_property(TARGET tilelang_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(tilelang_utils SYSTEM PUBLIC "${TILELANG_TVM_FFI_INCLUDE}")
target_link_libraries(tilelang_utils PUBLIC "${TILELANG_TVM_FFI_LIBRARY}")
