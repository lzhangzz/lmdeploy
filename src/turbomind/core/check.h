
// Inspired by <glog/logging.h>

#pragma once

#include <sstream>

namespace turbomind::core {

#define TM_LIKELY(expr) (__builtin_expect(bool(expr), 1))
#define TM_UNLIKELY(expr) (__builtin_expect(bool(expr), 0))
#define TM_NOINLINE __attribute__((noinline))
#define TM_UNREACHABLE __builtin_unreachable()

#define TM_DISABLE_CHECK_STREAM 0
#define TM_DISABLE_CHECK_OP 0

class CheckErrorStream {
public:
    CheckErrorStream(const char* file, int line, const char* expr);

    CheckErrorStream(const char* file, int line, const char* expr, std::string* str);

    ~CheckErrorStream()
    {
        Report();
    }

    template<class T>
    CheckErrorStream& operator<<(const T& msg)
    {
#if TM_DISABLE_CHECK_STREAM
#else
        *oss_ << msg;
#endif
        return *this;
    }

private:
    [[noreturn]] void Report();

    std::ostringstream* oss_;
    const char*         file_;
    int                 line_;
};

class CheckOpStringBuilder {
public:
    CheckOpStringBuilder();
    std::ostream* ForVal1();
    std::ostream* ForVal2();
    std::string*  NewString();

private:
    std::ostringstream* oss_;
};

template<class T1, class T2>
std::string* MakeCheckOpString(const T1& v1, const T2& v2) TM_NOINLINE;

template<class T1, class T2>
std::string* MakeCheckOpString(const T1& v1, const T2& v2)
{
    CheckOpStringBuilder builder;
    *builder.ForVal1() << v1;
    *builder.ForVal2() << v2;
    return builder.NewString();
}

#define DEFINE_CHECK_OP_IMPL(name, op)                                                                                 \
    template<class T1, class T2>                                                                                       \
    inline std::pair<bool, std::string*> name##Impl(const T1& v1, const T2& v2)                                        \
    {                                                                                                                  \
        if (TM_LIKELY(v1 op v2))                                                                                       \
            return {false, nullptr};                                                                                   \
        else                                                                                                           \
            return {true, MakeCheckOpString(v1, v2)};                                                                  \
    }

DEFINE_CHECK_OP_IMPL(Check_EQ, ==);
DEFINE_CHECK_OP_IMPL(Check_NE, !=);
DEFINE_CHECK_OP_IMPL(Check_LE, <=);
DEFINE_CHECK_OP_IMPL(Check_LT, <);
DEFINE_CHECK_OP_IMPL(Check_GE, >=);
DEFINE_CHECK_OP_IMPL(Check_GT, >);

#undef DEFINE_CHECK_OP_IMPL

// clang-format off
#define TM_CHECK(e)                                                                  \
    if (TM_UNLIKELY(!(e))) turbomind::core::CheckErrorStream(__FILE__, __LINE__, #e)

#define TM_CHECK_OP(name, op, a, b)                                                  \
    if (auto&& [__p, __s] = turbomind::core::Check##name##Impl(a, b); __p) \
        turbomind::core::CheckErrorStream(__FILE__, __LINE__, #a " " #op " " #b, __s)
// clang-format on

#if TM_DISABLE_CHECK_OP

#define TM_CHECK_EQ(a, b) TM_CHECK(a == b)
#define TM_CHECK_NE(a, b) TM_CHECK(a != b)
#define TM_CHECK_LE(a, b) TM_CHECK(a <= b)
#define TM_CHECK_LT(a, b) TM_CHECK(a < b)
#define TM_CHECK_GE(a, b) TM_CHECK(a >= b)
#define TM_CHECK_GT(a, b) TM_CHECK(a > b)

#else

#define TM_CHECK_EQ(a, b) TM_CHECK_OP(_EQ, ==, a, b)
#define TM_CHECK_NE(a, b) TM_CHECK_OP(_NE, !=, a, b)
#define TM_CHECK_LE(a, b) TM_CHECK_OP(_LE, <=, a, b)
#define TM_CHECK_LT(a, b) TM_CHECK_OP(_LT, <, a, b)
#define TM_CHECK_GE(a, b) TM_CHECK_OP(_GE, >=, a, b)
#define TM_CHECK_GT(a, b) TM_CHECK_OP(_GT, >, a, b)

#endif

[[noreturn]] void ReportNullError(const char* file, int line, const char* expr);

template<class T>
decltype(auto) EnsureNotNull(const char* file, int line, const char* expr, T&& p)
{
    if (TM_UNLIKELY(p == nullptr)) {
        ReportNullError(file, line, expr);
    }
    return (T &&) p;
}

#define TM_CHECK_NOTNULL(p) ::turbomind::core::EnsureNotNull(__FILE__, __LINE__, #p, (p))

}  // namespace turbomind::core
