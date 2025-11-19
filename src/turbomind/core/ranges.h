#pragma once

namespace turbomind::core {

template<class Iterator>
class subrange {
public:
    subrange(Iterator first, Iterator last): first_{first}, last_{last} {}

    Iterator begin()
    {
        return first_;
    }

    Iterator end()
    {
        return last_;
    }

private:
    Iterator first_;
    Iterator last_;
};

}  // namespace turbomind::core