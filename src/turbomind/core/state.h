
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/core/layout.h"
#include "src/turbomind/core/tensor.h"
#include <algorithm>

namespace turbomind {

// Goals:
// 1. constant number of cudaMemcpy / kernel launches
// 2. single stream synchronization / iteration

template<class T>
struct ConstState_ {
    Tensor_<T> h_state;
    Tensor_<T> h_buf;
    Tensor_<T> d_buf;

    ssize_t size;

    ssize_t vec_size;

    explicit ConstState_(Layout layout)
    {
        h_state  = {layout, kCPUpinned};
        h_buf    = {layout, kCPUpinned};
        d_buf    = {layout, kDEVICE};
        size     = 0;
        vec_size = layout.stride(0);
    }

    void Set(int i, const T* value)
    {
        std::copy_n(value, vec_size, h_state.data() + i * vec_size);
    }

    T& operator[](int i)
    {
        return h_state.data() + i * vec_size;
    }

    void Reorder(const Buffer_<int>& h_perm)
    {
        const auto* src = h_state.data();
        auto*       dst = h_buf.data();
        for (int i = 0; i < h_perm.size(); ++i) {
            std::copy_n(src + h_perm[i] * vec_size, vec_size, dst + i * vec_size);
        }
        size = h_perm.size();
        std::swap(h_buf, h_state);
    }

    void Release()
    {
        Copy_(h_state.buffer(), size * vec_size, d_buf.buffer());
    }
};

void MaskedGather(const Tensor& src, const Buffer_<int>& perm, const Buffer_<int>& mask, const Tensor& dst);

template<class T>
struct MutableState_ {
    Tensor_<T> h_state;
    Tensor_<T> d_state;
    Tensor_<T> h_buf;
    Tensor_<T> d_buf;

    ssize_t size;
    ssize_t vec_size;

    explicit MutableState_(Layout shape)
    {
        h_state = {shape, kCPUpinned};
        h_buf   = {shape, kCPUpinned};
        d_state = {shape, kDEVICE};
        d_buf   = {shape, kDEVICE};

        vec_size = h_state.stride(0);
    }

    T& operator[](int i)
    {
        return h_state.data() + i * vec_size;
    }

    void Reorder_0(const Buffer_<int>& h_perm)
    {
        const auto* src = h_state.data();
        auto*       dst = h_buf.data();

        for (int i = 0; i < h_perm.size(); ++i) {
            std::copy_n(src + h_perm[i] * vec_size, vec_size, dst + i * vec_size);
        }

        std::swap(h_buf, h_state);
    }

    void Acquire_0(ssize_t size)
    {
        Copy_(d_buf, size * vec_size, h_buf);

        core::Context::stream().Sync();

        // const auto src = h_buf.data();
        // auto       dst = h_state.data();

        // for (int i = 0; i < h_perm.size(); ++i) {
        //     std::copy_n(src + h_perm[i] * vec_size, vec_size, dst + i * vec_size);
        // }
    }

    void Patch_0(const Buffer_<int>& h_perm, const Buffer_<int>& h_mask)
    {
        TM_CHECK_EQ(h_perm.size(), h_mask.size());
    }

    void Release_0(ssize_t size)
    {
        Copy_(h_state, vec_size * size, d_buf);
    }

    void Acquire_1(ssize_t size) {}

    // void Acquire_1(const Buffer_<int>& d_perm)
    // {
    //     MaskedGather(d_state, d_perm, 0, d_buf);
    //     std::swap(d_state, d_buf);
    // }

    void Patch_1(const Buffer_<int>& d_perm, const Buffer_<int>& d_mask) {}

    void Release_1(ssize_t size)
    {
        Copy_(d_state, size * vec_size, d_buf);
    }
};

// input embeddings, mrope position ids
template<class T>
struct ConstVarlenState_ {
    Tensor_<T> d_state;
    Buffer_<T> h_state_offset;
    Buffer_<T> d_state_offset;
    // Tensor_<T>   h_new;
    Tensor_<T> d_buf;
    Buffer_<T> h_buf_offset;
    Buffer_<T> d_buf_offset;

    // Buffer_<int> d_state_len;
    // Buffer_<int> d_buf_len;
    // Buffer_<int> h_state_len;
    // Buffer_<int> h_buf_len;

    size_t elem_size;

    ConstVarlenState_(Layout layout, int max_slots)
    {
        d_state = {layout, kDEVICE};
        d_buf   = {layout, kDEVICE};

        elem_size = layout.stride(0);
    }

    void Set(int i, const T* value, int n)
    {
        // set data
        Copy_(value, n * elem_size, d_state.data() + i * elem_size);

        // set length
        h_state_offset[i + 1] = h_state_offset[i] + n;
    }

    void Reorder_0(const Buffer_<int> h_perm, const Buffer_<int>& d_perm)
    {
        Copy(h_state_offset, d_perm.size() + 10, d_state_offset);

        h_buf_offset[0] = 0;
        for (int i = 0; i < h_perm.size(); ++i) {
            const int j         = h_perm[i];
            const int n         = h_state_offset[j + 1] - h_state_offset[j + 1];
            h_buf_offset[i + 1] = h_buf_offset[i] + n;
        }

        Copy(h_buf_offset, h_perm.size() + 1, d_buf_offset);

        // kernel launch
        // Reorder(h_buf_offset.)

        std::swap(d_state, d_buf);
        std::swap(d_state_offset, d_buf_offset);
        std::swap(h_state_offset, h_buf_offset);
    }

    void Release_0(ssize_t) {}  // NOP
};

template<class T>
struct MutableState0_ {
    Tensor_<T> front;
    Tensor_<T> back;

    ssize_t vec_size;

    Tensor_<T>* operator->()
    {
        return &front;
    }

    void Reorder(const Buffer_<int>& perm)
    {
        for (int i = 0; i < perm.size(); ++i) {
            back[i] = front[perm[i]];
        }
        std::swap(front, back);
    }

    void Release(Tensor_<T>& outgoing, ssize_t size)
    {
        Copy(front, size * vec_size, outgoing);
    }

    void Acquire(const Tensor_<T>& state, ssize_t size)
    {
        Copy(state, size * vec_size, back);
    }

    void Patch(const Buffer_<int>& perm, const Buffer_<int>& mask) {}
};

template<class T>
struct MutableState1_ {
    Tensor_<T> data;

    void Patch(Tensor_<T>& incoming, const Buffer_<int>& perm, const Buffer_<int>& mask)
    {
        // MaskedGather
        std::swap(incoming, data);
    }

    void Release(Tensor_<T>& outgoing)
    {
        Copy(data, outgoing);
    }
};

template<class T>
struct ConstVarlenState0_ {
    Tensor_<T>   data_0;
    Tensor_<T>   data_1;
    Buffer_<int> h_offset_0;
    Buffer_<int> h_offset_1;
    // Buffer_<int> d_offset;

    ssize_t vec_size;

    ssize_t size;

    void Add(int i, const T* value, int n)
    {
        // set data
        Copy_(value, n * vec_size, data_0.data() + h_offset_0[i] * vec_size);

        // set length
        h_offset_0[i + 1] = h_offset_0[i] + n;

        size = i + 1;
    }

    void Reorder(const Buffer_<int> h_perm)
    {
        
    }
};

template<class T>
struct MutableVarlenState_ {
    Tensor_<T> d_state;
    Tensor_<T> d_buf;
};

}  // namespace turbomind