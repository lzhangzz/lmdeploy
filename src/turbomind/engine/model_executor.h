#include <memory>

#include "src/turbomind/core/core.h"

#include "src/turbomind/engine/batch_data.h"
#include "src/turbomind/engine/queue.h"
#include "src/turbomind/models/language_model.h"

namespace turbomind {

class ModelExecutor {
public:
    ~ModelExecutor();

    ModelExecutor();
    ModelExecutor(ModelExecutor&&) noexcept;
    ModelExecutor& operator=(ModelExecutor&&) noexcept;

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    ModelExecutor(LanguageModel&                     model,
                  Queue<std::unique_ptr<BatchData>>& inbound,
                  Queue<std::unique_ptr<BatchData>>& outbound);

    void Start();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
