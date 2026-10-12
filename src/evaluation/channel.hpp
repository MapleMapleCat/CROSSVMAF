#pragma once
#if defined(CROSSVMAF_EVALUATION_IMPLEMENTATION) || defined(CROSSVMAF_EVALUATION_TESTING)
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace crossvmaf::evaluation {

enum class ChannelReceiveStatus {
    value,
    unavailable,
    closed,
};

template <typename Value> struct ChannelReceiveResult final {
    ChannelReceiveStatus status{ChannelReceiveStatus::unavailable};
    std::optional<Value> value;
};

// A closeable bounded MPMC channel. Closing preserves already queued values,
// wakes every blocked sender and receiver, and rejects future sends.
template <typename Value> class BlockingBoundedChannel final {
  public:
    explicit BlockingBoundedChannel(std::size_t capacity) : capacity_(capacity) {
        if (capacity_ == 0U) {
            throw std::invalid_argument("blocking channel capacity must be greater than zero");
        }
    }

    BlockingBoundedChannel(const BlockingBoundedChannel &) = delete;
    BlockingBoundedChannel &operator=(const BlockingBoundedChannel &) = delete;

    [[nodiscard]] bool send(Value value) {
        std::unique_lock lock(mutex_);
        not_full_.wait(lock, [this] { return values_.size() < capacity_ || closed_; });
        if (closed_) return false;
        values_.push_back(std::move(value));
        lock.unlock();
        not_empty_.notify_one();
        return true;
    }

    [[nodiscard]] ChannelReceiveResult<Value> try_receive() {
        std::lock_guard lock(mutex_);
        return receive_locked();
    }

    [[nodiscard]] ChannelReceiveResult<Value> receive() {
        std::unique_lock lock(mutex_);
        not_empty_.wait(lock, [this] { return !values_.empty() || closed_; });
        return receive_locked();
    }

    void close() noexcept {
        {
            std::lock_guard lock(mutex_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    [[nodiscard]] bool closed() const noexcept {
        std::lock_guard lock(mutex_);
        return closed_;
    }

  private:
    [[nodiscard]] ChannelReceiveResult<Value> receive_locked() {
        if (values_.empty()) {
            return {closed_ ? ChannelReceiveStatus::closed : ChannelReceiveStatus::unavailable,
                    std::nullopt};
        }

        Value value = std::move(values_.front());
        values_.pop_front();
        not_full_.notify_one();
        return {ChannelReceiveStatus::value, std::optional<Value>{std::move(value)}};
    }

    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<Value> values_;
    bool closed_{};
};

} // namespace crossvmaf::evaluation
#endif
