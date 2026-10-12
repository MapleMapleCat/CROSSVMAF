#pragma once

// Background single-file probing for the New and Compare screens (P0-2 app
// side).
// Probing runs off the UI thread; every request carries a version so stale
// results from superseded file selections are dropped by the receiver.

#include "types.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <thread>

namespace crossvmaf::studio {

class ProbeService final {
public:
    struct ProbeOutcome final {
        std::uint64_t version{};
        InputRole role{InputRole::none};
        // Tells apart several inputs of one role; 0 on the New screen.
        std::uint64_t slot{};
        Result<VideoMetadata> metadata;
    };
    // Fires on the probe worker thread; marshal to the UI before touching
    // Slint.
    using Callback = std::function<void(ProbeOutcome)>;

    explicit ProbeService(Callback callback);
    ~ProbeService();

    ProbeService(const ProbeService&) = delete;
    ProbeService& operator=(const ProbeService&) = delete;

    // Queues a probe for one input and returns its version. Only the newest
    // request per role and slot is executed.
    std::uint64_t probe(InputRole role, std::filesystem::path path,
                        std::uint64_t slot = 0U);

private:
    void worker_main() noexcept;

    Callback callback_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    struct Request final {
        std::uint64_t version{};
        InputRole role{InputRole::none};
        std::uint64_t slot{};
        std::filesystem::path path;
    };
    std::deque<Request> pending_;
    std::uint64_t next_version_{1U};
    bool shutdown_{};
    std::thread worker_;
};

}  // namespace crossvmaf::studio
