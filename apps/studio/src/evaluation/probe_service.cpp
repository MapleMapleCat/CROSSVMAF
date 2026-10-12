#include "evaluation/probe_service.hpp"

#include <utility>
#include <cstdio>

namespace crossvmaf::studio {

ProbeService::ProbeService(Callback callback)
    : callback_(std::move(callback)) {
    worker_ = std::thread([this] { worker_main(); });
}

ProbeService::~ProbeService() {
    {
        std::lock_guard lock(mutex_);
        shutdown_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
}

std::uint64_t ProbeService::probe(
    InputRole role,
    std::filesystem::path path,
    std::uint64_t slot) {
    std::uint64_t version;
    {
        std::lock_guard lock(mutex_);
        version = next_version_++;
        // Supersede any queued request for the same slot.
        for (auto it = pending_.begin(); it != pending_.end();) {
            it = it->role == role && it->slot == slot ? pending_.erase(it)
                                                       : std::next(it);
        }
        pending_.push_back({version, role, slot, std::move(path)});
    }
    condition_.notify_all();
    return version;
}

void ProbeService::worker_main() noexcept {
    for (;;) {
        Request request;
        {
            std::unique_lock lock(mutex_);
            condition_.wait(lock, [this] {
                return shutdown_ || !pending_.empty();
            });
            if (shutdown_) return;
            request = std::move(pending_.front());
            pending_.pop_front();
        }
        try {
            ProbeOutcome outcome{
                request.version, request.role, request.slot,
                probe_video(request.path)};
            callback_(std::move(outcome));
        } catch (const std::exception& exception) {
            try {
                callback_({request.version, request.role, request.slot,
                    Result<VideoMetadata>::failure({ErrorCode::backend_error, exception.what()})});
            } catch (...) {
                std::fputs("Studio could not deliver the probe failure\n", stderr);
            }
        } catch (...) {
            try {
                callback_({request.version, request.role, request.slot,
                    Result<VideoMetadata>::failure({ErrorCode::backend_error, "Unknown probe exception"})});
            } catch (...) {
                std::fputs("Studio could not deliver the probe failure\n", stderr);
            }
        }
    }
}

}  // namespace crossvmaf::studio
