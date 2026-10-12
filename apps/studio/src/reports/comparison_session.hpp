#pragma once

// The comparison on the Compare screen: the finished reports it compares
// (ComparisonSelection) plus the videos still being evaluated for it. A
// finished evaluation joins the selection when it is comparable; otherwise
// it stays listed with the reason, like a failed one, until it is removed.

#include "reports/comparison_selection.hpp"
#include "types.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace crossvmaf::studio {

class ComparisonSession final {
public:
    struct PendingTrack final {
        std::string task_id;
        std::filesystem::path distorted_path;
        TaskState state{TaskState::queued};
        // Completed share of the frames, when the frame count is known.
        std::optional<double> progress;
        std::optional<TaskError> error;
        // Why the finished report could not join the comparison.
        std::optional<ComparisonIssue> issue;
    };

    // Starts over with a reference and scene and no tracks.
    void start(std::filesystem::path reference_path, SceneId scene);
    void clear();

    [[nodiscard]] bool empty() const noexcept {
        return selection_.empty() && pending_.empty();
    }
    [[nodiscard]] std::size_t size() const noexcept {
        return selection_.entries().size() + pending_.size();
    }
    [[nodiscard]] const std::filesystem::path& reference_path() const noexcept {
        return reference_path_;
    }
    [[nodiscard]] SceneId scene() const noexcept { return scene_; }
    [[nodiscard]] ComparisonSelection& selection() noexcept { return selection_; }
    [[nodiscard]] const ComparisonSelection& selection() const noexcept {
        return selection_;
    }
    [[nodiscard]] const std::vector<PendingTrack>& pending() const noexcept {
        return pending_;
    }
    // Pending tracks that are waiting or being evaluated.
    [[nodiscard]] std::size_t active_count() const;
    [[nodiscard]] bool owns_task(std::string_view task_id) const;

    void add_pending(std::string task_id, std::filesystem::path distorted_path);
    // Each returns whether a pending track changed.
    bool update_task(const TaskSnapshot& snapshot);
    bool update_progress(std::string_view task_id, const ProgressSnapshot& progress);
    // Moves the finished report of a pending track into the selection;
    // false when the report is not one of this session's tasks.
    bool complete(std::shared_ptr<const EvaluationReport> report);
    // Removes a finished report or a pending track by its id.
    void remove(std::string_view id);

private:
    [[nodiscard]] PendingTrack* find_pending(std::string_view task_id);

    std::filesystem::path reference_path_;
    SceneId scene_{SceneId::tv_standard};
    ComparisonSelection selection_;
    std::vector<PendingTrack> pending_;
};

[[nodiscard]] bool task_state_active(TaskState state) noexcept;

}  // namespace crossvmaf::studio
