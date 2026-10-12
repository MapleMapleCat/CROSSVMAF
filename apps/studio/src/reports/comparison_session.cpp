#include "reports/comparison_session.hpp"

#include <algorithm>
#include <utility>

namespace crossvmaf::studio {

bool task_state_active(TaskState state) noexcept {
    switch (state) {
    case TaskState::queued:
    case TaskState::opening:
    case TaskState::running:
    case TaskState::finishing:
    case TaskState::cancelling:
        return true;
    case TaskState::completed:
    case TaskState::failed:
    case TaskState::cancelled:
        return false;
    }
    return false;
}

void ComparisonSession::start(std::filesystem::path reference_path, SceneId scene) {
    clear();
    reference_path_ = std::move(reference_path);
    scene_ = scene;
}

void ComparisonSession::clear() {
    reference_path_.clear();
    scene_ = SceneId::tv_standard;
    selection_.clear();
    pending_.clear();
}

std::size_t ComparisonSession::active_count() const {
    return static_cast<std::size_t>(std::count_if(pending_.begin(), pending_.end(),
        [](const PendingTrack& track) { return task_state_active(track.state); }));
}

bool ComparisonSession::owns_task(std::string_view task_id) const {
    return std::any_of(pending_.begin(), pending_.end(),
        [task_id](const PendingTrack& track) { return track.task_id == task_id; });
}

void ComparisonSession::add_pending(std::string task_id,
                                    std::filesystem::path distorted_path) {
    PendingTrack track;
    track.task_id = std::move(task_id);
    track.distorted_path = std::move(distorted_path);
    pending_.push_back(std::move(track));
}

bool ComparisonSession::update_task(const TaskSnapshot& snapshot) {
    PendingTrack* track = find_pending(snapshot.task_id);
    // A completed task is handled by complete(), which may keep the track
    // listed with the reason it could not join.
    if (!track || snapshot.state == TaskState::completed) return false;
    track->state = snapshot.state;
    track->error = snapshot.error;
    track->issue.reset();
    if (snapshot.state == TaskState::queued) track->progress.reset();
    return true;
}

bool ComparisonSession::update_progress(std::string_view task_id,
                                        const ProgressSnapshot& progress) {
    PendingTrack* track = find_pending(task_id);
    if (!track) return false;
    if (progress.frame_count_hint.has_value() && *progress.frame_count_hint > 0U) {
        track->progress = std::min(1.0,
            static_cast<double>(progress.completed_frames) /
                static_cast<double>(*progress.frame_count_hint));
    } else {
        track->progress.reset();
    }
    return true;
}

bool ComparisonSession::complete(std::shared_ptr<const EvaluationReport> report) {
    if (!report) return false;
    PendingTrack* track = find_pending(report->report_id);
    if (!track) return false;
    if (auto issue = selection_.add(report)) {
        track->state = TaskState::completed;
        track->progress = 1.0;
        track->issue = std::move(issue);
        return true;
    }
    const std::string task_id = track->task_id;
    std::erase_if(pending_, [&task_id](const PendingTrack& pending) {
        return pending.task_id == task_id;
    });
    return true;
}

void ComparisonSession::remove(std::string_view id) {
    const std::string removed(id);
    selection_.remove(removed);
    std::erase_if(pending_, [&removed](const PendingTrack& track) {
        return track.task_id == removed;
    });
}

ComparisonSession::PendingTrack* ComparisonSession::find_pending(std::string_view task_id) {
    const auto track = std::find_if(pending_.begin(), pending_.end(),
        [task_id](const PendingTrack& candidate) { return candidate.task_id == task_id; });
    return track == pending_.end() ? nullptr : &*track;
}

}  // namespace crossvmaf::studio
