#pragma once

// Draft of the Compare screen's set-up step: one reference video and the
// videos to compare with it. Probing and the saved-report lookup run in the
// controller; this class records their outcomes, checks every video against
// the reference, and says whether the draft can start. Messages are rendered
// on demand, so a language change only needs the draft to be shown again.

#include "evaluation/scene_policy.hpp"
#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace crossvmaf::studio {

class ComparisonSetup final {
public:
    enum class ProbeState : std::uint8_t {
        pending,
        ready,
        failed,
    };

    struct Reference final {
        std::filesystem::path path;
        ProbeState probe{ProbeState::pending};
        std::uint64_t probe_version{};
        std::optional<VideoMetadata> metadata;
        std::string error;
    };

    enum class RowStatus : std::uint8_t {
        // Waiting for its own probe or for the reference.
        checking,
        // Will be evaluated.
        ready,
        // A saved result of this video is used instead of a new evaluation.
        saved,
        // Already part of the comparison being edited.
        included,
        // Cannot be compared; message() says why.
        invalid,
    };

    struct Row final {
        // Stable within the draft; also the probe slot of the video.
        std::uint64_t id{};
        std::filesystem::path path;
        ProbeState probe{ProbeState::pending};
        std::uint64_t probe_version{};
        std::optional<VideoMetadata> metadata;
        std::string probe_error;
        // A finished evaluation of this video: the saved result that
        // matches it, or the report of an included track.
        std::shared_ptr<const EvaluationReport> saved;
        bool reuse_saved{true};
        // Added from the saved evaluations list; never probed.
        bool from_library{};
        // Report or task id of the comparison track this row stands for.
        std::string track_id;
    };

    enum class AddOutcome : std::uint8_t {
        added,
        duplicate,
        full,
        same_as_reference,
    };

    // Starts a new draft; editing keeps the reference of the comparison.
    void start(bool editing);
    void clear();
    [[nodiscard]] bool active() const noexcept { return active_; }
    [[nodiscard]] bool editing() const noexcept { return editing_; }

    [[nodiscard]] const Reference& reference() const noexcept { return reference_; }
    [[nodiscard]] const std::vector<Row>& rows() const noexcept { return rows_; }
    [[nodiscard]] SceneId scene() const { return model_scene(model()); }
    // The model evaluations of this draft would use.
    [[nodiscard]] ModelVariant model() const;

    // Replacing the reference drops rows that only made sense for the old
    // one: saved results and included tracks.
    void set_reference(std::filesystem::path path, std::uint64_t probe_version);
    // Stale versions are ignored; returns whether the draft changed.
    bool apply_reference_probe(std::uint64_t version, Result<VideoMetadata> metadata);

    [[nodiscard]] AddOutcome add_file(std::filesystem::path path);
    // The id of the last added row.
    [[nodiscard]] std::uint64_t last_row_id() const noexcept { return next_row_id_ - 1U; }
    void set_row_probe_version(std::uint64_t row_id, std::uint64_t version);
    bool apply_row_probe(std::uint64_t row_id, std::uint64_t version,
                         Result<VideoMetadata> metadata);
    [[nodiscard]] AddOutcome add_saved(std::shared_ptr<const EvaluationReport> report);
    // A track of the comparison being edited; `report` is empty while it is
    // still being evaluated.
    void add_included(std::string track_id, std::filesystem::path path,
                      std::shared_ptr<const EvaluationReport> report);
    void remove_row(std::uint64_t row_id);
    void set_saved(std::uint64_t row_id, std::shared_ptr<const EvaluationReport> report);
    void toggle_reuse(std::uint64_t row_id);

    // Whether a saved report evaluated this row's file, unchanged, against
    // this draft's reference with this draft's model.
    [[nodiscard]] bool matches_saved(const Row& row, const EvaluationReport& report) const;
    [[nodiscard]] bool contains_report(const std::string& report_id) const;

    [[nodiscard]] RowStatus status(const Row& row) const;
    // Why the row cannot be compared, or what it is waiting for.
    [[nodiscard]] std::string message(const Row& row) const;
    [[nodiscard]] std::size_t count(RowStatus status) const;
    [[nodiscard]] bool ready() const;
    // What stops the draft, or a summary of what starting it will do.
    [[nodiscard]] std::string status_message() const;

private:
    [[nodiscard]] Row* find(std::uint64_t row_id);
    [[nodiscard]] bool has_path(const std::filesystem::path& path) const;

    bool active_{};
    bool editing_{};
    Reference reference_;
    std::vector<Row> rows_;
    std::uint64_t next_row_id_{1U};
};

// Same file, falling back to comparing normalized paths when the file is
// gone.
[[nodiscard]] bool same_file(const std::filesystem::path& left,
                             const std::filesystem::path& right);

}  // namespace crossvmaf::studio
