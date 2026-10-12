#include "evaluation/comparison_setup.hpp"

#include "i18n/translator.hpp"
#include "reports/comparison_selection.hpp"

#include <algorithm>
#include <system_error>
#include <utility>

namespace crossvmaf::studio {
namespace {

// The first blocking issue of a pair check, or null when the pair can run.
[[nodiscard]] const PairIssue* blocking_issue(const PairValidation& validation) {
    if (validation.status != PairValidationStatus::invalid) return nullptr;
    const auto issue = std::find_if(
        validation.issues.begin(), validation.issues.end(),
        [](const PairIssue& candidate) {
            return candidate.severity == PairIssueSeverity::error;
        });
    return issue == validation.issues.end() ? nullptr : &*issue;
}

}  // namespace

bool same_file(const std::filesystem::path& left,
               const std::filesystem::path& right) {
    if (left.empty() || right.empty()) return false;
    std::error_code error;
    const bool equivalent = std::filesystem::equivalent(left, right, error);
    if (!error) return equivalent;
    return left.lexically_normal() == right.lexically_normal();
}

void ComparisonSetup::start(bool editing) {
    clear();
    active_ = true;
    editing_ = editing;
}

void ComparisonSetup::clear() {
    active_ = false;
    editing_ = false;
    reference_ = {};
    rows_.clear();
}

ModelVariant ComparisonSetup::model() const {
    return resolve_evaluation_config(
        reference_.metadata.has_value() ? &*reference_.metadata : nullptr).model;
}

void ComparisonSetup::set_reference(std::filesystem::path path,
                                    std::uint64_t probe_version) {
    const bool replaced = !reference_.path.empty() && !same_file(path, reference_.path);
    reference_ = {};
    reference_.path = std::move(path);
    reference_.probe_version = probe_version;
    if (!replaced) return;
    std::erase_if(rows_, [](const Row& row) {
        return row.from_library || !row.track_id.empty();
    });
    for (Row& row : rows_) row.saved.reset();
}

bool ComparisonSetup::apply_reference_probe(std::uint64_t version,
                                            Result<VideoMetadata> metadata) {
    if (reference_.path.empty() || version != reference_.probe_version) return false;
    if (metadata) {
        reference_.probe = ProbeState::ready;
        reference_.metadata = std::move(metadata).value();
        reference_.error.clear();
    } else {
        reference_.probe = ProbeState::failed;
        reference_.metadata.reset();
        reference_.error = metadata.error().message;
    }
    // Saved results were matched against the old reference snapshot.
    for (Row& row : rows_) {
        if (!row.from_library && row.track_id.empty()) row.saved.reset();
    }
    return true;
}

ComparisonSetup::AddOutcome ComparisonSetup::add_file(std::filesystem::path path) {
    if (same_file(path, reference_.path)) return AddOutcome::same_as_reference;
    if (has_path(path)) return AddOutcome::duplicate;
    if (rows_.size() >= ComparisonSelection::maximum_reports) return AddOutcome::full;
    Row row;
    row.id = next_row_id_++;
    row.path = std::move(path);
    rows_.push_back(std::move(row));
    return AddOutcome::added;
}

void ComparisonSetup::set_row_probe_version(std::uint64_t row_id, std::uint64_t version) {
    if (Row* row = find(row_id)) {
        row->probe = ProbeState::pending;
        row->probe_version = version;
    }
}

bool ComparisonSetup::apply_row_probe(std::uint64_t row_id, std::uint64_t version,
                                      Result<VideoMetadata> metadata) {
    Row* row = find(row_id);
    if (!row || version != row->probe_version) return false;
    row->saved.reset();
    if (metadata) {
        row->probe = ProbeState::ready;
        row->metadata = std::move(metadata).value();
        row->probe_error.clear();
    } else {
        row->probe = ProbeState::failed;
        row->metadata.reset();
        row->probe_error = metadata.error().message;
    }
    return true;
}

ComparisonSetup::AddOutcome ComparisonSetup::add_saved(
    std::shared_ptr<const EvaluationReport> report) {
    if (!report) return AddOutcome::duplicate;
    if (contains_report(report->report_id) || has_path(report->request.distorted_path)) {
        return AddOutcome::duplicate;
    }
    if (rows_.size() >= ComparisonSelection::maximum_reports) return AddOutcome::full;
    Row row;
    row.id = next_row_id_++;
    row.path = report->request.distorted_path;
    row.probe = ProbeState::ready;
    row.from_library = true;
    row.saved = std::move(report);
    rows_.push_back(std::move(row));
    return AddOutcome::added;
}

void ComparisonSetup::add_included(std::string track_id, std::filesystem::path path,
                                   std::shared_ptr<const EvaluationReport> report) {
    Row row;
    row.id = next_row_id_++;
    row.path = std::move(path);
    row.probe = ProbeState::ready;
    row.saved = std::move(report);
    row.track_id = std::move(track_id);
    rows_.push_back(std::move(row));
}

void ComparisonSetup::remove_row(std::uint64_t row_id) {
    std::erase_if(rows_, [row_id](const Row& row) { return row.id == row_id; });
}

void ComparisonSetup::set_saved(std::uint64_t row_id,
                                std::shared_ptr<const EvaluationReport> report) {
    Row* row = find(row_id);
    if (!row || row->from_library || !row->track_id.empty()) return;
    row->saved = std::move(report);
    row->reuse_saved = true;
}

void ComparisonSetup::toggle_reuse(std::uint64_t row_id) {
    if (Row* row = find(row_id); row && row->saved && !row->from_library &&
                                 row->track_id.empty()) {
        row->reuse_saved = !row->reuse_saved;
    }
}

bool ComparisonSetup::matches_saved(const Row& row, const EvaluationReport& report) const {
    if (report.frames.empty() || reference_.probe != ProbeState::ready ||
        !row.metadata.has_value() || report.model != model()) {
        return false;
    }
    return same_file(report.request.distorted_path, row.path) &&
        same_file(report.request.reference_path, reference_.path) &&
        report.request.distorted_identity.matches(row.metadata->identity) &&
        report.request.reference_identity.matches(reference_.metadata->identity);
}

bool ComparisonSetup::contains_report(const std::string& report_id) const {
    return std::any_of(rows_.begin(), rows_.end(), [&report_id](const Row& row) {
        return row.track_id == report_id ||
            (row.saved && row.saved->report_id == report_id);
    });
}

ComparisonSetup::RowStatus ComparisonSetup::status(const Row& row) const {
    if (!row.track_id.empty()) {
        if (row.saved && reference_.probe == ProbeState::ready && row.saved->model != model()) {
            return RowStatus::invalid;
        }
        return RowStatus::included;
    }
    if (row.from_library) {
        if (!same_file(row.saved->request.reference_path, reference_.path)) {
            return RowStatus::invalid;
        }
        if (reference_.probe != ProbeState::ready) return RowStatus::checking;
        return row.saved->model == model() ? RowStatus::saved : RowStatus::invalid;
    }
    if (same_file(row.path, reference_.path)) return RowStatus::invalid;
    if (row.probe == ProbeState::failed) return RowStatus::invalid;
    if (row.probe == ProbeState::pending || reference_.probe != ProbeState::ready) {
        return RowStatus::checking;
    }
    const auto validation = validate_video_pair(*reference_.metadata, *row.metadata);
    if (!validation || validation.value().status == PairValidationStatus::invalid) {
        return RowStatus::invalid;
    }
    return row.saved && row.reuse_saved ? RowStatus::saved : RowStatus::ready;
}

std::string ComparisonSetup::message(const Row& row) const {
    switch (status(row)) {
    case RowStatus::checking:
        return row.probe == ProbeState::pending
            ? i18n::tr("Checking the video…")
            : i18n::tr("Waiting for the reference video");
    case RowStatus::invalid:
        break;
    case RowStatus::ready: {
        // Issues the run itself verifies, worth knowing before it starts.
        const auto validation = validate_video_pair(*reference_.metadata, *row.metadata);
        if (!validation) return {};
        const auto& issues = validation.value().issues;
        const auto notable = std::find_if(issues.begin(), issues.end(),
            [](const PairIssue& issue) {
                return issue.severity == PairIssueSeverity::warning ||
                    issue.requires_runtime_validation;
            });
        return notable == issues.end() ? std::string{} : notable->message;
    }
    case RowStatus::saved:
    case RowStatus::included:
        return {};
    }
    if (row.from_library || !row.track_id.empty()) {
        return same_file(row.saved->request.reference_path, reference_.path)
            ? i18n::tr("Evaluated with a different quality model")
            : i18n::tr("Evaluated against another reference video");
    }
    if (same_file(row.path, reference_.path)) return i18n::tr("This is the reference video");
    if (row.probe == ProbeState::failed) {
        return i18n::tr("The video could not be read: {}", row.probe_error);
    }
    const auto validation = validate_video_pair(*reference_.metadata, *row.metadata);
    if (!validation) return validation.error().message;
    // Issue messages come from the core library and are not translated.
    if (const PairIssue* issue = blocking_issue(validation.value())) {
        return i18n::tr("{0} ({1} ↔ {2})", issue->message,
                        issue->reference_value, issue->distorted_value);
    }
    return i18n::tr("The video does not match the reference");
}

std::size_t ComparisonSetup::count(RowStatus wanted) const {
    return static_cast<std::size_t>(std::count_if(rows_.begin(), rows_.end(),
        [this, wanted](const Row& row) { return status(row) == wanted; }));
}

bool ComparisonSetup::ready() const {
    return active_ && reference_.probe == ProbeState::ready && rows_.size() >= 2U &&
        count(RowStatus::checking) == 0U && count(RowStatus::invalid) == 0U;
}

std::string ComparisonSetup::status_message() const {
    if (reference_.path.empty()) return i18n::tr("Choose the reference video first");
    if (reference_.probe == ProbeState::pending) {
        return i18n::tr("Checking the reference video…");
    }
    if (reference_.probe == ProbeState::failed) {
        return i18n::tr("The reference video could not be read: {}", reference_.error);
    }
    if (rows_.size() < 2U) return i18n::tr("Add at least two videos to compare");
    if (count(RowStatus::checking) > 0U) return i18n::tr("Checking the videos…");
    if (count(RowStatus::invalid) > 0U) {
        return i18n::tr("Remove the videos that cannot be compared");
    }
    std::vector<std::string> parts;
    if (const std::size_t evaluated = count(RowStatus::ready); evaluated > 0U) {
        parts.push_back(i18n::trn("{0} video to evaluate", "{0} videos to evaluate",
                                  evaluated, std::to_string(evaluated)));
    }
    if (const std::size_t saved = count(RowStatus::saved); saved > 0U) {
        parts.push_back(i18n::trn("{0} saved result reused", "{0} saved results reused",
                                  saved, std::to_string(saved)));
    }
    if (const std::size_t included = count(RowStatus::included); included > 0U) {
        parts.push_back(i18n::trn("{0} already in the comparison",
                                  "{0} already in the comparison",
                                  included, std::to_string(included)));
    }
    std::string summary;
    for (const std::string& part : parts) {
        if (!summary.empty()) summary += " · ";
        summary += part;
    }
    return summary;
}

ComparisonSetup::Row* ComparisonSetup::find(std::uint64_t row_id) {
    const auto row = std::find_if(rows_.begin(), rows_.end(),
        [row_id](const Row& candidate) { return candidate.id == row_id; });
    return row == rows_.end() ? nullptr : &*row;
}

bool ComparisonSetup::has_path(const std::filesystem::path& path) const {
    return std::any_of(rows_.begin(), rows_.end(), [&path](const Row& row) {
        return same_file(row.path, path);
    });
}

}  // namespace crossvmaf::studio
