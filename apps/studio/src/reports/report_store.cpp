#include "reports/report_store.hpp"

#include "i18n/translator.hpp"
#include "reports/report_dto.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>
#include <sstream>
#include <system_error>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace crossvmaf::studio {
namespace {

[[nodiscard]] std::string path_to_utf8(const std::filesystem::path& path) {
    const std::u8string utf8 = path.u8string();
    return std::string(
        reinterpret_cast<const char*>(utf8.data()), utf8.size());
}

[[nodiscard]] Error io_failure(
    std::string message,
    const std::filesystem::path& path) {
    message += ": ";
    message += path_to_utf8(path);
    return {ErrorCode::io_error, std::move(message)};
}

// Writes content through a sibling temporary file and renames it over the
// destination, so readers never observe a partially written record.
[[nodiscard]] Result<void> write_file_atomically(
    const std::filesystem::path& destination,
    const std::string& content) {
    const std::filesystem::path temporary =
        destination.parent_path() / (destination.filename().string() + ".tmp");
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            return Result<void>::failure(
                io_failure(i18n::tr("Cannot create a temporary file"), temporary));
        }
        output << content;
        output.close();
        if (!output) {
            return Result<void>::failure(
                io_failure(i18n::tr("Failed to write a temporary file"), temporary));
        }
    }
    std::error_code error;
#ifdef _WIN32
    const bool replaced = MoveFileExW(temporary.c_str(), destination.c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    std::filesystem::rename(temporary, destination, error);
    const bool replaced = !error;
#endif
    if (!replaced) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary, cleanup_error);
        return Result<void>::failure(
            io_failure(i18n::tr("Failed to save the report file"), destination));
    }
    return Result<void>::success();
}

[[nodiscard]] std::string file_stem_of(const std::filesystem::path& path) {
    const std::filesystem::path filename = path.filename();
    const std::u8string utf8 =
        (filename.empty() ? path : filename).u8string();
    return std::string(
        reinterpret_cast<const char*>(utf8.data()), utf8.size());
}

[[nodiscard]] std::string_view scene_key(SceneId scene) noexcept {
    switch (scene) {
    case SceneId::tv_standard: return "tv_standard";
    case SceneId::tv_4k_close: return "tv_4k_close";
    case SceneId::tv_4k_standard: return "tv_4k_standard";
    case SceneId::cinema_far: return "cinema_far";
    }
    return "tv_standard";
}

[[nodiscard]] SceneId parse_scene_key(std::string_view key) noexcept {
    if (key == "tv_4k_close") return SceneId::tv_4k_close;
    if (key == "tv_4k_standard") return SceneId::tv_4k_standard;
    if (key == "cinema_far") return SceneId::cinema_far;
    return SceneId::tv_standard;
}

[[nodiscard]] ReportSummary summarize(const EvaluationReport& report) {
    ReportSummary summary;
    summary.report_id = report.report_id;
    summary.distorted_name = file_stem_of(report.request.distorted_path);
    summary.reference_name = file_stem_of(report.request.reference_path);
    summary.pooled_score = report.pooled_score;
    summary.completed_at_utc_seconds =
        std::chrono::duration_cast<std::chrono::seconds>(
            report.completed_at.time_since_epoch()).count();
    summary.frame_count = report.frames.size();
    summary.model = report.model;
    summary.scene = report.request.scene;
    return summary;
}

}  // namespace

ReportStore::ReportStore(std::filesystem::path directory)
    : directory_(std::move(directory)) {}

std::filesystem::path ReportStore::report_path(
    const std::string& report_id) const {
    return directory_ / (report_id + ".json");
}

std::filesystem::path ReportStore::index_path() const {
    return directory_ / "index.json";
}

std::filesystem::path ReportStore::comparison_path() const {
    return directory_ / "comparison.json";
}

std::string ReportStore::generate_report_id() const {
    const auto now = std::chrono::system_clock::now();
    const std::time_t raw_time = std::chrono::system_clock::to_time_t(now);
    std::tm utc_time{};
#ifdef _WIN32
    gmtime_s(&utc_time, &raw_time);
#else
    gmtime_r(&raw_time, &utc_time);
#endif
    char timestamp[24]{};
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M%S", &utc_time);

    std::random_device random;
    std::mt19937 generator(random());
    std::uniform_int_distribution<unsigned> digits(0U, 15U);
    std::string id = std::string("r-") + timestamp + "-";
    for (int index = 0; index < 6; ++index) {
        static constexpr char hex[] = "0123456789abcdef";
        id.push_back(hex[digits(generator)]);
    }
    return id;
}

Result<std::vector<ReportSummary>> ReportStore::read_index() const {
    std::error_code error;
    if (!std::filesystem::exists(index_path(), error) && !error)
        return Result<std::vector<ReportSummary>>::success({});
    std::ifstream input(index_path(), std::ios::binary);
    if (!input) {
        return Result<std::vector<ReportSummary>>::failure(
            io_failure("Cannot read the report index", index_path()));
    }
    nlohmann::json document;
    try {
        document = nlohmann::json::parse(input);
        std::vector<ReportSummary> entries;
        if (!document.is_object() || !document.contains("reports") ||
            !document["reports"].is_array()) {
            throw std::runtime_error("expected an object containing a reports array");
        }
        for (const nlohmann::json& entry : document["reports"]) {
            ReportSummary summary;
            summary.report_id = entry.value("report_id", std::string{});
            if (summary.report_id.empty()) throw std::runtime_error("missing report_id");
            summary.distorted_name = entry.value("distorted_name", std::string{});
            summary.reference_name = entry.value("reference_name", std::string{});
            summary.pooled_score = entry.value("pooled_score", 0.0);
            summary.completed_at_utc_seconds =
                entry.value("completed_at_utc_seconds", std::int64_t{});
            summary.frame_count = entry.value("frame_count", std::uint64_t{});
            if (const auto model =
                    parse_model(entry.value("model", std::string_view{}));
                model) {
                summary.model = model.value();
            }
            summary.scene = parse_scene_key(
                entry.value("scene_id", std::string{"tv_standard"}));
            entries.push_back(std::move(summary));
        }
        return Result<std::vector<ReportSummary>>::success(std::move(entries));
    } catch (const std::exception& exception) {
        return Result<std::vector<ReportSummary>>::failure(io_failure(
            i18n::tr("The report index is corrupt and cannot be parsed") +
            ": " + exception.what(), index_path()));
    }
}

Result<void> ReportStore::write_index(
    const std::vector<ReportSummary>& entries) const {
    nlohmann::json document;
    document["index_version"] = 1;
    document["reports"] = nlohmann::json::array();
    for (const ReportSummary& summary : entries) {
        document["reports"].push_back({
            {"report_id", summary.report_id},
            {"distorted_name", summary.distorted_name},
            {"reference_name", summary.reference_name},
            {"pooled_score", summary.pooled_score},
            {"completed_at_utc_seconds", summary.completed_at_utc_seconds},
            {"frame_count", summary.frame_count},
            {"model", to_string(summary.model)},
            {"scene_id", scene_key(summary.scene)},
        });
    }
    return write_file_atomically(index_path(), document.dump(2));
}

Result<std::string> ReportStore::save_report(
    const EvaluationReport& report) try {
    std::error_code error;
    std::filesystem::create_directories(directory_, error);
    if (error) {
        return Result<std::string>::failure(
            io_failure(i18n::tr("Cannot create the report folder"), directory_));
    }

    auto index_entries = read_index();
    if (!index_entries) return Result<std::string>::failure(index_entries.error());

    EvaluationReport stored = report;
    if (stored.report_id.empty() ||
        std::filesystem::exists(report_path(stored.report_id), error)) {
        for (int attempt = 0; attempt < 8; ++attempt) {
            const std::string candidate = generate_report_id();
            if (!std::filesystem::exists(report_path(candidate), error)) {
                stored.report_id = candidate;
                break;
            }
        }
        if (stored.report_id.empty() ||
            std::filesystem::exists(report_path(stored.report_id), error)) {
            return Result<std::string>::failure(
                {ErrorCode::backend_error, i18n::tr("Cannot allocate a unique report ID")});
        }
    }

    const nlohmann::json document = evaluation_report_to_json(stored);
    const auto written = write_file_atomically(
        report_path(stored.report_id), document.dump(2));
    if (!written) {
        return Result<std::string>::failure(written.error());
    }

    auto& entries = index_entries.value();
    entries.erase(
        std::remove_if(
            entries.begin(), entries.end(),
            [&stored](const ReportSummary& summary) {
                return summary.report_id == stored.report_id;
            }),
        entries.end());
    entries.insert(entries.begin(), summarize(stored));
    const auto index_written = write_index(entries);
    if (!index_written) {
        return Result<std::string>::failure(index_written.error());
    }
    return Result<std::string>::success(stored.report_id);
} catch (const std::exception& exception) {
    return Result<std::string>::failure({ErrorCode::io_error,
        std::string("saving report: ") + exception.what()});
}

Result<ReportListPage> ReportStore::list_reports(
    std::size_t cursor,
    std::size_t limit) const {
    auto entries = read_index();
    if (!entries) {
        return Result<ReportListPage>::failure(entries.error());
    }
    ReportListPage page;
    const auto& all = entries.value();
    if (cursor >= all.size()) return Result<ReportListPage>::success(page);
    const std::size_t end = limit == 0U
        ? all.size()
        : std::min(all.size(), cursor + limit);
    for (std::size_t index = cursor; index < end; ++index) {
        std::error_code error;
        if (!std::filesystem::exists(report_path(all[index].report_id),
                                     error)) {
            page.corrupt_report_ids.push_back(all[index].report_id);
        }
        page.items.push_back(all[index]);
    }
    if (end < all.size()) page.next_cursor = end;
    return Result<ReportListPage>::success(std::move(page));
}

Result<EvaluationReport> ReportStore::load_report(
    const std::string& report_id) const {
    const std::filesystem::path path = report_path(report_id);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return Result<EvaluationReport>::failure(
            io_failure(i18n::tr("The report file is missing or cannot be opened"), path));
    }
    nlohmann::json document;
    try {
        document = nlohmann::json::parse(input);
    } catch (const std::exception& exception) {
        return Result<EvaluationReport>::failure(
            {ErrorCode::unsupported_format,
             i18n::tr("The report file is corrupt; JSON parsing failed: {} ({})",
                      path_to_utf8(path), exception.what())});
    }
    auto report = evaluation_report_from_json(document);
    if (!report) {
        Error error = report.error();
        error.message += " (" + path_to_utf8(path) + ")";
        return Result<EvaluationReport>::failure(std::move(error));
    }
    report.value().report_id = report_id;
    return report;
}

Result<void> ReportStore::delete_report(const std::string& report_id) {
    auto index_entries = read_index();
    if (!index_entries) return Result<void>::failure(index_entries.error());
    auto& entries = index_entries.value();
    const auto before = entries.size();
    entries.erase(
        std::remove_if(
            entries.begin(), entries.end(),
            [&report_id](const ReportSummary& summary) {
                return summary.report_id == report_id;
            }),
        entries.end());
    if (entries.size() == before &&
        !std::filesystem::exists(report_path(report_id))) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, i18n::tr("Report not found: {}", report_id)});
    }
    std::error_code error;
    std::filesystem::remove(report_path(report_id), error);
    if (error) {
        return Result<void>::failure(
            io_failure(i18n::tr("Failed to delete the report file"), report_path(report_id)));
    }
    return write_index(entries);
}

Result<void> ReportStore::save_comparison(const SavedComparison& comparison) const {
    std::error_code error;
    std::filesystem::create_directories(directory_, error);
    if (error) {
        return Result<void>::failure(
            io_failure(i18n::tr("Cannot create the report folder"), directory_));
    }
    nlohmann::json document;
    document["comparison_version"] = 1;
    document["reference_path"] = path_to_utf8(comparison.reference_path);
    document["scene_id"] = scene_key(comparison.scene);
    document["report_ids"] = comparison.report_ids;
    document["baseline_report_id"] = comparison.baseline_report_id;
    return write_file_atomically(comparison_path(), document.dump(2));
}

std::optional<SavedComparison> ReportStore::load_comparison() const {
    std::ifstream input(comparison_path(), std::ios::binary);
    if (!input) return std::nullopt;
    try {
        const nlohmann::json document = nlohmann::json::parse(input);
        if (!document.is_object() || !document.contains("report_ids") ||
            !document["report_ids"].is_array()) {
            return std::nullopt;
        }
        SavedComparison comparison;
        const std::string reference = document.value("reference_path", std::string{});
        comparison.reference_path = std::filesystem::path(
            std::u8string(reference.begin(), reference.end()));
        comparison.scene = parse_scene_key(
            document.value("scene_id", std::string{"tv_standard"}));
        for (const nlohmann::json& id : document["report_ids"]) {
            if (id.is_string()) comparison.report_ids.push_back(id.get<std::string>());
        }
        comparison.baseline_report_id =
            document.value("baseline_report_id", std::string{});
        return comparison;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

Result<std::string> ReportStore::import_report(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return Result<std::string>::failure(
            io_failure(i18n::tr("Cannot open the report to import"), path));
    }
    nlohmann::json document;
    try {
        document = nlohmann::json::parse(input);
    } catch (const std::exception& exception) {
        return Result<std::string>::failure(
            {ErrorCode::unsupported_format,
             i18n::tr("The imported file is not valid JSON: {}", exception.what())});
    }
    auto report = evaluation_report_from_json(document);
    if (!report) return Result<std::string>::failure(report.error());
    // Import never overwrites an existing report with the same id.
    std::error_code error;
    if (!report.value().report_id.empty() &&
        std::filesystem::exists(report_path(report.value().report_id),
                                error)) {
        report.value().report_id.clear();
    }
    return save_report(report.value());
}

}  // namespace crossvmaf::studio
