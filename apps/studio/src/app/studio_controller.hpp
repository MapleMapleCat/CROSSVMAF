#pragma once

// Thin adapter between the Slint VM and the Studio application services
// (probe, task queue, report store, analysis, comparison, diagnostics).
// Contains no evaluation or persistence logic of its own.

#include "evaluation/comparison_setup.hpp"
#include "evaluation/probe_service.hpp"
#include "presentation/report_presenter.hpp"
#include "presentation/display_format.hpp"
#include "reports/comparison_session.hpp"
#include "reports/report_store.hpp"
#include "reports/report_writer.hpp"
#include "evaluation/scene_policy.hpp"
#include "app/settings.hpp"
#include "evaluation/task_queue.hpp"
#include "i18n/language.hpp"
#include "types.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace crossvmaf::studio {

class StudioController final {
public:
    explicit StudioController(const VM& view_model,
                              std::filesystem::path settings_path = {});
    ~StudioController();

    StudioController(const StudioController&) = delete;
    StudioController& operator=(const StudioController&) = delete;

    void initialize();

private:
    friend struct StudioControllerTestAccess;

    // C++-rendered VM strings that must follow a runtime language change.
    // Each slot keeps the renderer of its latest value; refreshLocalizedText()
    // renders all of them again in the new language.
    enum class TextSlot : std::uint8_t {
        input_status,
        run_title,
        run_backend,
        run_status,
        run_percent,
        run_frames,
        run_speed,
        run_eta,
        self_test_status,
        decode_device_status,
        settings_status,
        action_message,
        count,
    };
    using TextRenderer = std::function<std::string()>;

    void installCallbacks();
    void loadDevices();
    void loadSettings();
    void loadDecodeDevices();
    void saveSettings();
    [[nodiscard]] StudioSettings currentSettings() const;

    // UI language: both translation layers, then every C++-rendered string.
    void changeLanguage();
    void applyLanguage();
    void refreshLocalizedText();
    void setText(TextSlot slot, TextRenderer render);
    void applyText(TextSlot slot) const;
    void applyLanguageOptions() const;
    void applyDeviceOptions() const;
    void applyDecodeBackendOptions() const;
    void applyDecodeDeviceOptions();

    // New screen: selection, probing, and pair validation.
    void chooseReference();
    void chooseDistorted();
    void applySelectedInputs();
    void applyInputSlots();
    void requestProbe(InputRole role);
    void handleProbeOutcome(ProbeService::ProbeOutcome outcome);
    void refreshInputReadiness();
    [[nodiscard]] std::optional<EvaluationRequest> currentRequest();
    // The request for one pair with the current settings and a resolved model.
    [[nodiscard]] EvaluationRequest buildRequest(
        const std::filesystem::path& reference_path,
        const VideoMetadata* reference_metadata,
        const std::filesystem::path& distorted_path,
        const VideoMetadata* distorted_metadata) const;
    [[nodiscard]] bool inputIdentityUnchanged() const;

    // Queue operations.
    void addCurrentRequestToQueue();
    void startCurrentRequest();
    void cancelActiveTask();
    void startQueue();
    void pauseQueue();
    void queueAction(std::int32_t index);

    // TaskQueue event handlers; always entered on the UI thread via
    // postToUi.
    void onTaskStateChanged(const TaskSnapshot& snapshot);
    void onTaskOpened(const std::string& task_id, const OpenedInfo& opened);
    void onTaskProgress(const ProgressSnapshot& progress);
    void onTaskFramesAppended(std::vector<double> scores);
    void onTaskCompleted(std::shared_ptr<const EvaluationReport> report);

    void applyOpenedToRunScreen(const OpenedInfo& opened);
    void applyProgressToRunScreen(const ProgressSnapshot& progress);
    void resetRunScreen(const EvaluationRequest& request);
    void updateQueueModel();
    void updateRecentModel(std::size_t limit = 20U);
    void loadMoreRecentReports();
    void applyRecentModel();

    // Compare screen. The comparison changes only through the user: the
    // set-up step, the table's actions, and the evaluations it started.
    void initializeComparison();
    void updateComparisonModel();
    // The table and status line, after a video's state or progress changed.
    void applyComparisonProgress();
    void applyComparisonStatus();
    void saveComparison();
    void seedComparisonFromLibrary();
    [[nodiscard]] std::vector<std::shared_ptr<const EvaluationReport>>
    reportsRelatedTo(const EvaluationReport& anchor);
    [[nodiscard]] Result<std::shared_ptr<const EvaluationReport>> loadReport(
        const std::string& report_id) const;
    // CmpTrack.track → report id or task id; empty when out of range.
    [[nodiscard]] std::string comparisonTrackId(std::int32_t track) const;
    void setComparisonBaseline(std::int32_t track);
    void removeComparisonTrack(std::int32_t track);
    void retryComparisonTrack(std::int32_t track);
    void openComparisonTrack(std::int32_t track);
    void stopComparison();
    void onComparisonTaskProgress(const std::string& task_id,
                                  const ProgressSnapshot& progress);

    // Compare set-up step: a reference and the videos to compare with it.
    void openComparisonSetup(bool editing);
    void closeComparisonSetup();
    void setComparisonSetupReference(std::filesystem::path path);
    void addComparisonSetupFiles(const std::vector<std::filesystem::path>& paths);
    void handleComparisonProbe(ProbeService::ProbeOutcome outcome);
    void findComparisonSavedResults();
    [[nodiscard]] const ComparisonSetup::Row* comparisonSetupRow(std::int32_t index) const;
    void applyComparisonSetup();
    void applyComparisonCandidates();
    void addSavedToComparisonSetup(std::size_t index);
    void startComparisonSetup();
    // Entry points from the Report and New screens.
    void compareDisplayedReport();
    void compareFromNewScreen();
    void selectRecentReport(std::size_t index);
    void reuseRecentReport(std::size_t index);
    void deleteRecentReport(std::size_t index);
    void showReport(std::shared_ptr<const EvaluationReport> report);
    void navigateTo(std::int32_t screen);
    void locateFrame(std::size_t frame_index);

    void exportCurrentReport(ReportFormat format);
    void exportCurrentComparison();
    void exportTaskDiagnostics(bool current_task = false);
    void runSelfTest();
    void showActionMessage(TextRenderer message, bool is_error);
    // For messages that are already final, such as errors from a service.
    void showActionMessage(std::string message, bool is_error);

    void postToUi(std::function<void()> action);

    [[nodiscard]] std::optional<DeviceSelector> selectedDevice() const;
    [[nodiscard]] std::filesystem::path settingsPath() const;
    [[nodiscard]] static std::filesystem::path reportsDirectory();

    const VM& view_model_;
    // Outlive the worker services, which may post during their destruction.
    std::atomic_bool shutting_down_{false};
    std::shared_ptr<std::atomic_bool> ui_alive_{
        std::make_shared<std::atomic_bool>(true)};
    std::filesystem::path settings_path_;
    ReportPresenter report_presenter_;
    ReportStore report_store_;
    TaskQueue task_queue_;
    ProbeService probe_service_;

    std::filesystem::path reference_path_;
    std::filesystem::path distorted_path_;
    std::optional<VideoMetadata> reference_metadata_;
    std::optional<VideoMetadata> distorted_metadata_;
    std::uint64_t reference_probe_version_{};
    std::uint64_t distorted_probe_version_{};
    std::string reference_probe_error_;
    std::string distorted_probe_error_;
    bool reference_probe_pending_{};
    bool distorted_probe_pending_{};

    std::optional<i18n::Language> language_preference_;
    std::array<TextRenderer, static_cast<std::size_t>(TextSlot::count)> text_renderers_;

    std::vector<DeviceInfo> devices_;
    std::vector<std::string> device_uuids_;
    std::vector<DecodeBackendInfo> decode_backends_;
    // Filled on the UI thread once the background enumeration finishes.
    std::optional<std::vector<DecodeDeviceInfo>> decode_devices_;
    std::vector<std::string> decode_device_ids_{"same"};
    std::thread decode_devices_thread_;

    // Run-screen state, UI thread only.
    std::string active_task_id_;
    std::vector<double> live_scores_;
    ProgressSnapshot last_progress_;
    std::vector<TaskSnapshot> task_rows_;
    std::vector<std::shared_ptr<const EvaluationReport>> session_reports_;
    std::vector<ReportSummary> recent_summaries_;
    std::optional<std::size_t> recent_cursor_;
    ComparisonSession comparison_;
    ComparisonSetup comparison_setup_;
    // Every task started for the Compare screen, also after it left the
    // comparison: finishing one never replaces the displayed report.
    std::set<std::string> comparison_task_ids_;
    // Rows of VM.cmp-candidates: saved evaluations of the set-up reference.
    std::vector<ReportSummary> comparison_candidates_;
    ComparisonResult last_comparison_;

    std::thread background_thread_;
};

}  // namespace crossvmaf::studio
