#include "app/studio_controller.hpp"

#include "app/settings_store.hpp"
#include "i18n/translator.hpp"
#include "platform/diagnostics.hpp"
#include "platform/native_dialogs.hpp"
#include "presentation/ui_language.hpp"

#include <nlohmann/json.hpp>
#include <slint.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string_view>
#include <system_error>
#include <utility>

namespace crossvmaf::studio {
namespace {

using SteadyClock = std::chrono::steady_clock;

[[nodiscard]] slint::SharedString sharedString(const std::string& value) {
    return slint::SharedString(std::string_view(value));
}

template <typename Item>
[[nodiscard]] std::shared_ptr<slint::VectorModel<Item>> vectorModel(
    std::vector<Item> values) {
    return std::make_shared<slint::VectorModel<Item>>(std::move(values));
}

[[nodiscard]] std::string videoCodecName(StreamVideoCodec codec) {
    switch (codec) {
    case StreamVideoCodec::h264: return "H.264";
    case StreamVideoCodec::hevc: return "HEVC";
    case StreamVideoCodec::av1: return "AV1";
    case StreamVideoCodec::vp8: return "VP8";
    case StreamVideoCodec::vp9: return "VP9";
    case StreamVideoCodec::mpeg1: return "MPEG-1";
    case StreamVideoCodec::mpeg2: return "MPEG-2";
    case StreamVideoCodec::mpeg4: return "MPEG-4 Part 2";
    case StreamVideoCodec::vc1: return "VC-1";
    case StreamVideoCodec::flv1: return "FLV1";
    case StreamVideoCodec::mjpeg: return "MJPEG";
    }
    return i18n::tr("Unknown codec");
}

[[nodiscard]] std::string decodeBackendName(StreamDecodeBackend backend) {
    switch (backend) {
    case StreamDecodeBackend::software: return i18n::tr("Software decoding");
    case StreamDecodeBackend::vulkan_video: return "Vulkan Video";
    case StreamDecodeBackend::qsv: return "Intel QSV";
    case StreamDecodeBackend::nvdec: return "NVIDIA NVDEC";
    case StreamDecodeBackend::vaapi: return "VA-API";
    }
    return i18n::tr("Unknown backend");
}

// Language picker rows: 0 follows the system, then supported_languages().
[[nodiscard]] std::int32_t languageIndex(std::optional<i18n::Language> language) {
    if (!language.has_value()) return 0;
    const auto languages = i18n::supported_languages();
    for (std::size_t index = 0U; index < languages.size(); ++index) {
        if (languages[index].language == *language) {
            return static_cast<std::int32_t>(index + 1U);
        }
    }
    return 0;
}

[[nodiscard]] std::optional<i18n::Language> languageAt(std::int32_t index) {
    const auto languages = i18n::supported_languages();
    if (index <= 0 || static_cast<std::size_t>(index) > languages.size()) {
        return std::nullopt;
    }
    return languages[static_cast<std::size_t>(index) - 1U].language;
}

[[nodiscard]] std::string pathStemUtf8(const std::filesystem::path& path) {
    const std::u8string stem = path.stem().u8string();
    return std::string(reinterpret_cast<const char*>(stem.data()), stem.size());
}

// Probe slot of the Compare set-up reference; its videos use their row ids.
constexpr std::uint64_t comparison_reference_slot{1U};

// Recomputes the identity snapshot the same way the core probe does, so a
// file changed after probing is detected before a run starts.
[[nodiscard]] FileIdentity captureIdentity(const std::filesystem::path& path) {
    FileIdentity identity;
    identity.path = path;
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    if (error) return identity;
    const auto file_time = std::filesystem::last_write_time(path, error);
    if (error) return identity;
    const auto system_time = std::chrono::time_point_cast<
        std::chrono::system_clock::duration>(
        file_time - std::filesystem::file_time_type::clock::now() +
        std::chrono::system_clock::now());
    identity.size_bytes = static_cast<std::uint64_t>(size);
    identity.modification_time_utc_seconds =
        std::chrono::duration_cast<std::chrono::seconds>(
            system_time.time_since_epoch()).count();
    identity.captured = true;
    return identity;
}

[[nodiscard]] std::vector<slint::SharedString> metadataChips(
    const VideoMetadata& metadata) {
    std::vector<slint::SharedString> chips;
    chips.push_back(sharedString(videoCodecName(metadata.codec)));
    if (metadata.descriptor.has_value()) {
        const FrameDescriptor& descriptor = *metadata.descriptor;
        chips.push_back(sharedString(
            std::to_string(descriptor.width) + " × " +
            std::to_string(descriptor.height)));
        chips.push_back(sharedString(
            std::to_string(descriptor.bit_depth) + " bit"));
    } else {
        chips.push_back(sharedString(i18n::tr("Unknown spec")));
    }
    if (metadata.frame_rate.average.valid()) {
        chips.push_back(sharedString(
            formatDecimal(metadata.frame_rate.average.value(), 2) + " fps" +
            (metadata.frame_rate.mode == FrameRateMode::variable
                 ? " (VFR)"
                 : "")));
    } else {
        chips.push_back(sharedString(i18n::tr("Unknown frame rate")));
    }
    if (metadata.duration_seconds.has_value()) {
        chips.push_back(
            sharedString(formatMediaTime(*metadata.duration_seconds)));
    }
    if (metadata.identity.captured) {
        chips.push_back(
            sharedString(formatByteSize(metadata.identity.size_bytes)));
    }
    return chips;
}

[[nodiscard]] std::string metadataSummary(const VideoMetadata& metadata) {
    std::string summary;
    for (const slint::SharedString& chip : metadataChips(metadata)) {
        if (!summary.empty()) summary += " · ";
        summary += std::string_view(chip);
    }
    return summary;
}

} // namespace

StudioController::StudioController(const VM& view_model, std::filesystem::path settings_path)
    : view_model_(view_model),
      settings_path_(std::move(settings_path)),
      report_store_(reportsDirectory()),
      task_queue_(TaskQueue::Callbacks{
          [this](const TaskSnapshot& snapshot) {
              postToUi([this, snapshot] { onTaskStateChanged(snapshot); });
          },
          [this](const std::string& task_id, const OpenedInfo& opened) {
              postToUi([this, task_id, opened] {
                  onTaskOpened(task_id, opened);
              });
          },
          [this](const std::string& task_id,
                 const ProgressSnapshot& progress) {
              postToUi([this, task_id, progress] {
                  if (task_id == active_task_id_) onTaskProgress(progress);
                  onComparisonTaskProgress(task_id, progress);
              });
          },
          [this](const std::string& task_id,
                 std::span<const FrameResult> appended) {
              std::vector<double> scores;
              scores.reserve(appended.size());
              for (const FrameResult& frame : appended) {
                  scores.push_back(frame.vmaf);
              }
              postToUi([this, task_id, scores = std::move(scores)]() mutable {
                  if (task_id == active_task_id_) {
                      onTaskFramesAppended(std::move(scores));
                  }
              });
          },
          [this](std::shared_ptr<const EvaluationReport> report) {
              postToUi([this, report = std::move(report)] {
                  onTaskCompleted(report);
              });
          }}),
      probe_service_(ProbeService::Callback(
          [this](ProbeService::ProbeOutcome outcome) {
              postToUi([this, outcome = std::move(outcome)]() mutable {
                  handleProbeOutcome(std::move(outcome));
              });
          })) {}

StudioController::~StudioController() {
    shutting_down_.store(true, std::memory_order_release);
    ui_alive_->store(false, std::memory_order_release);
    if (background_thread_.joinable()) background_thread_.join();
    if (decode_devices_thread_.joinable()) decode_devices_thread_.join();
}

void StudioController::initialize() {
    report_presenter_.clearReport(view_model_);
    loadDevices();
    loadDecodeDevices();
    // Selects the UI language and re-renders the text produced above.
    loadSettings();
    applySelectedInputs();
    updateQueueModel();
    updateRecentModel();
    initializeComparison();
    installCallbacks();
}

void StudioController::postToUi(std::function<void()> action) {
    if (shutting_down_.load(std::memory_order_acquire)) return;
    slint::invoke_from_event_loop(
        [ui_alive = ui_alive_, action = std::move(action)]() mutable {
            if (!ui_alive->load(std::memory_order_acquire)) return;
            action();
        });
}

void StudioController::installCallbacks() {
    view_model_.on_nav_to([this](std::int32_t screen) { navigateTo(screen); });
    view_model_.on_choose_reference([this] { chooseReference(); });
    view_model_.on_choose_distorted([this] { chooseDistorted(); });
    view_model_.on_start_run([this] { startCurrentRequest(); });
    view_model_.on_add_to_queue([this] { addCurrentRequestToQueue(); });
    view_model_.on_cancel_run([this] { cancelActiveTask(); });
    view_model_.on_queue_start([this] { startQueue(); });
    view_model_.on_queue_pause([this] { pauseQueue(); });
    view_model_.on_queue_action([this](std::int32_t index) {
        queueAction(index);
    });
    view_model_.on_select_recent([this](std::int32_t index) {
        if (index >= 0) selectRecentReport(static_cast<std::size_t>(index));
    });
    view_model_.on_reuse_recent([this](std::int32_t index) {
        if (index >= 0) reuseRecentReport(static_cast<std::size_t>(index));
    });
    view_model_.on_delete_recent([this](std::int32_t index) {
        if (index >= 0) deleteRecentReport(static_cast<std::size_t>(index));
    });
    view_model_.on_load_more_recent([this] { loadMoreRecentReports(); });
    view_model_.on_report_hover(
        [this](float x, float y, float width, float height) {
            report_presenter_.applyHover(view_model_, x, y, width, height);
        });
    view_model_.on_report_hover_end([this] {
        view_model_.set_hov_visible(false);
    });
    view_model_.on_toggle_cmp_series([this](std::int32_t track) {
        if (track < 0) return;
        report_presenter_.toggleComparisonSeries(static_cast<std::size_t>(track));
        report_presenter_.applyComparison(view_model_);
    });
    view_model_.on_set_cmp_baseline([this](std::int32_t track) {
        setComparisonBaseline(track);
    });
    view_model_.on_remove_cmp_track([this](std::int32_t track) {
        removeComparisonTrack(track);
    });
    view_model_.on_retry_cmp_track([this](std::int32_t track) {
        retryComparisonTrack(track);
    });
    view_model_.on_open_cmp_track([this](std::int32_t track) {
        openComparisonTrack(track);
    });
    view_model_.on_stop_comparison([this] { stopComparison(); });
    view_model_.on_edit_comparison([this] { openComparisonSetup(true); });
    view_model_.on_new_comparison([this] { openComparisonSetup(false); });
    view_model_.on_cmps_choose_reference([this] {
        auto selected = chooseVideoFile();
        if (!selected) return;
        setComparisonSetupReference(std::move(*selected));
        view_model_.set_action_message_visible(false);
        applyComparisonSetup();
        applyComparisonCandidates();
    });
    view_model_.on_cmps_add_files([this] {
        const auto selected = chooseVideoFiles();
        if (!selected.empty()) addComparisonSetupFiles(selected);
    });
    view_model_.on_cmps_remove_row([this](std::int32_t index) {
        if (const ComparisonSetup::Row* row = comparisonSetupRow(index)) {
            comparison_setup_.remove_row(row->id);
            applyComparisonSetup();
            applyComparisonCandidates();
        }
    });
    view_model_.on_cmps_toggle_reuse([this](std::int32_t index) {
        if (const ComparisonSetup::Row* row = comparisonSetupRow(index)) {
            comparison_setup_.toggle_reuse(row->id);
            applyComparisonSetup();
        }
    });
    view_model_.on_cmps_add_saved([this](std::int32_t index) {
        if (index >= 0) addSavedToComparisonSetup(static_cast<std::size_t>(index));
    });
    view_model_.on_cmps_start([this] { startComparisonSetup(); });
    view_model_.on_cmps_cancel([this] {
        if (!comparison_.empty()) closeComparisonSetup();
    });
    view_model_.on_compare_report([this] { compareDisplayedReport(); });
    view_model_.on_compare_from_new([this] { compareFromNewScreen(); });
    view_model_.on_cmp_hover([this](float x, float width) {
        report_presenter_.applyComparisonHover(view_model_, x, width);
    });
    view_model_.on_cmp_hover_end([this] {
        view_model_.set_cmp_hov_visible(false);
    });
    view_model_.on_locate_frame([this](std::int32_t frame_index) {
        if (frame_index >= 0) locateFrame(static_cast<std::size_t>(frame_index));
    });
    view_model_.on_export_report([this](std::int32_t format) {
        if (format == 2) {
            exportTaskDiagnostics();
        } else {
            exportCurrentReport(
                format == 1 ? ReportFormat::csv : ReportFormat::json);
        }
    });
    view_model_.on_export_comparison([this] { exportCurrentComparison(); });
    view_model_.on_generate_pdf([this] {
        exportCurrentReport(ReportFormat::pdf);
    });
    view_model_.on_run_self_test([this] { runSelfTest(); });
    view_model_.on_export_run_diagnostics([this] { exportTaskDiagnostics(true); });
    view_model_.on_settings_changed([this] { saveSettings(); });
    view_model_.on_language_changed([this] { changeLanguage(); });
    view_model_.on_decode_device_changed([this](int index) {
        if (index >= 0 && static_cast<std::size_t>(index) < decode_device_ids_.size()) {
            view_model_.set_decode_device_id(sharedString(decode_device_ids_[static_cast<std::size_t>(index)]));
            saveSettings();
        }
    });
    view_model_.on_settings_device_changed([this] {
        setText(TextSlot::self_test_status, [] { return i18n::tr("Self-test not run"); });
        view_model_.set_self_test_kind(-1);
        saveSettings();
    });
    view_model_.on_open_reports_directory([this] {
        const auto opened = openDirectory(report_store_.directory());
        if (!opened) showActionMessage(opened.error().message, true);
    });
}

void StudioController::loadDevices() {
    devices_.clear();
    device_uuids_.clear();
    device_uuids_.push_back({});
    auto enumerated_devices = EvaluationStream::enumerate_devices();
    if (!enumerated_devices) {
        applyDeviceOptions();
        setText(TextSlot::self_test_status,
                [message = enumerated_devices.error().message] {
                    return i18n::tr("Device enumeration failed: {}", message);
                });
        view_model_.set_self_test_kind(3);
        return;
    }
    devices_ = std::move(enumerated_devices).value();
    for (const DeviceInfo& device : devices_) device_uuids_.push_back(device.uuid);
    applyDeviceOptions();
    setText(TextSlot::self_test_status, [] { return i18n::tr("Self-test not run yet"); });
    view_model_.set_self_test_kind(4);
}

void StudioController::applyDeviceOptions() const {
    std::vector<slint::SharedString> options;
    options.push_back(sharedString(i18n::tr("Automatic (recommended)")));
    for (const DeviceInfo& device : devices_) {
        std::string label = device.name;
        if (device.device_local_heap_bytes > 0U) {
            label += " · " + formatByteSize(device.device_local_heap_bytes);
        }
        label += " · " + (device.discrete ? i18n::tr("Discrete GPU")
                                          : i18n::tr("Integrated / other"));
        options.push_back(sharedString(label));
    }
    view_model_.set_device_options(vectorModel(std::move(options)));
}

void StudioController::loadDecodeDevices() {
    decode_backends_ = enumerate_decode_backends();
    applyDecodeBackendOptions();
    setText(TextSlot::decode_device_status,
            [] { return i18n::tr("Looking for decode devices…"); });
    decode_devices_thread_ = std::thread([this] {
        try {
            auto result = enumerate_decode_devices();
            postToUi([this, result = std::move(result)]() mutable {
                decode_devices_ = result ? std::move(result).value()
                                         : std::vector<DecodeDeviceInfo>{};
                applyDecodeDeviceOptions();
                if (result) {
                    setText(TextSlot::decode_device_status, [] {
                        return i18n::tr(
                            "Devices found; each video format is verified when a task starts.");
                    });
                } else {
                    setText(TextSlot::decode_device_status,
                            [message = result.error().message] { return message; });
                }
            });
        } catch (const std::exception& error) {
            const std::string message = error.what();
            postToUi([this, message] {
                setText(TextSlot::decode_device_status, [message] { return message; });
            });
        }
    });
}

void StudioController::applyDecodeBackendOptions() const {
    std::vector<slint::SharedString> labels{sharedString(i18n::tr("Automatic"))};
    for (const auto& backend : decode_backends_) {
        if (backend.backend == StreamDecodeBackend::software) continue;
        auto label = decodeBackendName(backend.backend);
        if (!backend.platform_supported) {
            label = i18n::tr("{} (not supported on this platform)", label);
        } else if (!backend.built) {
            label = i18n::tr("{} (not built)", label);
        }
        labels.push_back(sharedString(label));
    }
    view_model_.set_decode_backend_options(vectorModel(std::move(labels)));
}

void StudioController::applyDecodeDeviceOptions() {
    // Until enumeration finishes, the Slint default row stays in place.
    if (!decode_devices_.has_value()) return;
    std::vector<slint::SharedString> options{sharedString(i18n::tr("Same as compute GPU"))};
    decode_device_ids_ = {"same"};
    for (const auto& device : *decode_devices_) {
        options.push_back(sharedString(decodeBackendName(device.backend) + " · " + device.name));
        decode_device_ids_.push_back(device.id);
    }
    const std::string saved(view_model_.get_decode_device_id());
    auto selected = std::find(decode_device_ids_.begin(), decode_device_ids_.end(), saved);
    if (selected == decode_device_ids_.end()) {
        options.push_back(sharedString(i18n::tr("{} (currently unavailable)", saved)));
        decode_device_ids_.push_back(saved);
        selected = std::prev(decode_device_ids_.end());
    }
    view_model_.set_decode_device_idx(static_cast<int>(std::distance(decode_device_ids_.begin(), selected)));
    view_model_.set_decode_device_options(vectorModel(std::move(options)));
}

void StudioController::loadSettings() {
    // A settings file that fails to load means following the system language,
    // so describe the failure in that language.
    i18n::set_language(i18n::system_language());
    const auto loaded = SettingsStore(settingsPath()).load();
    const StudioSettings settings = loaded ? loaded.value() : StudioSettings{};
    language_preference_ = settings.language;
    view_model_.set_lang_idx(languageIndex(settings.language));
    view_model_.set_theme_idx(settings.theme);
    view_model_.set_export_idx(settings.export_format);
    view_model_.set_decode_idx(settings.decode_mode == DecodeMode::hardware ? 1 :
        settings.decode_mode == DecodeMode::software ? 2 : 0);
    view_model_.set_decode_backend_idx(static_cast<int>(settings.decode_backend));
    view_model_.set_decode_transfer_idx(static_cast<int>(settings.decode_transfer));
    view_model_.set_decode_device_id(sharedString(settings.decode_device));
    view_model_.set_decoder_threads(static_cast<int>(settings.decoder_threads_per_source));
    view_model_.set_decode_buffer(static_cast<int>(settings.decoded_frame_queue_capacity));
    view_model_.set_queue_failure_idx(settings.failure_policy == QueueFailurePolicy::pause_on_failure ? 0 : 1);
    task_queue_.set_failure_policy(settings.failure_policy);
    const auto selected = std::find(device_uuids_.begin(), device_uuids_.end(), settings.device_uuid);
    view_model_.set_device_idx(selected == device_uuids_.end() ? 0 :
        static_cast<int>(std::distance(device_uuids_.begin(), selected)));
    const auto directory_utf8 = report_store_.directory().u8string();
    view_model_.set_reports_directory(sharedString(std::string(
        reinterpret_cast<const char*>(directory_utf8.data()), directory_utf8.size())));
    view_model_.set_settings_save_error(!loaded);
    if (loaded) {
        setText(TextSlot::settings_status,
                [] { return i18n::tr("Changes are saved automatically"); });
    } else {
        setText(TextSlot::settings_status,
                [message = loaded.error().message] { return message; });
    }
    applyLanguage();
}

StudioSettings StudioController::currentSettings() const {
    StudioSettings settings;
    settings.theme = view_model_.get_theme_idx();
    settings.language = languageAt(view_model_.get_lang_idx());
    if (const auto device = selectedDevice()) settings.device_uuid = device->uuid;
    settings.export_format = view_model_.get_export_idx();
    settings.decode_mode = view_model_.get_decode_idx() == 1 ? DecodeMode::hardware :
        view_model_.get_decode_idx() == 2 ? DecodeMode::software : DecodeMode::automatic;
    settings.decode_backend = static_cast<DecodeBackendPreference>(view_model_.get_decode_backend_idx());
    settings.decode_transfer = static_cast<DecodeTransferPolicy>(view_model_.get_decode_transfer_idx());
    settings.decode_device = std::string(view_model_.get_decode_device_id());
    if (settings.decode_mode == DecodeMode::software) {
        settings.decode_backend = DecodeBackendPreference::automatic;
        settings.decode_transfer = DecodeTransferPolicy::automatic;
        settings.decode_device = "same";
    }
    settings.decoder_threads_per_source = static_cast<std::size_t>(view_model_.get_decoder_threads());
    settings.decoded_frame_queue_capacity = static_cast<std::size_t>(view_model_.get_decode_buffer());
    settings.failure_policy = view_model_.get_queue_failure_idx() == 0 ?
        QueueFailurePolicy::pause_on_failure : QueueFailurePolicy::continue_on_failure;
    return settings;
}

void StudioController::saveSettings() {
    const auto settings = currentSettings();
    task_queue_.set_failure_policy(settings.failure_policy);
    const auto saved = SettingsStore(settingsPath()).save(settings);
    view_model_.set_settings_save_error(!saved);
    if (saved) {
        setText(TextSlot::settings_status, [] { return i18n::tr("Saved automatically"); });
    } else {
        setText(TextSlot::settings_status, [message = saved.error().message] { return message; });
    }
}

void StudioController::changeLanguage() {
    language_preference_ = languageAt(view_model_.get_lang_idx());
    applyLanguage();
    saveSettings();
}

void StudioController::applyLanguage() {
    applyUiLanguage(i18n::resolve_language(language_preference_));
    refreshLocalizedText();
}

// Slint re-evaluates its own @tr bindings; everything C++ rendered is
// rebuilt here from controller state, never from the old strings.
void StudioController::refreshLocalizedText() {
    applyLanguageOptions();
    applyDeviceOptions();
    applyDecodeBackendOptions();
    applyDecodeDeviceOptions();
    applyInputSlots();
    updateQueueModel();
    applyRecentModel();
    if (report_presenter_.report()) {
        report_presenter_.applyReport(view_model_);
    } else {
        report_presenter_.clearReport(view_model_);
    }
    report_presenter_.applyLive(view_model_, live_scores_, last_progress_);
    // Comparison issues are described when the comparison is built.
    updateComparisonModel();
    if (comparison_setup_.active()) {
        applyComparisonSetup();
        applyComparisonCandidates();
    }
    for (std::size_t slot = 0U; slot < text_renderers_.size(); ++slot) {
        applyText(static_cast<TextSlot>(slot));
    }
}

void StudioController::applyLanguageOptions() const {
    std::vector<slint::SharedString> options{sharedString(i18n::tr(
        "Follow system ({})", i18n::language_info(i18n::system_language()).native_name))};
    for (const i18n::LanguageInfo& language : i18n::supported_languages()) {
        options.push_back(sharedString(std::string(language.native_name)));
    }
    view_model_.set_lang_options(vectorModel(std::move(options)));
}

void StudioController::setText(TextSlot slot, TextRenderer render) {
    text_renderers_[static_cast<std::size_t>(slot)] = std::move(render);
    applyText(slot);
}

void StudioController::applyText(TextSlot slot) const {
    const TextRenderer& render = text_renderers_[static_cast<std::size_t>(slot)];
    // Untouched slots keep their Slint default, which @tr already translates.
    if (!render) return;
    const slint::SharedString text = sharedString(render());
    switch (slot) {
    case TextSlot::input_status: view_model_.set_input_status(text); break;
    case TextSlot::run_title: view_model_.set_run_title(text); break;
    case TextSlot::run_backend: view_model_.set_run_backend(text); break;
    case TextSlot::run_status: view_model_.set_run_status(text); break;
    case TextSlot::run_percent: view_model_.set_run_pct(text); break;
    case TextSlot::run_frames: view_model_.set_run_frames(text); break;
    case TextSlot::run_speed: view_model_.set_run_speed(text); break;
    case TextSlot::run_eta: view_model_.set_run_eta(text); break;
    case TextSlot::self_test_status: view_model_.set_self_test_status(text); break;
    case TextSlot::decode_device_status: view_model_.set_decode_device_status(text); break;
    case TextSlot::settings_status: view_model_.set_settings_status(text); break;
    case TextSlot::action_message: view_model_.set_action_message(text); break;
    case TextSlot::count: break;
    }
}

void StudioController::chooseReference() {
    const auto selected = chooseVideoFile();
    if (!selected) return;
    reference_path_ = *selected;
    reference_metadata_.reset();
    requestProbe(InputRole::reference);
    applySelectedInputs();
}

void StudioController::chooseDistorted() {
    const auto selected = chooseVideoFile();
    if (!selected) return;
    distorted_path_ = *selected;
    distorted_metadata_.reset();
    requestProbe(InputRole::distorted);
    applySelectedInputs();
}

void StudioController::requestProbe(InputRole role) {
    if (role == InputRole::reference) {
        reference_probe_pending_ = true;
        reference_probe_error_.clear();
        reference_probe_version_ =
            probe_service_.probe(role, reference_path_);
    } else if (role == InputRole::distorted) {
        distorted_probe_pending_ = true;
        distorted_probe_error_.clear();
        distorted_probe_version_ =
            probe_service_.probe(role, distorted_path_);
    }
}

void StudioController::applySelectedInputs() {
    applyInputSlots();
    refreshInputReadiness();
}

void StudioController::applyInputSlots() {
    const auto display_path = [](const std::filesystem::path& path) {
        const auto utf8 = path.u8string();
        return sharedString(std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size()));
    };
    view_model_.set_reference_selected(!reference_path_.empty());
    view_model_.set_distorted_selected(!distorted_path_.empty());
    view_model_.set_reference_path(display_path(reference_path_));
    view_model_.set_distorted_path(display_path(distorted_path_));
    if (reference_path_.empty()) {
        view_model_.set_reference_name(sharedString(i18n::tr("Choose a reference video")));
        view_model_.set_reference_chips(vectorModel(std::vector<slint::SharedString>{
            sharedString(i18n::tr("Click to choose a local file"))}));
    } else {
        view_model_.set_reference_name(sharedString(fileDisplayName(reference_path_)));
        view_model_.set_reference_chips(vectorModel(
            reference_probe_pending_
                ? std::vector<slint::SharedString>{sharedString(i18n::tr("Probing…"))}
                : reference_metadata_.has_value()
                    ? metadataChips(*reference_metadata_)
                    : std::vector<slint::SharedString>{
                          sharedString(i18n::tr("Waiting for probe"))}));
    }
    if (distorted_path_.empty()) {
        view_model_.set_distorted_name(sharedString(i18n::tr("Choose a distorted video")));
        view_model_.set_distorted_chips(vectorModel(std::vector<slint::SharedString>{
            sharedString(i18n::tr("Click to choose a local file"))}));
    } else {
        view_model_.set_distorted_name(sharedString(fileDisplayName(distorted_path_)));
        view_model_.set_distorted_chips(vectorModel(
            distorted_probe_pending_
                ? std::vector<slint::SharedString>{sharedString(i18n::tr("Probing…"))}
                : distorted_metadata_.has_value()
                    ? metadataChips(*distorted_metadata_)
                    : std::vector<slint::SharedString>{
                          sharedString(i18n::tr("Waiting for probe"))}));
    }
}

void StudioController::handleProbeOutcome(
    ProbeService::ProbeOutcome outcome) {
    if (outcome.slot != 0U) {
        handleComparisonProbe(std::move(outcome));
        return;
    }
    if (outcome.role == InputRole::reference) {
        if (outcome.version != reference_probe_version_) return;
        reference_probe_pending_ = false;
        if (outcome.metadata) {
            reference_metadata_ = std::move(outcome.metadata).value();
            reference_probe_error_.clear();
        } else {
            reference_metadata_.reset();
            reference_probe_error_ = outcome.metadata.error().message;
        }
    } else if (outcome.role == InputRole::distorted) {
        if (outcome.version != distorted_probe_version_) return;
        distorted_probe_pending_ = false;
        if (outcome.metadata) {
            distorted_metadata_ = std::move(outcome.metadata).value();
            distorted_probe_error_.clear();
        } else {
            distorted_metadata_.reset();
            distorted_probe_error_ = outcome.metadata.error().message;
        }
    } else {
        return;
    }
    applySelectedInputs();
}

void StudioController::refreshInputReadiness() {
    if (reference_path_.empty() || distorted_path_.empty()) {
        view_model_.set_input_ready(false);
        view_model_.set_input_status_kind(0);
        setText(TextSlot::input_status, [] {
            return i18n::tr("Choose a reference video and a distorted video first");
        });
        return;
    }
    if (same_file(reference_path_, distorted_path_)) {
        view_model_.set_input_ready(false);
        view_model_.set_input_status_kind(2);
        setText(TextSlot::input_status, [] {
            return i18n::tr("The reference and distorted videos cannot be the same file");
        });
        return;
    }
    if (reference_probe_pending_ || distorted_probe_pending_) {
        view_model_.set_input_ready(false);
        view_model_.set_input_status_kind(0);
        setText(TextSlot::input_status,
                [] { return i18n::tr("Probing video codec, spec and frame rate…"); });
        return;
    }
    if (!reference_probe_error_.empty() || !distorted_probe_error_.empty()) {
        view_model_.set_input_ready(false);
        view_model_.set_input_status_kind(2);
        const std::string& message = !reference_probe_error_.empty()
            ? reference_probe_error_
            : distorted_probe_error_;
        setText(TextSlot::input_status, [message] {
            return i18n::tr("Video probing failed: {}", message);
        });
        return;
    }
    if (!reference_metadata_.has_value() ||
        !distorted_metadata_.has_value()) {
        view_model_.set_input_ready(false);
        view_model_.set_input_status_kind(0);
        setText(TextSlot::input_status,
                [] { return i18n::tr("Waiting for video probe results"); });
        return;
    }

    auto validation = validate_video_pair(
        *reference_metadata_, *distorted_metadata_);
    if (!validation) {
        view_model_.set_input_ready(false);
        view_model_.set_input_status_kind(2);
        setText(TextSlot::input_status, [message = validation.error().message] {
            return i18n::tr("Pair validation failed: {}", message);
        });
        return;
    }
    // Issue messages come from the core library and are not translated.
    const PairValidation& result = validation.value();
    if (result.status == PairValidationStatus::invalid) {
        const auto issue = std::find_if(
            result.issues.begin(), result.issues.end(), [](const PairIssue& candidate) {
                return candidate.severity == PairIssueSeverity::error;
            });
        view_model_.set_input_ready(false);
        view_model_.set_input_status_kind(2);
        if (issue == result.issues.end()) {
            setText(TextSlot::input_status, [] {
                return i18n::tr("The inputs do not match; the evaluation cannot start");
            });
        } else {
            setText(TextSlot::input_status, [issue = *issue] {
                return i18n::tr("{0} ({1} ↔ {2})", issue.message,
                                issue.reference_value, issue.distorted_value);
            });
        }
        return;
    }
    if (result.status == PairValidationStatus::needs_runtime_validation) {
        const auto issue = std::find_if(
            result.issues.begin(), result.issues.end(), [](const PairIssue& candidate) {
                return candidate.requires_runtime_validation;
            });
        view_model_.set_input_ready(true);
        view_model_.set_input_status_kind(1);
        if (issue == result.issues.end()) {
            setText(TextSlot::input_status, [] {
                return i18n::tr(
                    "Pre-check passed; some constraints are verified per frame at run time");
            });
        } else {
            setText(TextSlot::input_status, [message = issue->message] { return message; });
        }
        return;
    }
    const auto issue = std::find_if(
        result.issues.begin(), result.issues.end(), [](const PairIssue& candidate) {
            return candidate.severity == PairIssueSeverity::warning;
        });
    view_model_.set_input_ready(true);
    view_model_.set_input_status_kind(1);
    if (issue == result.issues.end()) {
        setText(TextSlot::input_status, [] {
            return i18n::tr("Codec, spec and frame rate pre-check passed; ready to evaluate");
        });
    } else {
        setText(TextSlot::input_status, [message = issue->message] { return message; });
    }
}

bool StudioController::inputIdentityUnchanged() const {
    if (reference_metadata_.has_value() &&
        reference_metadata_->identity.captured &&
        !captureIdentity(reference_path_).matches(
            reference_metadata_->identity)) {
        return false;
    }
    if (distorted_metadata_.has_value() &&
        distorted_metadata_->identity.captured &&
        !captureIdentity(distorted_path_).matches(
            distorted_metadata_->identity)) {
        return false;
    }
    return true;
}

std::optional<EvaluationRequest> StudioController::currentRequest() {
    if (!view_model_.get_input_ready() || reference_path_.empty() ||
        distorted_path_.empty() ||
        same_file(reference_path_, distorted_path_)) {
        return std::nullopt;
    }
    if (!inputIdentityUnchanged()) {
        view_model_.set_input_ready(false);
        view_model_.set_input_status_kind(2);
        setText(TextSlot::input_status,
                [] { return i18n::tr("A file changed after probing; probing again"); });
        reference_metadata_.reset();
        distorted_metadata_.reset();
        requestProbe(InputRole::reference);
        requestProbe(InputRole::distorted);
        applySelectedInputs();
        return std::nullopt;
    }

    return buildRequest(
        reference_path_,
        reference_metadata_.has_value() ? &*reference_metadata_ : nullptr,
        distorted_path_,
        distorted_metadata_.has_value() ? &*distorted_metadata_ : nullptr);
}

EvaluationRequest StudioController::buildRequest(
    const std::filesystem::path& reference_path,
    const VideoMetadata* reference_metadata,
    const std::filesystem::path& distorted_path,
    const VideoMetadata* distorted_metadata) const {
    EvaluationRequest request;
    request.reference_path = reference_path;
    request.distorted_path = distorted_path;
    if (const auto device = selectedDevice(); device.has_value()) {
        request.device = *device;
    }
    const auto settings = currentSettings();
    request.decode_mode = settings.decode_mode;
    request.decode_backend = settings.decode_backend;
    request.decode_device = settings.decode_device;
    request.decode_transfer = settings.decode_transfer;

    request.decoder_threads_per_source = settings.decoder_threads_per_source;
    request.decoded_frame_queue_capacity = settings.decoded_frame_queue_capacity;
    // Preview the shared policy; the stream confirms it from decoded dimensions.
    const ResolvedEvaluationConfig resolved = resolve_evaluation_config(
        reference_metadata);
    request.model = resolved.model;
    request.scene = resolved.scene;
    request.model_decision_reason = resolved.decision_reason;
    request.scene_policy_version = resolved.policy_version;
    if (reference_metadata) request.reference_identity = reference_metadata->identity;
    if (distorted_metadata) request.distorted_identity = distorted_metadata->identity;
    return request;
}

std::optional<DeviceSelector> StudioController::selectedDevice() const {
    const std::int32_t device_index = view_model_.get_device_idx();
    if (device_index >= 0 &&
        static_cast<std::size_t>(device_index) < device_uuids_.size()) {
        DeviceSelector selector;
        selector.uuid = device_uuids_[static_cast<std::size_t>(device_index)];
        return selector;
    }
    return std::nullopt;
}

void StudioController::addCurrentRequestToQueue() {
    const auto request = currentRequest();
    if (!request) return;
    const std::string display = pairDisplayName(*request);
    task_queue_.enqueue(*request);
    updateQueueModel();
    view_model_.set_input_status_kind(1);
    setText(TextSlot::input_status, [display, active = task_queue_.queue_active()] {
        return active
            ? i18n::tr("Added to the task queue: {}", display)
            : i18n::tr("Added to the idle task queue; select “Start queue” on the Task "
                       "queue page to run it: {}", display);
    });
}

void StudioController::startCurrentRequest() {
    const auto request = currentRequest();
    if (!request) return;
    task_queue_.enqueue(*request);
    task_queue_.start_queue();
    updateQueueModel();
    view_model_.set_nav_live(true);
    view_model_.set_screen(1);
}

void StudioController::cancelActiveTask() {
    if (active_task_id_.empty()) return;
    if (task_queue_.cancel(active_task_id_)) {
        view_model_.set_run_stopping(true);
        setText(TextSlot::run_status, [] {
            return i18n::tr(
                "Cancelling: waiting inputs wake up at once; GPU work in flight "
                "finishes safely first…");
        });
        view_model_.set_run_status_kind(0);
    }
}

void StudioController::startQueue() {
    task_queue_.start_queue();
    updateQueueModel();
    applyComparisonStatus();
}

void StudioController::pauseQueue() {
    task_queue_.pause_queue();
    updateQueueModel();
    applyComparisonStatus();
}

void StudioController::queueAction(std::int32_t index) {
    if (index < 0 || static_cast<std::size_t>(index) >= task_rows_.size()) {
        return;
    }
    const TaskSnapshot& row = task_rows_[static_cast<std::size_t>(index)];
    switch (row.state) {
    case TaskState::queued:
        // Removing a waiting video from the queue also removes it from its
        // comparison, which would otherwise wait for it forever.
        if (const std::string task_id = row.task_id;
            task_queue_.remove_queued(task_id) && comparison_.owns_task(task_id)) {
            comparison_.remove(task_id);
            updateComparisonModel();
        }
        break;
    case TaskState::opening:
    case TaskState::running:
    case TaskState::finishing:
    case TaskState::cancelling:
        task_queue_.cancel(row.task_id);
        break;
    case TaskState::failed:
    case TaskState::cancelled:
        if (task_queue_.retry(row.task_id)) task_queue_.start_queue();
        break;
    case TaskState::completed:
        return;
    }
    updateQueueModel();
}

void StudioController::onTaskStateChanged(const TaskSnapshot& snapshot) {
    if (comparison_.update_task(snapshot)) applyComparisonProgress();
    for (TaskSnapshot& row : task_rows_) {
        if (row.task_id == snapshot.task_id) {
            row = snapshot;
            break;
        }
    }
    if (snapshot.state == TaskState::opening &&
        snapshot.task_id != active_task_id_) {
        active_task_id_ = snapshot.task_id;
        resetRunScreen(snapshot.request);
    }
    if (snapshot.task_id == active_task_id_ &&
        snapshot.state == TaskState::finishing) {
        setText(TextSlot::run_status, [] {
            return i18n::tr("All inputs submitted; finishing GPU work and model results…");
        });
    }
    if (snapshot.task_id == active_task_id_ &&
        (snapshot.state == TaskState::failed ||
         snapshot.state == TaskState::cancelled)) {
        view_model_.set_run_active(false);
        view_model_.set_run_stopping(false);
        view_model_.set_run_done(false);
        view_model_.set_nav_live(false);
        setText(TextSlot::run_title, [cancelled = snapshot.state == TaskState::cancelled] {
            return cancelled ? i18n::tr("Task stopped") : i18n::tr("Evaluation failed");
        });
        setText(TextSlot::run_status, [error = snapshot.error] {
            return error.has_value() ? describe_task_error(*error) : i18n::tr("Task failed");
        });
        view_model_.set_run_status_kind(
            snapshot.state == TaskState::cancelled ? 0 : 2);
        active_task_id_.clear();
    }
    updateQueueModel();
}

void StudioController::resetRunScreen(const EvaluationRequest& request) {
    live_scores_.clear();
    last_progress_ = {};
    view_model_.set_run_active(true);
    view_model_.set_run_stopping(false);
    view_model_.set_run_done(false);
    view_model_.set_nav_live(true);
    setText(TextSlot::run_title, [] { return i18n::tr("Evaluating"); });
    view_model_.set_run_description(sharedString(pairDisplayName(request)));
    setText(TextSlot::run_backend, [] { return i18n::tr("Initializing the core"); });
    setText(TextSlot::run_status, [] {
        return i18n::tr(
            "Opening the videos, checking timelines and initializing the Vulkan pipeline…");
    });
    view_model_.set_run_status_kind(0);
    setText(TextSlot::run_percent, [] { return i18n::tr("Preparing"); });
    setText(TextSlot::run_frames, [] { return i18n::trn("{0} frame", "{0} frames", 0U, "0"); });
    setText(TextSlot::run_speed, [] { return std::string("—"); });
    view_model_.set_run_elapsed(sharedString("0.0 s"));
    setText(TextSlot::run_eta, [] { return std::string("—"); });
    view_model_.set_run_frac(0.0F);
    report_presenter_.applyLive(view_model_, live_scores_, last_progress_);
}

void StudioController::onTaskOpened(
    const std::string& task_id,
    const OpenedInfo& opened) {
    if (task_id != active_task_id_) return;
    applyOpenedToRunScreen(opened);
}

void StudioController::applyOpenedToRunScreen(const OpenedInfo& opened) {
    const FrameDescriptor& descriptor = opened.descriptor;
    const bool hardware_decode =
        opened.statistics.reference.decode_backend != StreamDecodeBackend::software ||
        opened.statistics.distorted.decode_backend != StreamDecodeBackend::software;
    setText(TextSlot::run_backend, [hardware_decode] {
        return hardware_decode ? std::string("Vulkan Video + GPU")
                               : i18n::tr("Software decoding + Vulkan GPU");
    });
    std::string details =
        std::to_string(descriptor.width) + " × " +
        std::to_string(descriptor.height) + " · " +
        std::to_string(descriptor.bit_depth) + " bit · " +
        opened.device.name;
    if (opened.media.reference.frame_rate.average.valid()) {
        details += " · " +
            formatDecimal(
                opened.media.reference.frame_rate.average.value(), 3) +
            " fps";
    }
    setText(TextSlot::run_status,
            [details] { return i18n::tr("Inputs validated · {}", details); });
    std::vector<slint::SharedString> reference_chips{
        sharedString(videoCodecName(opened.statistics.reference.codec)),
        sharedString(std::to_string(descriptor.width) + " × " +
            std::to_string(descriptor.height)),
        sharedString(decodeBackendName(
            opened.statistics.reference.decode_backend)),
    };
    std::vector<slint::SharedString> distorted_chips{
        sharedString(videoCodecName(opened.statistics.distorted.codec)),
        sharedString(std::to_string(descriptor.width) + " × " +
            std::to_string(descriptor.height)),
        sharedString(decodeBackendName(
            opened.statistics.distorted.decode_backend)),
    };
    view_model_.set_reference_chips(vectorModel(std::move(reference_chips)));
    view_model_.set_distorted_chips(vectorModel(std::move(distorted_chips)));
}

void StudioController::onTaskProgress(const ProgressSnapshot& progress) {
    last_progress_ = progress;
    applyProgressToRunScreen(progress);
    report_presenter_.applyLive(view_model_, live_scores_, progress);
}

void StudioController::onTaskFramesAppended(std::vector<double> scores) {
    live_scores_.insert(
        live_scores_.end(),
        std::make_move_iterator(scores.begin()),
        std::make_move_iterator(scores.end()));
}

void StudioController::applyProgressToRunScreen(
    const ProgressSnapshot& progress) {
    const std::uint64_t completed = progress.completed_frames;
    if (progress.frame_count_hint.has_value() &&
        *progress.frame_count_hint > 0U) {
        const double fraction = std::min(
            1.0,
            static_cast<double>(completed) /
                static_cast<double>(*progress.frame_count_hint));
        view_model_.set_run_frac(static_cast<float>(fraction));
        setText(TextSlot::run_percent,
                [text = formatDecimal(fraction * 100.0, 1) + "%"] { return text; });
        setText(TextSlot::run_frames,
                [text = formatInteger(static_cast<double>(completed)) + " / " +
                     formatInteger(static_cast<double>(*progress.frame_count_hint))] {
                    return text;
                });
        const double remaining_seconds = progress.frames_per_second > 0.0
            ? static_cast<double>(*progress.frame_count_hint -
                std::min<std::uint64_t>(*progress.frame_count_hint, completed)) /
                progress.frames_per_second
            : 0.0;
        setText(TextSlot::run_eta,
                [text = remaining_seconds > 0.0
                     ? formatDecimal(remaining_seconds, 1) + " s"
                     : std::string("—")] { return text; });
    } else {
        setText(TextSlot::run_percent, [] { return i18n::tr("Running"); });
        view_model_.set_run_frac(0.0F);
        setText(TextSlot::run_frames, [completed] {
            return i18n::trn("{0} frame", "{0} frames", completed,
                             formatInteger(static_cast<double>(completed)));
        });
        setText(TextSlot::run_eta, [] { return i18n::trc("remaining time", "Unknown"); });
    }
    const std::string rate = progress.frames_per_second > 0.0
        ? formatDecimal(progress.frames_per_second, 1) + " fps"
        : "—";
    if (progress.source_frames_per_second.has_value() &&
        *progress.source_frames_per_second > 0.0 &&
        progress.frames_per_second > 0.0) {
        setText(TextSlot::run_speed,
                [rate, ratio = formatDecimal(
                     progress.frames_per_second / *progress.source_frames_per_second, 2)] {
                    return i18n::tr("{0} · {1}× speed", rate, ratio);
                });
    } else {
        setText(TextSlot::run_speed, [rate] { return rate; });
    }
    view_model_.set_run_elapsed(sharedString(
        formatDecimal(progress.elapsed_seconds, 1) + " s"));
}

void StudioController::onTaskCompleted(
    std::shared_ptr<const EvaluationReport> report) {
    session_reports_.insert(session_reports_.begin(), report);
    if (session_reports_.size() > 20U) session_reports_.resize(20U);
    // Evaluations started for the Compare screen report there; they never
    // replace the report the user may be reading.
    const bool for_comparison = comparison_task_ids_.contains(report->report_id);
    if (!for_comparison) showReport(report);
    if (comparison_.complete(report)) {
        updateComparisonModel();
        saveComparison();
        if (comparison_.active_count() == 0U && view_model_.get_screen() != 3) {
            showActionMessage(
                [name = fileDisplayName(comparison_.reference_path())] {
                    return i18n::tr("The comparison of {} is ready on the Compare page", name);
                },
                false);
        }
    }

    if (report->report_id == active_task_id_) {
        // The report is authoritative even when the last incremental UI
        // update was throttled or the stream finished without a tail batch.
        live_scores_ = report_scores(*report);
        applyOpenedToRunScreen(OpenedInfo{
            report->descriptor, report->device, report->media,
            report->stream_statistics});
        ProgressSnapshot completed;
        completed.completed_frames = report->frames.size();
        completed.frame_count_hint = report->frames.size();
        completed.decoded_frame_count = report->frames.size();
        completed.elapsed_seconds = report->processing_milliseconds / 1000.0;
        completed.frames_per_second = report->frames_per_second;
        if (report->media.reference.frame_rate.average.valid()) {
            completed.source_frames_per_second =
                report->media.reference.frame_rate.average.value();
        }
        onTaskProgress(completed);
        view_model_.set_run_active(false);
        view_model_.set_run_stopping(false);
        view_model_.set_run_done(true);
        setText(TextSlot::run_title, [] { return i18n::tr("Evaluation complete"); });
        setText(TextSlot::run_status,
                [] { return i18n::tr("Evaluation complete; real results are ready"); });
        view_model_.set_run_status_kind(0);
        // The worker may still hold this completed task while its callback
        // is handled. Only other runnable tasks keep the queue indicator on.
        const auto tasks = task_queue_.list_tasks();
        const bool queue_running = task_queue_.queue_active();
        const bool more_tasks = std::any_of(tasks.begin(), tasks.end(),
            [&](const TaskSnapshot& task) {
                return task.task_id != report->report_id &&
                    ((task.state == TaskState::queued && queue_running) ||
                     task.state == TaskState::opening ||
                     task.state == TaskState::running ||
                     task.state == TaskState::finishing ||
                     task.state == TaskState::cancelling);
            });
        view_model_.set_nav_live(more_tasks);
        // Someone watching this run gets its report; other pages stay put.
        if (!more_tasks && !for_comparison && view_model_.get_screen() == 1) {
            view_model_.set_screen(2);
        }
        active_task_id_.clear();
    }

    // Persist in the background; the in-memory report already drives the UI.
    if (background_thread_.joinable()) background_thread_.join();
    background_thread_ = std::thread([this, report] {
        const auto saved = report_store_.save_report(*report);
        postToUi([this, saved] {
            if (!saved) {
                showActionMessage(
                    [message = saved.error().message] {
                        return i18n::tr("Saving the report to the local library failed: {}",
                                        message);
                    },
                    true);
            }
            updateRecentModel();
            applyComparisonCandidates();
        });
    });
}

void StudioController::updateQueueModel() {
    task_rows_ = task_queue_.list_tasks();
    std::vector<QueueItem> rows;
    rows.reserve(task_rows_.size());
    std::size_t terminal_shown = 0U;
    for (const TaskSnapshot& task : task_rows_) {
        const bool terminal = task.state == TaskState::completed ||
            task.state == TaskState::failed ||
            task.state == TaskState::cancelled;
        if (terminal && ++terminal_shown > 8U) break;
        QueueItem row;
        row.pair = sharedString(pairDisplayName(task.request));
        row.in_comparison = comparison_.owns_task(task.task_id) ||
            comparison_.selection().contains(task.task_id);
        std::string status(task_state_name(task.state));
        if (task.state == TaskState::running ||
            task.state == TaskState::finishing) {
            if (task.frame_count_hint.has_value() &&
                *task.frame_count_hint > 0U) {
                status += " · " + formatDecimal(
                    static_cast<double>(task.completed_frames) /
                        static_cast<double>(*task.frame_count_hint) * 100.0,
                    0) + "%";
            }
        }
        if (task.attempt > 1U) {
            status = i18n::tr("{} · attempt {}", status, std::to_string(task.attempt));
        }
        row.status = sharedString(status);
        switch (task.state) {
        case TaskState::running:
        case TaskState::finishing:
            row.status_kind = 1;
            row.action_kind = 1;
            break;
        case TaskState::opening:
        case TaskState::cancelling:
            row.status_kind = 1;
            row.action_kind = task.state == TaskState::opening ? 1 : 0;
            break;
        case TaskState::failed:
            row.status_kind = 2;
            row.action_kind = 3;
            break;
        case TaskState::cancelled:
            row.status_kind = 0;
            row.action_kind = 3;
            break;
        case TaskState::completed:
            row.status_kind = 4;
            row.action_kind = 0;
            break;
        case TaskState::queued:
        default:
            row.status_kind = 0;
            row.action_kind = 2;
            break;
        }
        rows.push_back(row);
    }
    view_model_.set_queue_items(vectorModel(std::move(rows)));
    view_model_.set_queue_active(task_queue_.queue_active());
}

void StudioController::updateRecentModel(std::size_t limit) {
    const auto page = report_store_.list_reports(0U, limit);
    if (page) {
        recent_summaries_ = page.value().items;
        recent_cursor_ = page.value().next_cursor;
    } else {
        recent_summaries_.clear();
        recent_cursor_.reset();
        showActionMessage(page.error().message, true);
    }
    applyRecentModel();
}

void StudioController::loadMoreRecentReports() {
    if (!recent_cursor_) return;
    const auto page = report_store_.list_reports(*recent_cursor_, 20U);
    if (!page) {
        showActionMessage(page.error().message, true);
        return;
    }
    recent_summaries_.insert(recent_summaries_.end(),
        page.value().items.begin(), page.value().items.end());
    recent_cursor_ = page.value().next_cursor;
    applyRecentModel();
}

void StudioController::applyRecentModel() {
    std::vector<RecentItem> recent_items;
    recent_items.reserve(recent_summaries_.size());
    for (const ReportSummary& summary : recent_summaries_) {
        RecentItem item;
        item.report_id = sharedString(summary.report_id);
        item.name = sharedString(summary.distorted_name);
        item.score = sharedString(formatDecimal(summary.pooled_score, 1));
        item.score_kind = qualityBandKind(summary.pooled_score);
        item.scene = sharedString(scene_display_name(summary.scene));
        item.details = sharedString(i18n::trn(
            "{0} frame · reference {1}", "{0} frames · reference {1}",
            summary.frame_count, std::to_string(summary.frame_count),
            summary.reference_name));
        recent_items.push_back(item);
    }
    view_model_.set_recent(vectorModel(std::move(recent_items)));
    view_model_.set_recent_has_more(recent_cursor_.has_value());
}

// The comparison saved at its last change, or, when none was ever saved, a
// reference of the library that was evaluated more than once.
void StudioController::initializeComparison() {
    if (const auto saved = report_store_.load_comparison()) {
        comparison_.start(saved->reference_path, saved->scene);
        for (const std::string& report_id : saved->report_ids) {
            if (auto report = loadReport(report_id)) {
                static_cast<void>(comparison_.selection().add(std::move(report).value()));
            }
        }
        comparison_.selection().set_baseline(saved->baseline_report_id);
    } else {
        seedComparisonFromLibrary();
    }
    updateComparisonModel();
}

void StudioController::updateComparisonModel() {
    last_comparison_ = comparison_.selection().build();
    report_presenter_.setComparison(comparison_, last_comparison_);
    report_presenter_.applyComparison(view_model_);
    applyComparisonStatus();
    // With nothing to compare, the page starts at the set-up step.
    if (comparison_.empty() && !comparison_setup_.active()) {
        openComparisonSetup(false);
    } else if (comparison_setup_.active()) {
        view_model_.set_cmps_can_cancel(!comparison_.empty());
    }
}

void StudioController::applyComparisonProgress() {
    report_presenter_.setComparisonPending(comparison_.pending());
    report_presenter_.applyComparisonTracks(view_model_);
    applyComparisonStatus();
}

void StudioController::applyComparisonStatus() {
    const std::size_t active = comparison_.active_count();
    const bool paused = active > 0U && !task_queue_.queue_active();
    view_model_.set_cmp_live(active > 0U && !paused);
    view_model_.set_cmp_queue_paused(paused);
    std::string progress;
    if (paused) {
        progress = i18n::trn(
            "{0} video is waiting; the task queue is paused",
            "{0} videos are waiting; the task queue is paused",
            active, std::to_string(active));
    } else if (active > 0U) {
        progress = i18n::trn(
            "{0} video is still being evaluated",
            "{0} videos are still being evaluated",
            active, std::to_string(active));
    }
    view_model_.set_cmp_progress(sharedString(progress));
}

void StudioController::saveComparison() {
    SavedComparison saved;
    saved.reference_path = comparison_.reference_path();
    saved.scene = comparison_.scene();
    for (const auto& entry : comparison_.selection().entries()) {
        saved.report_ids.push_back(entry.report->report_id);
    }
    saved.baseline_report_id = comparison_.selection().baseline_report_id();
    // A failed write only forgets which reports were compared; the reports
    // themselves are saved separately.
    static_cast<void>(report_store_.save_comparison(saved));
}

// Start from the newest saved report whose reference was evaluated more
// than once, or else the newest report.
void StudioController::seedComparisonFromLibrary() {
    const auto page = report_store_.list_reports(0U, 0U);
    if (!page || page.value().items.empty()) return;
    const std::vector<ReportSummary>& items = page.value().items;
    const auto has_peer = [&items](const ReportSummary& summary) {
        return std::any_of(items.begin(), items.end(),
            [&summary](const ReportSummary& other) {
                return other.report_id != summary.report_id &&
                    other.reference_name == summary.reference_name &&
                    other.frame_count == summary.frame_count &&
                    other.model == summary.model;
            });
    };
    const auto chosen = std::find_if(items.begin(), items.end(), has_peer);
    auto anchor =
        loadReport((chosen != items.end() ? *chosen : items.front()).report_id);
    if (!anchor) return;
    const auto& report = anchor.value();
    comparison_.start(report->request.reference_path, report->request.scene);
    comparison_.selection().reset(report, reportsRelatedTo(*report));
}

// Other reports of the anchor's reference, newest first: this session's,
// then saved ones whose summary can match. Parsing a saved report is not
// free, so only as many are loaded as can join the anchor.
std::vector<std::shared_ptr<const EvaluationReport>>
StudioController::reportsRelatedTo(const EvaluationReport& anchor) {
    constexpr std::size_t maximum_related = ComparisonSelection::maximum_reports - 1U;
    std::vector<std::shared_ptr<const EvaluationReport>> related;
    const auto is_listed = [&related](const std::string& report_id) {
        return std::any_of(related.begin(), related.end(),
            [&report_id](const auto& report) {
                return report->report_id == report_id;
            });
    };
    for (const auto& report : session_reports_) {
        if (report->report_id != anchor.report_id &&
            report->request.reference_path == anchor.request.reference_path) {
            related.push_back(report);
        }
    }
    const auto page = report_store_.list_reports(0U, 0U);
    if (!page) return related;
    const std::string reference_name =
        fileDisplayName(anchor.request.reference_path);
    for (const ReportSummary& summary : page.value().items) {
        if (related.size() >= maximum_related) break;
        if (summary.report_id == anchor.report_id ||
            summary.reference_name != reference_name ||
            summary.frame_count != anchor.frames.size() ||
            summary.model != anchor.model || is_listed(summary.report_id)) {
            continue;
        }
        if (auto report = loadReport(summary.report_id)) {
            related.push_back(std::move(report).value());
        }
    }
    return related;
}

Result<std::shared_ptr<const EvaluationReport>> StudioController::loadReport(
    const std::string& report_id) const {
    using Loaded = Result<std::shared_ptr<const EvaluationReport>>;
    for (const auto& report : session_reports_) {
        if (report->report_id == report_id) return Loaded::success(report);
    }
    auto loaded = report_store_.load_report(report_id);
    if (!loaded) return Loaded::failure(loaded.error());
    return Loaded::success(std::make_shared<const EvaluationReport>(
        std::move(loaded).value()));
}

std::string StudioController::comparisonTrackId(std::int32_t track) const {
    if (track < 0) return {};
    const auto index = static_cast<std::size_t>(track);
    const auto& entries = comparison_.selection().entries();
    if (index < entries.size()) return entries[index].report->report_id;
    const auto& pending = comparison_.pending();
    if (index - entries.size() < pending.size()) {
        return pending[index - entries.size()].task_id;
    }
    return {};
}

void StudioController::setComparisonBaseline(std::int32_t track) {
    const auto& entries = comparison_.selection().entries();
    if (track < 0 || static_cast<std::size_t>(track) >= entries.size()) return;
    comparison_.selection().set_baseline(
        entries[static_cast<std::size_t>(track)].report->report_id);
    updateComparisonModel();
    saveComparison();
}

void StudioController::removeComparisonTrack(std::int32_t track) {
    const std::string id = comparisonTrackId(track);
    if (id.empty()) return;
    if (comparison_.owns_task(id)) {
        // A waiting evaluation leaves the queue; a running one is stopped.
        if (!task_queue_.remove_queued(id)) static_cast<void>(task_queue_.cancel(id));
    }
    comparison_.remove(id);
    updateQueueModel();
    updateComparisonModel();
    saveComparison();
}

void StudioController::retryComparisonTrack(std::int32_t track) {
    const std::string id = comparisonTrackId(track);
    if (id.empty() || !comparison_.owns_task(id) || !task_queue_.retry(id)) return;
    task_queue_.start_queue();
    updateQueueModel();
    applyComparisonStatus();
}

void StudioController::openComparisonTrack(std::int32_t track) {
    const auto& entries = comparison_.selection().entries();
    if (track < 0 || static_cast<std::size_t>(track) >= entries.size()) return;
    showReport(entries[static_cast<std::size_t>(track)].report);
    view_model_.set_report_from_compare(true);
    view_model_.set_screen(2);
}

// Stopped videos stay listed and can be retried.
void StudioController::stopComparison() {
    for (const auto& track : comparison_.pending()) {
        if (task_state_active(track.state)) {
            static_cast<void>(task_queue_.cancel(track.task_id));
        }
    }
    updateQueueModel();
}

void StudioController::onComparisonTaskProgress(
    const std::string& task_id, const ProgressSnapshot& progress) {
    if (comparison_.update_progress(task_id, progress)) applyComparisonProgress();
}

// Editing starts from the comparison's reference and videos; a new
// comparison starts empty and selects its model from the reference video.
void StudioController::openComparisonSetup(bool editing) {
    editing = editing && !comparison_.empty();
    comparison_setup_.start(editing);
    if (editing) {
        setComparisonSetupReference(comparison_.reference_path());
        for (const auto& entry : comparison_.selection().entries()) {
            comparison_setup_.add_included(
                entry.report->report_id, entry.report->request.distorted_path, entry.report);
        }
        for (const auto& track : comparison_.pending()) {
            comparison_setup_.add_included(track.task_id, track.distorted_path, nullptr);
        }
    }
    view_model_.set_cmp_setup_open(true);
    view_model_.set_action_message_visible(false);
    applyComparisonSetup();
    applyComparisonCandidates();
}

void StudioController::closeComparisonSetup() {
    comparison_setup_.clear();
    comparison_candidates_.clear();
    view_model_.set_cmp_candidates(vectorModel(std::vector<CmpCandidate>{}));
    view_model_.set_cmp_setup_open(false);
}

void StudioController::setComparisonSetupReference(std::filesystem::path path) {
    const std::uint64_t version =
        probe_service_.probe(InputRole::reference, path, comparison_reference_slot);
    comparison_setup_.set_reference(std::move(path), version);
}

void StudioController::addComparisonSetupFiles(
    const std::vector<std::filesystem::path>& paths) {
    TextRenderer problem;
    std::size_t left_out = 0U;
    for (const std::filesystem::path& path : paths) {
        switch (comparison_setup_.add_file(path)) {
        case ComparisonSetup::AddOutcome::added: {
            const std::uint64_t row_id = comparison_setup_.last_row_id();
            comparison_setup_.set_row_probe_version(
                row_id, probe_service_.probe(InputRole::distorted, path, row_id));
            break;
        }
        case ComparisonSetup::AddOutcome::duplicate:
            if (!problem) {
                problem = [name = fileDisplayName(path)] {
                    return i18n::tr("{} is already in the list", name);
                };
            }
            break;
        case ComparisonSetup::AddOutcome::same_as_reference:
            if (!problem) {
                problem = [name = fileDisplayName(path)] {
                    return i18n::tr("{} is the reference video", name);
                };
            }
            break;
        case ComparisonSetup::AddOutcome::full:
            ++left_out;
            break;
        }
    }
    if (left_out > 0U) {
        problem = [left_out] {
            return i18n::trn(
                "Up to {1} videos can be compared at once; {0} video was not added",
                "Up to {1} videos can be compared at once; {0} videos were not added",
                left_out, std::to_string(left_out),
                std::to_string(ComparisonSelection::maximum_reports));
        };
    }
    if (problem) {
        showActionMessage(std::move(problem), true);
    } else {
        view_model_.set_action_message_visible(false);
    }
    applyComparisonSetup();
    applyComparisonCandidates();
}

void StudioController::handleComparisonProbe(ProbeService::ProbeOutcome outcome) {
    const bool reference = outcome.role == InputRole::reference;
    const bool changed = reference
        ? comparison_setup_.apply_reference_probe(outcome.version, std::move(outcome.metadata))
        : comparison_setup_.apply_row_probe(
              outcome.slot, outcome.version, std::move(outcome.metadata));
    if (!changed) return;
    findComparisonSavedResults();
    applyComparisonSetup();
    // Candidates depend on the model, which needs the reference's frame rate.
    if (reference) applyComparisonCandidates();
}

// A saved evaluation of the same unchanged files is used instead of
// evaluating again. Only saved reports with matching file names are loaded.
void StudioController::findComparisonSavedResults() {
    const ComparisonSetup::Reference& reference = comparison_setup_.reference();
    if (reference.probe != ComparisonSetup::ProbeState::ready) return;
    const std::string reference_name = fileDisplayName(reference.path);
    std::optional<std::vector<ReportSummary>> summaries;
    std::vector<std::pair<std::uint64_t, std::shared_ptr<const EvaluationReport>>> matches;
    for (const ComparisonSetup::Row& row : comparison_setup_.rows()) {
        if (row.from_library || !row.track_id.empty() || row.saved ||
            row.probe != ComparisonSetup::ProbeState::ready) {
            continue;
        }
        std::shared_ptr<const EvaluationReport> match;
        for (const auto& report : session_reports_) {
            if (comparison_setup_.matches_saved(row, *report)) {
                match = report;
                break;
            }
        }
        if (!match) {
            if (!summaries) {
                const auto page = report_store_.list_reports(0U, 0U);
                summaries = page ? page.value().items : std::vector<ReportSummary>{};
            }
            const std::string name = fileDisplayName(row.path);
            for (const ReportSummary& summary : *summaries) {
                if (summary.distorted_name != name ||
                    summary.reference_name != reference_name) {
                    continue;
                }
                auto loaded = loadReport(summary.report_id);
                if (loaded && comparison_setup_.matches_saved(row, *loaded.value())) {
                    match = std::move(loaded).value();
                    break;
                }
            }
        }
        if (match) matches.emplace_back(row.id, std::move(match));
    }
    for (auto& [row_id, report] : matches) {
        comparison_setup_.set_saved(row_id, std::move(report));
    }
}

const ComparisonSetup::Row* StudioController::comparisonSetupRow(std::int32_t index) const {
    const auto& rows = comparison_setup_.rows();
    if (index < 0 || static_cast<std::size_t>(index) >= rows.size()) return nullptr;
    return &rows[static_cast<std::size_t>(index)];
}

void StudioController::applyComparisonSetup() {
    const ComparisonSetup& setup = comparison_setup_;
    const ComparisonSetup::Reference& reference = setup.reference();
    view_model_.set_cmps_editing(setup.editing());
    view_model_.set_cmps_can_cancel(!comparison_.empty());
    view_model_.set_cmps_reference_selected(!reference.path.empty());
    if (reference.path.empty()) {
        view_model_.set_cmps_reference_name(sharedString(i18n::tr("Choose the reference video")));
        view_model_.set_cmps_reference_details(sharedString(
            i18n::tr("The original video that every version was encoded from")));
        view_model_.set_cmps_reference_kind(0);
    } else {
        view_model_.set_cmps_reference_name(sharedString(fileDisplayName(reference.path)));
        switch (reference.probe) {
        case ComparisonSetup::ProbeState::pending:
            view_model_.set_cmps_reference_details(sharedString(i18n::tr("Probing…")));
            view_model_.set_cmps_reference_kind(0);
            break;
        case ComparisonSetup::ProbeState::ready:
            view_model_.set_cmps_reference_details(
                sharedString(metadataSummary(*reference.metadata)));
            view_model_.set_cmps_reference_kind(1);
            break;
        case ComparisonSetup::ProbeState::failed:
            view_model_.set_cmps_reference_details(sharedString(
                i18n::tr("The video could not be read: {}", reference.error)));
            view_model_.set_cmps_reference_kind(2);
            break;
        }
    }

    const auto saved_details = [](const EvaluationReport& report) {
        return i18n::tr("VMAF {0} · evaluated {1}",
                        formatDecimal(report.pooled_score, 1),
                        formatTimestamp(report.completed_at));
    };
    std::vector<CmpSetupRow> rows;
    for (const ComparisonSetup::Row& row : setup.rows()) {
        CmpSetupRow item;
        item.name = sharedString(fileDisplayName(row.path));
        switch (setup.status(row)) {
        case ComparisonSetup::RowStatus::checking:
            item.status = sharedString(i18n::tr("Checking"));
            item.status_kind = 0;
            item.details = sharedString(setup.message(row));
            break;
        case ComparisonSetup::RowStatus::ready: {
            item.status = sharedString(i18n::tr("To evaluate"));
            item.status_kind = 1;
            const std::string note = setup.message(row);
            item.details = sharedString(note.empty() ? metadataSummary(*row.metadata) : note);
            item.action = row.saved ? 2 : 0;
            break;
        }
        case ComparisonSetup::RowStatus::saved:
            item.status = sharedString(i18n::tr("Saved result"));
            item.status_kind = 3;
            item.details = sharedString(saved_details(*row.saved));
            item.action = row.from_library ? 0 : 1;
            break;
        case ComparisonSetup::RowStatus::included:
            item.status_kind = 3;
            if (row.saved) {
                item.status = sharedString(i18n::tr("Compared"));
                item.details = sharedString(saved_details(*row.saved));
            } else {
                item.status = sharedString(i18n::tr("Not finished"));
                item.details = sharedString(
                    i18n::tr("Its evaluation has not finished; see the comparison"));
            }
            break;
        case ComparisonSetup::RowStatus::invalid:
            item.status = sharedString(i18n::tr("Cannot be compared"));
            item.status_kind = 2;
            item.details = sharedString(setup.message(row));
            break;
        }
        rows.push_back(item);
    }
    view_model_.set_cmps_rows(vectorModel(std::move(rows)));

    const bool ready = setup.ready();
    view_model_.set_cmps_ready(ready);
    view_model_.set_cmps_status(sharedString(setup.status_message()));
    view_model_.set_cmps_status_kind(ready ? 1
        : reference.probe == ComparisonSetup::ProbeState::failed ||
                setup.count(ComparisonSetup::RowStatus::invalid) > 0U
            ? 2 : 0);
    view_model_.set_cmps_start_label(sharedString(
        setup.editing() ? i18n::tr("Update comparison")
        : setup.count(ComparisonSetup::RowStatus::ready) == 0U ? i18n::tr("Compare")
        : i18n::tr("Evaluate and compare")));
}

// Saved evaluations of the set-up reference that are not in the list yet.
void StudioController::applyComparisonCandidates() {
    constexpr std::size_t maximum_candidates = 20U;
    comparison_candidates_.clear();
    std::vector<CmpCandidate> rows;
    const ComparisonSetup::Reference& reference = comparison_setup_.reference();
    const auto page = comparison_setup_.active() && !reference.path.empty()
        ? report_store_.list_reports(0U, 0U)
        : Result<ReportListPage>::success({});
    if (page) {
        const std::string reference_name = fileDisplayName(reference.path);
        const bool model_known = reference.probe == ComparisonSetup::ProbeState::ready;
        const ModelVariant model = comparison_setup_.model();
        // Finished evaluations in a comparison all have the same frame count.
        std::optional<std::size_t> frame_count;
        for (const ComparisonSetup::Row& row : comparison_setup_.rows()) {
            if (row.saved) {
                frame_count = row.saved->frames.size();
                break;
            }
        }
        for (const ReportSummary& summary : page.value().items) {
            if (comparison_candidates_.size() >= maximum_candidates) break;
            if (summary.reference_name != reference_name ||
                comparison_setup_.contains_report(summary.report_id)) {
                continue;
            }
            std::string reason;
            if (frame_count && summary.frame_count != *frame_count) {
                reason = i18n::tr("Different frame count");
            } else if (model_known && summary.model != model) {
                reason = i18n::tr("Different quality model");
            }
            CmpCandidate row;
            row.name = sharedString(summary.distorted_name);
            row.details = sharedString(i18n::trn(
                "{0} frame · {1}", "{0} frames · {1}", summary.frame_count,
                formatInteger(static_cast<double>(summary.frame_count)),
                formatTimestamp(std::chrono::system_clock::time_point(
                    std::chrono::seconds(summary.completed_at_utc_seconds)))));
            row.score = sharedString(formatDecimal(summary.pooled_score, 1));
            row.available = reason.empty();
            row.reason = sharedString(reason);
            comparison_candidates_.push_back(summary);
            rows.push_back(row);
        }
    }
    view_model_.set_cmp_candidates(vectorModel(std::move(rows)));
}

void StudioController::addSavedToComparisonSetup(std::size_t index) {
    if (index >= comparison_candidates_.size()) return;
    const ReportSummary summary = comparison_candidates_[index];
    auto loaded = loadReport(summary.report_id);
    if (!loaded) {
        showActionMessage(loaded.error().message, true);
        return;
    }
    switch (comparison_setup_.add_saved(std::move(loaded).value())) {
    case ComparisonSetup::AddOutcome::added:
        view_model_.set_action_message_visible(false);
        break;
    case ComparisonSetup::AddOutcome::full:
        showActionMessage([] {
            return i18n::trn("Up to {0} video can be compared at once",
                             "Up to {0} videos can be compared at once",
                             ComparisonSelection::maximum_reports,
                             std::to_string(ComparisonSelection::maximum_reports));
        }, true);
        break;
    case ComparisonSetup::AddOutcome::duplicate:
    case ComparisonSetup::AddOutcome::same_as_reference:
        showActionMessage([name = summary.distorted_name] {
            return i18n::tr("{} is already in the list", name);
        }, true);
        break;
    }
    applyComparisonSetup();
    applyComparisonCandidates();
}

// Saved results join the comparison at once; the other videos are queued
// and join it as they finish.
void StudioController::startComparisonSetup() {
    if (!comparison_setup_.ready()) return;
    const ComparisonSetup::Reference& reference = comparison_setup_.reference();
    const std::filesystem::path reference_path = reference.path;

    // Files changed since they were checked are checked again first.
    bool changed = reference.metadata->identity.captured &&
        !captureIdentity(reference_path).matches(reference.metadata->identity);
    if (changed) setComparisonSetupReference(reference_path);
    for (const ComparisonSetup::Row& row : comparison_setup_.rows()) {
        if (comparison_setup_.status(row) != ComparisonSetup::RowStatus::ready ||
            !row.metadata->identity.captured ||
            captureIdentity(row.path).matches(row.metadata->identity)) {
            continue;
        }
        comparison_setup_.set_row_probe_version(
            row.id, probe_service_.probe(InputRole::distorted, row.path, row.id));
        changed = true;
    }
    if (changed) {
        showActionMessage(
            [] { return i18n::tr("A file changed after it was checked; checking it again"); },
            true);
        applyComparisonSetup();
        return;
    }

    if (!comparison_setup_.editing()) {
        comparison_.start(reference_path, comparison_setup_.scene());
    } else {
        // Videos removed from the list leave the comparison.
        std::vector<std::string> removed;
        for (const auto& entry : comparison_.selection().entries()) {
            if (!comparison_setup_.contains_report(entry.report->report_id)) {
                removed.push_back(entry.report->report_id);
            }
        }
        for (const auto& track : comparison_.pending()) {
            if (!comparison_setup_.contains_report(track.task_id)) {
                removed.push_back(track.task_id);
            }
        }
        for (const std::string& id : removed) {
            if (comparison_.owns_task(id) && !task_queue_.remove_queued(id)) {
                static_cast<void>(task_queue_.cancel(id));
            }
            comparison_.remove(id);
        }
    }

    bool enqueued = false;
    TextRenderer problem;
    for (const ComparisonSetup::Row& row : comparison_setup_.rows()) {
        switch (comparison_setup_.status(row)) {
        case ComparisonSetup::RowStatus::saved:
            if (auto issue = comparison_.selection().add(row.saved); issue && !problem) {
                problem = [name = fileDisplayName(row.path), reason = issue->message] {
                    return i18n::tr("{0} was not added: {1}", name, reason);
                };
            }
            break;
        case ComparisonSetup::RowStatus::ready: {
            const std::string task_id = task_queue_.enqueue(buildRequest(
                reference_path, &*reference.metadata, row.path, &*row.metadata));
            comparison_task_ids_.insert(task_id);
            comparison_.add_pending(task_id, row.path);
            enqueued = true;
            break;
        }
        case ComparisonSetup::RowStatus::checking:
        case ComparisonSetup::RowStatus::included:
        case ComparisonSetup::RowStatus::invalid:
            break;
        }
    }
    closeComparisonSetup();
    if (enqueued) task_queue_.start_queue();
    updateQueueModel();
    updateComparisonModel();
    saveComparison();
    if (problem) {
        showActionMessage(std::move(problem), true);
    } else {
        view_model_.set_action_message_visible(false);
    }
}

// The displayed report joins the comparison of its reference, or starts a
// new one; either way the set-up step opens so more videos can be added.
void StudioController::compareDisplayedReport() {
    const auto report = report_presenter_.report();
    if (!report) return;
    view_model_.set_screen(3);
    if (comparison_.selection().contains(report->report_id)) {
        if (comparison_setup_.active() && !comparison_.empty()) closeComparisonSetup();
        return;
    }
    const bool same_reference = !comparison_.empty() &&
        same_file(comparison_.reference_path(), report->request.reference_path);
    openComparisonSetup(same_reference);
    if (!same_reference) {
        setComparisonSetupReference(report->request.reference_path);
    }
    if (comparison_setup_.add_saved(report) == ComparisonSetup::AddOutcome::full) {
        showActionMessage([] {
            return i18n::trn("Up to {0} video can be compared at once",
                             "Up to {0} videos can be compared at once",
                             ComparisonSelection::maximum_reports,
                             std::to_string(ComparisonSelection::maximum_reports));
        }, true);
    }
    applyComparisonSetup();
    applyComparisonCandidates();
}

// A new comparison starting from the New screen's files.
void StudioController::compareFromNewScreen() {
    openComparisonSetup(false);
    if (!reference_path_.empty()) setComparisonSetupReference(reference_path_);
    view_model_.set_screen(3);
    if (!distorted_path_.empty()) {
        addComparisonSetupFiles({distorted_path_});
    } else {
        applyComparisonSetup();
        applyComparisonCandidates();
    }
}

void StudioController::selectRecentReport(std::size_t index) {
    if (index >= recent_summaries_.size()) return;
    const auto loaded =
        report_store_.load_report(recent_summaries_[index].report_id);
    if (!loaded) {
        showActionMessage(loaded.error().message, true);
        return;
    }
    auto report = std::make_shared<const EvaluationReport>(
        std::move(loaded).value());
    showReport(report);
    view_model_.set_screen(2);
}

void StudioController::reuseRecentReport(std::size_t index) {
    if (index >= recent_summaries_.size()) return;
    const auto loaded = report_store_.load_report(recent_summaries_[index].report_id);
    if (!loaded) {
        showActionMessage(loaded.error().message, true);
        return;
    }
    const auto& request = loaded.value().request;
    reference_path_ = request.reference_path;
    distorted_path_ = request.distorted_path;
    reference_metadata_.reset();
    distorted_metadata_.reset();
    view_model_.set_action_message_visible(false);
    // Probe the current files, not the report's historical metadata. Versioned
    // outcomes also prevent an older selection's result from enabling Start.
    requestProbe(InputRole::reference);
    requestProbe(InputRole::distorted);
    applySelectedInputs();
    view_model_.set_screen(0);
}

void StudioController::deleteRecentReport(std::size_t index) {
    if (index >= recent_summaries_.size()) return;
    const std::string report_id = recent_summaries_[index].report_id;
    const auto deleted = report_store_.delete_report(report_id);
    if (!deleted) {
        showActionMessage(deleted.error().message, true);
        return;
    }
    if (report_presenter_.report() &&
        report_presenter_.report()->report_id == report_id) {
        report_presenter_.clearReport(view_model_);
        view_model_.set_report_from_compare(false);
    }
    if (comparison_.selection().contains(report_id)) {
        comparison_.remove(report_id);
        updateComparisonModel();
        saveComparison();
    }
    std::vector<std::uint64_t> removed_rows;
    bool setup_changed = false;
    for (const auto& row : comparison_setup_.rows()) {
        if (!row.saved || row.saved->report_id != report_id) continue;
        if (row.from_library || !row.track_id.empty()) {
            removed_rows.push_back(row.id);
        } else {
            comparison_setup_.set_saved(row.id, nullptr);
        }
        setup_changed = true;
    }
    for (const auto row_id : removed_rows) comparison_setup_.remove_row(row_id);
    if (setup_changed) applyComparisonSetup();
    applyComparisonCandidates();
    updateRecentModel(std::max<std::size_t>(20U, recent_summaries_.size()));
    view_model_.set_action_message_visible(false);
}

void StudioController::showReport(
    std::shared_ptr<const EvaluationReport> report) {
    report_presenter_.setReport(std::move(report));
    report_presenter_.applyReport(view_model_);
    view_model_.set_report_from_compare(false);
    view_model_.set_action_message_visible(false);
}

void StudioController::navigateTo(std::int32_t screen) {
    view_model_.set_screen(std::clamp(screen, std::int32_t{0}, std::int32_t{4}));
    if (screen == 3) updateComparisonModel();
}

void StudioController::locateFrame(std::size_t frame_index) {
    const auto& report = report_presenter_.report();
    if (!report || frame_index >= report->frames.size()) return;
    view_model_.set_screen(2);
    view_model_.set_hl_x_frac(static_cast<float>(
        report_presenter_.reportX(frame_index) / 1160.0));
    view_model_.set_hl_y_frac(static_cast<float>(
        report_presenter_.reportY(report->frames[frame_index].vmaf) / 280.0));
    view_model_.set_hl_visible(true);
    slint::Timer::single_shot(std::chrono::milliseconds(3600), [this] {
        if (!shutting_down_.load(std::memory_order_acquire)) {
            view_model_.set_hl_visible(false);
        }
    });
}

void StudioController::exportCurrentReport(ReportFormat format) {
    const auto& report = report_presenter_.report();
    if (!report) return;
    const std::string extension = format == ReportFormat::pdf ? "pdf" :
        format == ReportFormat::json ? "json" : "csv";
    const auto destination = chooseReportDestination(
        pathStemUtf8(report->request.distorted_path) + "-crossvmaf",
        extension,
        format == ReportFormat::pdf ? "PDF" : format == ReportFormat::json ? "JSON" : "CSV");
    if (!destination) return;
    const auto written = writeEvaluationReport(*destination, format, *report);
    if (written) {
        showActionMessage([name = fileDisplayName(*destination)] {
            return i18n::tr("Exported: {}", name);
        }, false);
    } else {
        showActionMessage(written.error().message, true);
    }
}

void StudioController::exportCurrentComparison() {
    if (!last_comparison_.validation.comparable) return;
    const auto destination = chooseReportDestination(
        "crossvmaf-comparison", "csv", "CSV");
    if (!destination) return;
    const auto written =
        writeComparisonReport(*destination, last_comparison_);
    if (written) {
        showActionMessage([name = fileDisplayName(*destination)] {
            return i18n::tr("Exported comparison data: {}", name);
        }, false);
    } else {
        showActionMessage(written.error().message, true);
    }
}

void StudioController::exportTaskDiagnostics(bool current_task) {
    const auto& report = report_presenter_.report();
    std::optional<TaskSnapshot> snapshot =
        task_queue_.task_snapshot(current_task ? active_task_id_ : report ? report->report_id : std::string{});
    if (!snapshot.has_value() && !current_task && report) {
        // Reports loaded from the library no longer share a live task;
        // rebuild an equivalent snapshot from the stored report.
        TaskSnapshot rebuilt;
        rebuilt.task_id = report->report_id;
        rebuilt.request = report->request;
        rebuilt.state = TaskState::completed;
        rebuilt.report_id = report->report_id;
        rebuilt.completed_frames = report->frames.size();
        rebuilt.frame_count_hint =
            report->stream_statistics.frame_count_hint;
        rebuilt.final_statistics = report->stream_statistics;
        rebuilt.device = report->device;
        rebuilt.finished_at = report->completed_at;
        snapshot = std::move(rebuilt);
    }
    if (!snapshot) {
        showActionMessage(
            [] { return i18n::tr("There is no task diagnostic information to export"); },
            true);
        return;
    }
    const auto destination = chooseReportDestination(
        pathStemUtf8(snapshot->request.distorted_path) + "-diagnostics", "json", "JSON");
    if (!destination) return;
    const auto written = export_support_bundle(
        build_task_diagnostics(*snapshot), *destination);
    if (written) {
        showActionMessage([name = fileDisplayName(*destination)] {
            return i18n::tr("Exported diagnostics: {}", name);
        }, false);
    } else {
        showActionMessage(written.error().message, true);
    }
}

void StudioController::runSelfTest() {
    if (background_thread_.joinable()) background_thread_.join();
    view_model_.set_self_test_busy(true);
    setText(TextSlot::self_test_status,
            [] { return i18n::tr("Running the device self-test…"); });
    view_model_.set_self_test_kind(4);
    DeviceSelector selector;
    if (const auto device = selectedDevice(); device.has_value()) {
        selector = *device;
    }
    background_thread_ = std::thread([this, selector] {
        const DeviceSelfTestResult result = run_device_self_test(selector);
        postToUi([this, result, selector] {
            view_model_.set_self_test_busy(false);
            const auto current_device = selectedDevice();
            if ((current_device ? current_device->uuid : std::string{}) != selector.uuid) {
                setText(TextSlot::self_test_status, [] {
                    return i18n::tr("The device changed; run the self-test again");
                });
                view_model_.set_self_test_kind(-1);
                return;
            }
            if (result.passed) {
                view_model_.set_self_test_kind(0);
                setText(TextSlot::self_test_status,
                        [name = result.device.name,
                         milliseconds = formatDecimal(result.elapsed_milliseconds, 0)] {
                            return i18n::tr("Self-test passed · {} · {} ms", name, milliseconds);
                        });
            } else {
                view_model_.set_self_test_kind(3);
                setText(TextSlot::self_test_status, [error = result.error] {
                    return error.has_value()
                        ? i18n::tr("Self-test failed: {}", error->message)
                        : i18n::tr("Self-test failed");
                });
            }
        });
    });
}

void StudioController::showActionMessage(
    TextRenderer message,
    bool is_error) {
    setText(TextSlot::action_message, std::move(message));
    view_model_.set_action_message_kind(is_error ? 2 : 0);
    view_model_.set_action_message_visible(true);
}

void StudioController::showActionMessage(std::string message, bool is_error) {
    showActionMessage([message = std::move(message)] { return message; }, is_error);
}

std::filesystem::path StudioController::settingsPath() const {
    if (!settings_path_.empty()) return settings_path_;
    return SettingsStore::defaultPath();
}

std::filesystem::path StudioController::reportsDirectory() {
#ifdef _WIN32
    if (const char* app_data = std::getenv("APPDATA")) {
        return std::filesystem::path(app_data) / "CrossVMAF" / "reports";
    }
#else
    if (const char* data_home = std::getenv("XDG_DATA_HOME")) {
        return std::filesystem::path(data_home) / "crossvmaf" / "reports";
    }
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) /
            ".local" / "share" / "crossvmaf" / "reports";
    }
#endif
    return std::filesystem::current_path() / "reports";
}

} // namespace crossvmaf::studio
