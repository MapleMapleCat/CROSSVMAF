#pragma once

// Public diagnostic seams: visible to every translation unit.
#include "evaluation/initialization_observer.hpp"
#include "evaluation/session_telemetry.hpp"
#include "evaluation/stream_opener.hpp"

// Internal per-open machinery: implementation units and explicit test targets
// only. Each part header repeats this guard so direct includes behave the same.
#if defined(CROSSVMAF_EVALUATION_IMPLEMENTATION) || defined(CROSSVMAF_EVALUATION_TESTING)
#include "evaluation/channel.hpp"
#include "evaluation/frame_pair_record.hpp"
#include "evaluation/frame_records.hpp"
#include "evaluation/input_staging_ring.hpp"
#include "evaluation/input_lease_pool.hpp"
#include "evaluation/session_detail.hpp"
#endif
