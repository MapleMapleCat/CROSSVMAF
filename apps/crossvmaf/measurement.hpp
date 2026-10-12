#pragma once

namespace crossvmaf::cli {

struct Options;

// Runs evaluation, optional CPU verification, pooling, and output; returns
// the CLI exit status while preserving the separate failure stages.
int run_measurement(const Options& options);

} // namespace crossvmaf::cli
