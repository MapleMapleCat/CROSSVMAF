# Bundled VMAF 1.0.16 models

`source/` contains only the eight model variants supported by CrossVMAF. They
are copied byte-for-byte from Netflix VMAF commit
`78e11b52c8fc1fcc6d15afd6c7479394fb3bc6af` and verified against
`MANIFEST.sha256` during model-pack generation.

The JSON files are build inputs, not runtime configuration. The build converts
every decimal token to signed Q32.32 using round-to-nearest, ties-to-even and
embeds the resulting support vectors in the library. Custom models and runtime
parameter overrides are intentionally rejected in the 1.0.16 engine.

