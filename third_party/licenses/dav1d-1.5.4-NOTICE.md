# dav1d 1.5.4 notice

CrossVMAF uses dav1d 1.5.4 for AV1 software decoding through FFmpeg's
official `libdav1d` decoder wrapper.

- Upstream: https://code.videolan.org/videolan/dav1d
- Source tag: `1.5.4`
- Tag commit: `54706fc6bc0cdecab7e9593974a4039cc038fca7`
- License: BSD 2-Clause; the complete upstream `COPYING` file is staged as
  `dav1d-BSD-2-Clause.txt` beside distributed executables.

The upstream Meson build produces a static library with 8-, 10-, and 12-bit
decoding and runtime-dispatched x86-64 assembly. Tools, examples, tests, and
documentation builds are disabled. The archive is linked into CrossVMAF's
`crossvmaf-ffmpeg` shared library; no separate dav1d executable or DLL is
required at runtime.

Upstream source files and library objects retain their original copyright
and license notices. Binary distributors remain responsible for applicable
codec patent obligations and for shipping the replaceable FFmpeg shared
library, its license texts, and the corresponding-source pointer.
