# FFmpeg 9.0.1 notice

CrossVMAF compiles selected source files from FFmpeg 9.0.1.

The complete untrimmed, unmodified official source tree is stored in
`third_party/ffmpeg`. CrossVMAF keeps that upstream source tree unmodified and
maintains its build integration in `cmake/FFmpeg.cmake`; the archive hash below
identifies its upstream source.
CrossVMAF's CMake targets compile only the configured
video decoder and demuxer closure from that tree. The closure builds as the
`crossvmaf-ffmpeg` shared library and is linked dynamically, so it can be
replaced with a user-rebuilt library independently of the application
(LGPL-2.1 shared-library mechanism).

- Upstream: <https://ffmpeg.org/>
- Source: <https://ffmpeg.org/releases/ffmpeg-9.0.1.tar.xz>
- SHA-256:
  `cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635`
- Selected configuration license: GNU Lesser General Public License version
  2.1 or later.
- License text in the vendored source tree:
  `third_party/ffmpeg/COPYING.LGPLv2.1`.

The CrossVMAF configuration disables GPL, version-3-only, and nonfree
components. It enables the H.264, HEVC, AV1, VP8, VP9, MPEG-1/2 video,
MPEG-4 Part 2, VC-1/WMV3, FLV1, and MJPEG software decoder closure, FFmpeg's
`libdav1d` decoder wrapper, the H.264/HEVC/AV1/VP9 Vulkan Video hardware
accelerators, and the local-file demuxing dependencies documented in
`cmake/FFmpeg.cmake`.

FFmpeg source files and generated library objects remain under their upstream
copyright notices and license. CrossVMAF's BSD-2-Clause-Patent license does not
replace or relicense FFmpeg.
