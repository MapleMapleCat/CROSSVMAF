#include "video/backends/ffmpeg_device.hpp"
namespace crossvmaf::video {
const AVCodec *find_software_codec(int codec_id) {
    return codec_id == AV_CODEC_ID_AV1 ? avcodec_find_decoder_by_name("libdav1d")
                                       : avcodec_find_decoder(static_cast<AVCodecID>(codec_id));
}
} // namespace crossvmaf::video
