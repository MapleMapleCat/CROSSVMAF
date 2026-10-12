#!/usr/bin/env python3
"""Generate deterministic Q32.32 C++ data from the eight pinned JSON models."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
from pathlib import Path


getcontext().prec = 80
Q32_ONE = Decimal(2) ** 32
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
UPSTREAM_COMMIT = "78e11b52c8fc1fcc6d15afd6c7479394fb3bc6af"

EXPECTED_HASHES = {
    "vmaf_v1.0.16_1d5h_2160.json": "634d36e0502cb797384726cdd1a106a2c207827595a9e4958d3dae1bd517b139",
    "vmaf_v1.0.16_3d0h_2160.json": "77f552fd8d5b6c6a2166bfc4470c9368ec68d2425ed40f1cc235b0cc500f33c2",
    "vmaf_v1.0.16_3d0h.json": "e4cf8c147e1368b35497d772920bc92f98c1ad7853c1033d8a836947f427140e",
    "vmaf_v1.0.16_5d0h.json": "64e12d9c1e4ed5d79da7fa515b82fac7247a9e2ccc91e0ebce31d05489500385",
    "vmaf_v1.0.16_hfr_1d5h_2160.json": "0371db55d3de81a760125f6d65eb4386f1d38af6cf1d00c2f48158e3596246d4",
    "vmaf_v1.0.16_hfr_3d0h_2160.json": "1bc502f00aeee11433e2d09468186bbf1bbce56ee650c9bade06c7435ebdd4db",
    "vmaf_v1.0.16_hfr_3d0h.json": "dcfdfab1c00685c1e5f75b6f1078bb4a44c722654e4de090fa5c66784c75241b",
    "vmaf_v1.0.16_hfr_5d0h.json": "590f612c87df94badca58e563b318fa9958eac631ab8aadd73bff8cc1adfc515",
}

VARIANTS = {
    "vmaf_v1.0.16_1d5h_2160": "ModelVariant::v1_0_16_1d5h_2160",
    "vmaf_v1.0.16_3d0h_2160": "ModelVariant::v1_0_16_3d0h_2160",
    "vmaf_v1.0.16_3d0h": "ModelVariant::v1_0_16_3d0h",
    "vmaf_v1.0.16_5d0h": "ModelVariant::v1_0_16_5d0h",
    "vmaf_v1.0.16_hfr_1d5h_2160": "ModelVariant::v1_0_16_hfr_1d5h_2160",
    "vmaf_v1.0.16_hfr_3d0h_2160": "ModelVariant::v1_0_16_hfr_3d0h_2160",
    "vmaf_v1.0.16_hfr_3d0h": "ModelVariant::v1_0_16_hfr_3d0h",
    "vmaf_v1.0.16_hfr_5d0h": "ModelVariant::v1_0_16_hfr_5d0h",
}

FEATURE_NAMES = [
    "Cambi_feature_cambi_score",
    "Speed_chroma_feature_speed_chroma_uv_score",
    "VMAF_integer_feature_adm3_score",
    "VMAF_integer_feature_motion3_score",
]

FEATURE_OPTION_KEYS = [
    {"cambi_high_res_speedup", "cambi_vis_lum_threshold", "cambi_max_val"},
    {
        "speed_kernelscale",
        "speed_prescale",
        "speed_prescale_method",
        "speed_sigma_nn",
        "speed_nn_floor",
        "speed_weight_var_mode",
        "speed_max_val",
    },
    {
        "adm_dlm_weight",
        "adm_enhn_gain_limit",
        "adm_noise_weight",
        "adm_norm_view_dist",
        "adm_ref_display_height",
        "adm_min_val",
        "adm_csf_mode",
    },
    {"motion_max_val", "motion_five_frame_window", "motion_moving_average"},
]

LEGACY_BLENDED_CSF = {
    (Decimal("1.5"), 2160): (
        ("0.01183", "0.025026", "0.04295", "0.058621"),
        ("0.004302", "0.011778", "0.023918", "0.035901"),
    ),
    (Decimal("3.0"), 1080): (
        ("0.01183", "0.025026", "0.04295", "0.058621"),
        ("0.004302", "0.011778", "0.023918", "0.035901"),
    ),
    (Decimal("5.0"), 1080): (
        ("0.004212", "0.014809", "0.029642", "0.047464"),
        ("0.000984", "0.005852", "0.0146", "0.027574"),
    ),
    (Decimal("3.0"), 2160): (
        ("0.00226", "0.01183", "0.025026", "0.04295"),
        ("0.000479", "0.004302", "0.011778", "0.023918"),
    ),
}


def q32(value: Decimal | int) -> int:
    decimal_value = value if isinstance(value, Decimal) else Decimal(value)
    raw = int((decimal_value * Q32_ONE).to_integral_value(rounding=ROUND_HALF_EVEN))
    if raw < INT64_MIN or raw > INT64_MAX:
        raise ValueError(f"Q32.32 constant is out of range: {value}")
    return raw


def binary32(value: Decimal | str) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def adm_csf_raw(nvd: Decimal, display_height: int) -> tuple[list[int], list[int]]:
    try:
        horizontal, diagonal = LEGACY_BLENDED_CSF[(nvd, display_height)]
    except KeyError as error:
        raise ValueError(
            f"unsupported legacy ADM3 CSF geometry: {display_height}@{nvd}H"
        ) from error

    horizontal_raw: list[int] = []
    diagonal_raw: list[int] = []
    for scale in range(4):
        horizontal_bits = 21 if scale == 0 else 32
        diagonal_bits = 23 if scale == 0 else 32
        horizontal_integer = int(binary32(horizontal[scale]) * (1 << horizontal_bits))
        diagonal_integer = int(binary32(diagonal[scale]) * (1 << diagonal_bits))
        horizontal_raw.append(horizontal_integer << (32 - horizontal_bits))
        diagonal_raw.append(diagonal_integer << (32 - diagonal_bits))
    return horizontal_raw, diagonal_raw


def parse_svm(text: str) -> tuple[Decimal, Decimal, list[tuple[Decimal, list[Decimal]]]]:
    fields: dict[str, str] = {}
    support_lines: list[str] = []
    in_support_vectors = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "SV":
            in_support_vectors = True
            continue
        if in_support_vectors:
            support_lines.append(line)
        else:
            key, value = line.split(maxsplit=1)
            fields[key] = value

    if fields.get("svm_type") != "nu_svr" or fields.get("kernel_type") != "rbf":
        raise ValueError("only LIBSVM nu_svr RBF models are supported")
    total_sv = int(fields["total_sv"])
    if total_sv != len(support_lines):
        raise ValueError("LIBSVM total_sv does not match the support-vector count")

    support_vectors: list[tuple[Decimal, list[Decimal]]] = []
    for line in support_lines:
        tokens = line.split()
        coefficient = Decimal(tokens[0])
        features = [Decimal(0)] * 4
        for token in tokens[1:]:
            index_text, value_text = token.split(":", maxsplit=1)
            index = int(index_text)
            if index < 1 or index > 4:
                raise ValueError(f"unexpected support-vector feature index {index}")
            features[index - 1] = Decimal(value_text)
        support_vectors.append((coefficient, features))
    return Decimal(fields["gamma"]), Decimal(fields["rho"]), support_vectors


def read_model(path: Path) -> dict[str, object]:
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    expected = EXPECTED_HASHES.get(path.name)
    if expected is None or digest != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {digest}")

    data = json.loads(raw_bytes.decode("utf-8"), parse_float=Decimal, parse_int=Decimal)
    model = data["model_dict"]
    name = path.stem
    if name not in VARIANTS:
        raise ValueError(f"unsupported model filename: {path.name}")
    if model["model_type"] != "LIBSVMNUSVR":
        raise ValueError(f"unexpected model type in {path.name}")
    if model["norm_type"] != "linear_rescale":
        raise ValueError(f"unexpected normalization in {path.name}")
    if model["feature_names"] != FEATURE_NAMES:
        raise ValueError(f"unexpected feature order in {path.name}")
    feature_options = model["feature_opts_dicts"]
    if len(feature_options) != len(FEATURE_NAMES):
        raise ValueError(f"unexpected feature option count in {path.name}")
    for feature_name, options, allowed_keys in zip(
        FEATURE_NAMES, feature_options, FEATURE_OPTION_KEYS
    ):
        unexpected = set(options) - allowed_keys
        if unexpected:
            raise ValueError(
                f"unexpected options for {feature_name} in {path.name}: "
                f"{sorted(unexpected)}"
            )

    cambi = feature_options[0]
    speed = feature_options[1]
    adm3 = feature_options[2]
    motion3 = feature_options[3]
    scaling_method = speed.get("speed_prescale_method", "nearest")
    if scaling_method not in {"nearest", "bilinear"}:
        raise ValueError(f"unsupported Speed Chroma scaling method in {path.name}")

    slopes = model["slopes"]
    intercepts = model["intercepts"]
    if len(slopes) != 5 or len(intercepts) != 5:
        raise ValueError(f"expected one output and four feature scales in {path.name}")
    gamma, rho, support_vectors = parse_svm(model["model"])
    transform = model["score_transform"]
    if transform.get("enabled") is not True:
        raise ValueError(f"score transform must be enabled in {path.name}")
    clip = model["score_clip"]
    if len(clip) != 2:
        raise ValueError(f"score clip must have two endpoints in {path.name}")

    adm_nvd = adm3.get("adm_norm_view_dist", Decimal(3))
    adm_display_height = int(adm3.get("adm_ref_display_height", Decimal(1080)))
    adm_horizontal_csf, adm_diagonal_csf = adm_csf_raw(
        adm_nvd, adm_display_height
    )

    return {
        "name": name,
        "variant": VARIANTS[name],
        "sha256": digest,
        "hfr": "_hfr_" in name,
        "cambi_high_res_speedup": cambi.get("cambi_high_res_speedup", Decimal(0)),
        "cambi_visibility_threshold": cambi.get(
            "cambi_vis_lum_threshold", Decimal(0)
        ),
        "cambi_maximum": cambi.get("cambi_max_val", Decimal(1000)),
        "speed_kernel_scale": speed.get("speed_kernelscale", Decimal(1)),
        "speed_prescale": speed.get("speed_prescale", Decimal(1)),
        "speed_prescale_method": scaling_method,
        "speed_sigma_nn": speed.get("speed_sigma_nn", Decimal("0.29")),
        "speed_nn_floor": speed.get("speed_nn_floor", Decimal(0)),
        "speed_weight_variance_mode": speed.get("speed_weight_var_mode", Decimal(0)),
        "speed_maximum": speed.get("speed_max_val", Decimal(1000)),
        "adm_dlm_weight": adm3.get("adm_dlm_weight", Decimal("0.5")),
        "adm_enhancement_gain_limit": adm3.get(
            "adm_enhn_gain_limit", Decimal(100)
        ),
        "adm_noise_weight": adm3.get("adm_noise_weight", Decimal("0.03125")),
        "adm_normalized_view_distance": adm_nvd,
        "adm_reference_display_height": adm_display_height,
        "adm_minimum": adm3.get("adm_min_val", Decimal(0)),
        "adm_csf_mode": adm3.get("adm_csf_mode", Decimal(0)),
        "adm_horizontal_csf": adm_horizontal_csf,
        "adm_diagonal_csf": adm_diagonal_csf,
        "motion_maximum": motion3.get("motion_max_val", Decimal(10000)),
        "motion_five_frame_window": motion3.get(
            "motion_five_frame_window", False
        ),
        "motion_moving_average": motion3.get("motion_moving_average", False),
        "gamma": gamma,
        "rho": rho,
        "output_slope": slopes[0],
        "output_intercept": intercepts[0],
        "feature_slopes": slopes[1:],
        "feature_intercepts": intercepts[1:],
        "p0": transform["p0"],
        "p1": transform["p1"],
        "p2": transform["p2"],
        "clip_min": clip[0],
        "clip_max": clip[1],
        "chroma_correction": model["chroma_correction_parameter"],
        "support_vectors": support_vectors,
    }


def raw_list(values: list[Decimal]) -> str:
    return ", ".join(str(q32(value)) for value in values)


def generate(models: list[dict[str, object]]) -> str:
    lines = [
        "#pragma once",
        "",
        "#include \"scoring/model.hpp\"",
        "",
        "#include <array>",
        "",
        "namespace crossvmaf::scoring::generated {",
        f'inline constexpr char upstream_commit[] = "{UPSTREAM_COMMIT}";',
        "",
    ]

    for model_index, model in enumerate(models):
        vectors = model["support_vectors"]
        lines.append(
            f"inline constexpr std::array<PackedSupportVector, {len(vectors)}> "
            f"support_vectors_{model_index}{{{{"
        )
        for coefficient, features in vectors:
            lines.append(
                f"    {{{q32(coefficient)}, {{{raw_list(features)}}}}},"
            )
        lines.extend(["}};", ""])

    lines.append(f"inline constexpr std::array<PackedModel, {len(models)}> models{{{{")
    for model_index, model in enumerate(models):
        lines.extend(
            [
                "    {",
                f"        {model['variant']},",
                f'        "{model["name"]}",',
                f'        "{model["sha256"]}",',
                f"        {'true' if model['hfr'] else 'false'},",
                "        {",
                f"            {int(model['cambi_high_res_speedup'])},",
                f"            {q32(model['cambi_visibility_threshold'])},",
                f"            {q32(model['cambi_maximum'])},",
                "        },",
                "        {",
                f"            {q32(model['speed_kernel_scale'])},",
                f"            {q32(model['speed_prescale'])},",
                "            ScalingMethod::"
                f"{model['speed_prescale_method']},",
                f"            {q32(model['speed_sigma_nn'])},",
                f"            {q32(model['speed_nn_floor'])},",
                f"            {int(model['speed_weight_variance_mode'])},",
                f"            {q32(model['speed_maximum'])},",
                "        },",
                "        {",
                f"            {q32(model['adm_dlm_weight'])},",
                f"            {q32(model['adm_enhancement_gain_limit'])},",
                f"            {q32(model['adm_noise_weight'])},",
                f"            {q32(model['adm_normalized_view_distance'])},",
                f"            {int(model['adm_reference_display_height'])},",
                f"            {q32(model['adm_minimum'])},",
                f"            {int(model['adm_csf_mode'])},",
                "            {"
                + ", ".join(str(value) for value in model["adm_horizontal_csf"])
                + "},",
                "            {"
                + ", ".join(str(value) for value in model["adm_diagonal_csf"])
                + "},",
                "        },",
                "        {",
                f"            {q32(model['motion_maximum'])},",
                "            "
                f"{'true' if model['motion_five_frame_window'] else 'false'},",
                "            "
                f"{'true' if model['motion_moving_average'] else 'false'},",
                "        },",
                f"        {q32(model['gamma'])},",
                f"        {q32(model['rho'])},",
                f"        {q32(model['output_slope'])},",
                f"        {q32(model['output_intercept'])},",
                f"        {{{raw_list(model['feature_slopes'])}}},",
                f"        {{{raw_list(model['feature_intercepts'])}}},",
                f"        {q32(model['p0'])},",
                f"        {q32(model['p1'])},",
                f"        {q32(model['p2'])},",
                f"        {q32(model['clip_min'])},",
                f"        {q32(model['clip_max'])},",
                f"        {q32(model['chroma_correction'])},",
                f"        support_vectors_{model_index}.data(),",
                f"        support_vectors_{model_index}.size(),",
                "    },",
            ]
        )
    lines.extend(["}};", "", "}  // namespace crossvmaf::scoring::generated", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("models", nargs=8, type=Path)
    args = parser.parse_args()

    models = [read_model(path) for path in args.models]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = generate(models)
    if not args.output.exists() or args.output.read_text(encoding="utf-8") != output:
        args.output.write_text(output, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
