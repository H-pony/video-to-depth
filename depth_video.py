#!/usr/bin/env python3
"""Offline grayscale relative-depth video converter for macOS."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


TOOL_DIR = Path(__file__).resolve().parent
MODEL_DIR = TOOL_DIR / "models" / "depth-anything-v2-small"
MODEL_NAME = "Depth Anything V2 Small"
SHORT_EDGE = 720
MAX_RANGE_SAMPLES = 48
DEPTH_SAMPLE_STRIDE = 12
MAX_SEGMENT_SECONDS = 15.0


def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"找不到 {name}，请先运行 setup.command")


def even(value: float) -> int:
    return max(2, int(round(value / 2.0) * 2))


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    fps_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1"
    if fps_text == "0/0":
        fps_text = "30/1"
    fps = float(Fraction(fps_text))
    duration = float(stream.get("duration") or data["format"].get("duration") or 0)
    frame_count = int(stream.get("nb_frames") or round(duration * fps))

    source_width = int(stream["width"])
    source_height = int(stream["height"])
    rotation = int(stream.get("tags", {}).get("rotate", 0))
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = int(side_data["rotation"])
            break
    if abs(rotation) % 180 == 90:
        source_width, source_height = source_height, source_width

    if source_width <= source_height:
        output_width = SHORT_EDGE
        output_height = even(SHORT_EDGE * source_height / source_width)
    else:
        output_height = SHORT_EDGE
        output_width = even(SHORT_EDGE * source_width / source_height)

    return {
        "fps_text": fps_text,
        "fps": fps,
        "duration": duration,
        "frame_count": max(frame_count, 1),
        "source_width": source_width,
        "source_height": source_height,
        "output_width": output_width,
        "output_height": output_height,
    }


def output_family_exists(candidate: Path) -> bool:
    if candidate.exists():
        return True
    segment_prefix = f"{candidate.stem}_第"
    return any(
        path.is_file()
        and path.suffix.lower() == candidate.suffix.lower()
        and path.stem.startswith(segment_prefix)
        for path in candidate.parent.iterdir()
    )


def unique_output_path(input_path: Path) -> Path:
    candidate = input_path.with_name(f"{input_path.stem}_黑白深度_720p.mp4")
    if not output_family_exists(candidate):
        return candidate
    counter = 2
    while True:
        candidate = input_path.with_name(
            f"{input_path.stem}_黑白深度_720p_{counter}.mp4"
        )
        if not output_family_exists(candidate):
            return candidate
        counter += 1


def build_segment_plan(duration: float) -> tuple[list[float], list[float]]:
    """Return split boundaries and balanced durations, each at most 15 seconds."""
    if duration <= MAX_SEGMENT_SECONDS + 1e-6:
        return [], [duration]

    segment_count = math.ceil(duration / MAX_SEGMENT_SECONDS)
    boundaries = [
        float(round(duration * index / segment_count))
        for index in range(1, segment_count)
    ]
    durations = [
        end - start
        for start, end in zip(
            [0.0, *boundaries],
            [*boundaries, duration],
        )
    ]
    if any(
        segment <= 0 or segment > MAX_SEGMENT_SECONDS + 1e-6
        for segment in durations
    ):
        boundaries = [
            duration * index / segment_count
            for index in range(1, segment_count)
        ]
        durations = [
            end - start
            for start, end in zip(
                [0.0, *boundaries],
                [*boundaries, duration],
            )
        ]
    return boundaries, durations


def segment_output_paths(output_path: Path, segment_count: int) -> list[Path]:
    if segment_count == 1:
        return [output_path]
    return [
        output_path.with_name(f"{output_path.stem}_第{index:02d}段.mp4")
        for index in range(1, segment_count + 1)
    ]


def display_duration(seconds: float) -> str:
    if abs(seconds - round(seconds)) < 0.005:
        return f"{round(seconds):.0f}"
    return f"{seconds:.2f}".rstrip("0").rstrip(".")


def read_exact(stream, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decoded_batches(
    input_path: Path,
    width: int,
    height: int,
    batch_size: int,
    select_interval: int | None = None,
):
    filters: list[str] = []
    if select_interval is not None and select_interval > 1:
        filters.append(f"select=not(mod(n\\,{select_interval}))")
    filters.append(f"scale={width}:{height}:flags=lanczos")
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-an",
        "-vf",
        ",".join(filters),
    ]
    if select_interval is not None:
        command.extend(["-vsync", "vfr"])
    command.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])

    decoder = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert decoder.stdout is not None
    assert decoder.stderr is not None
    frame_bytes = width * height * 3
    try:
        while True:
            images: list[Image.Image] = []
            for _ in range(batch_size):
                raw = read_exact(decoder.stdout, frame_bytes)
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise RuntimeError(
                        f"视频解码得到不完整帧：{len(raw)}/{frame_bytes} 字节"
                    )
                images.append(Image.frombytes("RGB", (width, height), raw))
            if not images:
                break
            yield images
    finally:
        decoder.stdout.close()
        stderr = decoder.stderr.read().decode("utf-8", errors="replace")
        return_code = decoder.wait()
        if return_code != 0 and sys.exc_info()[0] is None:
            raise RuntimeError(f"ffmpeg 解码失败：\n{stderr}")


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model():
    if not (MODEL_DIR / "config.json").exists():
        raise RuntimeError("本地深度模型不存在，请先运行 setup.command")
    device = select_device()
    print(f"加载 {MODEL_NAME}（{device}）……", flush=True)
    processor = AutoImageProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_fast=False,
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model.to(device).eval()
    return processor, model, device


def infer_batch(processor, model, device, images, height: int, width: int):
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        results = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(height, width)] * len(images),
        )
    return [
        result["predicted_depth"].detach().float().cpu().numpy()
        for result in results
    ]


def estimate_stable_range(
    input_path: Path,
    info: dict,
    processor,
    model,
    device,
    batch_size: int,
) -> tuple[float, float]:
    interval = max(1, math.ceil(info["frame_count"] / MAX_RANGE_SAMPLES))
    estimated_samples = math.ceil(info["frame_count"] / interval)
    samples: list[np.ndarray] = []
    processed = 0
    print(
        f"分析全片深度范围（约 {estimated_samples} 个采样帧）……",
        flush=True,
    )
    for images in decoded_batches(
        input_path,
        info["output_width"],
        info["output_height"],
        batch_size,
        select_interval=interval,
    ):
        depths = infer_batch(
            processor,
            model,
            device,
            images,
            info["output_height"],
            info["output_width"],
        )
        for depth in depths:
            samples.append(
                np.asarray(
                    depth[::DEPTH_SAMPLE_STRIDE, ::DEPTH_SAMPLE_STRIDE],
                    dtype=np.float32,
                ).reshape(-1)
            )
            processed += 1
        print(f"范围分析 {processed}/{estimated_samples}", flush=True)

    if not samples:
        raise RuntimeError("没有读到可分析的视频帧")
    merged = np.concatenate(samples)
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        raise RuntimeError("深度模型没有产生有效数值")
    low, high = np.percentile(merged, [1.0, 99.0]).tolist()
    if not high > low:
        raise RuntimeError(f"深度范围无效：{low}–{high}")
    print(f"固定灰度范围：{low:.4f}–{high:.4f}", flush=True)
    return low, high


def encode_video(
    input_path: Path,
    output_path: Path,
    info: dict,
    processor,
    model,
    device,
    low: float,
    high: float,
    batch_size: int,
    segment_boundaries: list[float],
) -> tuple[int, list[Path]]:
    final_paths = segment_output_paths(output_path, len(segment_boundaries) + 1)
    if segment_boundaries:
        partial_pattern = output_path.with_name(
            f".{output_path.stem}.partial_第%02d段.mp4"
        )
        partial_paths = [
            output_path.with_name(
                f".{output_path.stem}.partial_第{index:02d}段.mp4"
            )
            for index in range(1, len(final_paths) + 1)
        ]
    else:
        partial_pattern = output_path.with_name(
            f".{output_path.stem}.partial{output_path.suffix}"
        )
        partial_paths = [partial_pattern]

    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-video_size",
        f"{info['output_width']}x{info['output_height']}",
        "-framerate",
        info["fps_text"],
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-vsync",
        "cfr",
        "-metadata",
        f"comment=Relative depth map; near=white; far=black; model={MODEL_NAME}",
    ]
    if segment_boundaries:
        split_times = ",".join(
            f"{boundary:.6f}" for boundary in segment_boundaries
        )
        command.extend(
            [
                "-force_key_frames",
                split_times,
                "-f",
                "segment",
                "-segment_times",
                split_times,
                "-segment_start_number",
                "1",
                "-reset_timestamps",
                "1",
                "-segment_format",
                "mp4",
                "-segment_format_options",
                "movflags=+faststart",
                str(partial_pattern),
            ]
        )
    else:
        command.extend(["-movflags", "+faststart", str(partial_pattern)])

    encoder = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    assert encoder.stderr is not None
    processed = 0
    started = time.monotonic()
    try:
        for images in decoded_batches(
            input_path,
            info["output_width"],
            info["output_height"],
            batch_size,
        ):
            depths = infer_batch(
                processor,
                model,
                device,
                images,
                info["output_height"],
                info["output_width"],
            )
            for depth in depths:
                normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
                normalized = np.power(normalized, 0.9)
                gray = np.rint(normalized * 255.0).astype(np.uint8)
                encoder.stdin.write(gray.tobytes())
                processed += 1
            elapsed = max(time.monotonic() - started, 0.001)
            rate = processed / elapsed
            eta = max(info["frame_count"] - processed, 0) / rate
            print(
                f"生成 {processed}/{info['frame_count']} 帧 "
                f"（{rate:.1f} fps，约剩 {eta:.0f} 秒）",
                flush=True,
            )
        encoder.stdin.close()
        stderr = encoder.stderr.read().decode("utf-8", errors="replace")
        return_code = encoder.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg 编码失败：\n{stderr}")
        missing = [path for path in partial_paths if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"分段输出数量不正确，缺少：{', '.join(map(str, missing))}"
            )
        for partial_path, final_path in zip(partial_paths, final_paths):
            os.replace(partial_path, final_path)
    except Exception:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        for partial_path in partial_paths:
            if partial_path.exists():
                partial_path.unlink()
        raise
    return processed, final_paths


def verify_output(path: Path, expected: dict, enforce_segment_limit: bool) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    streams = data["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    if int(video["width"]) != expected["output_width"]:
        raise RuntimeError("输出宽度验收失败")
    if int(video["height"]) != expected["output_height"]:
        raise RuntimeError("输出高度验收失败")
    if video["codec_name"] != "h264" or video["pix_fmt"] != "yuv420p":
        raise RuntimeError("输出编码验收失败")
    if any(s["codec_type"] == "audio" for s in streams):
        raise RuntimeError("输出意外包含音频")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        check=True,
    )
    metadata = {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": float(Fraction(video["avg_frame_rate"])),
        "frames": int(video.get("nb_frames") or 0),
        "duration": float(video.get("duration") or data["format"]["duration"]),
    }
    if (
        enforce_segment_limit
        and metadata["duration"] > MAX_SEGMENT_SECONDS + 0.05
    ):
        raise RuntimeError(
            f"分段时长验收失败：{metadata['duration']:.3f} 秒超过 15 秒"
        )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="离线生成近白远黑的 720p 灰度深度视频"
    )
    parser.add_argument("input", type=Path, help="输入视频")
    parser.add_argument("output", type=Path, nargs="?", help="输出 MP4（可省略）")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    require_program("ffmpeg")
    require_program("ffprobe")
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入视频不存在：{input_path}")
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else unique_output_path(input_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = probe_video(input_path)
    print(
        f"输入：{info['source_width']}×{info['source_height']}，"
        f"{info['fps']:.3g} fps，约 {info['duration']:.2f} 秒",
        flush=True,
    )
    print(
        f"输出：{info['output_width']}×{info['output_height']}，"
        "H.264 MP4，无音频",
        flush=True,
    )
    segment_boundaries, segment_durations = build_segment_plan(info["duration"])
    if segment_boundaries:
        duration_text = " + ".join(
            f"{display_duration(duration)} 秒"
            for duration in segment_durations
        )
        print(
            f"自动拆分：{display_duration(info['duration'])} 秒 → "
            f"{len(segment_durations)} 段（{duration_text}）",
            flush=True,
        )

    processor, model, device = load_model()
    low, high = estimate_stable_range(
        input_path,
        info,
        processor,
        model,
        device,
        max(1, args.batch_size),
    )
    frame_count, output_paths = encode_video(
        input_path,
        output_path,
        info,
        processor,
        model,
        device,
        low,
        high,
        max(1, args.batch_size),
        segment_boundaries,
    )
    verified_outputs = [
        verify_output(
            path,
            info,
            enforce_segment_limit=bool(segment_boundaries),
        )
        for path in output_paths
    ]
    print("\n转换完成并通过验收：", flush=True)
    for index, (path, verified) in enumerate(
        zip(output_paths, verified_outputs),
        start=1,
    ):
        label = f"第 {index} 段" if len(output_paths) > 1 else "成片"
        print(
            f"{label}：{path}\n"
            f"  {verified['width']}×{verified['height']} / "
            f"{verified['fps']:.3g} fps / {verified['frames']} 帧 / "
            f"{verified['duration']:.2f} 秒",
            flush=True,
        )
    print(
        "OUTPUT_JSON="
        + json.dumps([str(path) for path in output_paths], ensure_ascii=False),
        flush=True,
    )
    print(f"总处理帧数：{frame_count}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"\n错误：{error}", file=sys.stderr)
        raise SystemExit(1)
