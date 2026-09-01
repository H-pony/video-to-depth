#!/usr/bin/env python3
"""Local-only web interface for the offline depth video converter."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import threading
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file


TOOL_DIR = Path(__file__).resolve().parent
PYTHON_PATH = TOOL_DIR / ".venv" / "bin" / "python"
CONVERTER_PATH = TOOL_DIR / "depth_video.py"
MODEL_CONFIG = TOOL_DIR / "models" / "depth-anything-v2-small" / "config.json"
RUNTIME_DIR = TOOL_DIR / "runtime"
OUTPUT_DIR = TOOL_DIR / "outputs"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024

app = Flask(__name__, template_folder=str(TOOL_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
conversion_lock = threading.Lock()


def update_job(job_id: str, **changes) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is not None:
            job.update(changes)


def append_log(job_id: str, line: str) -> None:
    clean_line = line.strip()
    if not clean_line:
        return
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return
        log = job.setdefault("log", [])
        log.append(clean_line)
        del log[:-60]


def public_job(job_id: str) -> dict | None:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return None
        internal_outputs = list(job.get("outputs") or [])
        payload = {
            key: value
            for key, value in job.items()
            if key not in {"input_path", "output_base", "outputs"}
        }
    if payload["state"] == "complete":
        payload["outputs"] = [
            {
                "index": index,
                "name": item["name"],
                "metadata": item["metadata"],
                "download_url": f"/api/jobs/{job_id}/files/{index}/download",
                "media_url": f"/api/jobs/{job_id}/files/{index}/media",
                "reveal_url": f"/api/jobs/{job_id}/files/{index}/reveal",
            }
            for index, item in enumerate(internal_outputs)
        ]
    return payload


def clean_filename(filename: str) -> tuple[str, str]:
    leaf = Path(filename or "video.mp4").name.replace("\x00", "")
    suffix = Path(leaf).suffix.lower()
    if not suffix or len(suffix) > 10:
        suffix = ".mp4"
    stem = Path(leaf).stem[:120]
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")
    return stem or "video", suffix


def unique_output_path(stem: str) -> Path:
    candidate = OUTPUT_DIR / f"{stem}_黑白深度_720p.mp4"
    if not output_family_exists(candidate):
        return candidate
    counter = 2
    while True:
        candidate = OUTPUT_DIR / f"{stem}_黑白深度_720p_{counter}.mp4"
        if not output_family_exists(candidate):
            return candidate
        counter += 1


def output_family_exists(candidate: Path) -> bool:
    if candidate.exists():
        return True
    segment_prefix = f"{candidate.stem}_第"
    return any(
        path.is_file()
        and path.suffix.lower() == ".mp4"
        and path.stem.startswith(segment_prefix)
        for path in candidate.parent.iterdir()
    )


def parse_progress(job_id: str, line: str) -> None:
    range_match = re.search(r"范围分析\s+(\d+)/(\d+)", line)
    if range_match:
        current, total = map(int, range_match.groups())
        progress = 5 + 20 * current / max(total, 1)
        update_job(
            job_id,
            progress=round(progress, 1),
            stage="正在分析全片深度范围",
        )
        return

    frame_match = re.search(r"生成\s+(\d+)/(\d+)\s+帧", line)
    if frame_match:
        current, total = map(int, frame_match.groups())
        progress = 25 + 70 * current / max(total, 1)
        update_job(
            job_id,
            progress=round(progress, 1),
            stage=f"正在生成深度视频 · {current}/{total} 帧",
        )
        return

    if line.startswith("加载 "):
        update_job(job_id, progress=3, stage="正在加载本地深度模型")
    elif line.startswith("自动拆分："):
        update_job(job_id, stage=line)
    elif "分析全片深度范围" in line:
        update_job(job_id, progress=5, stage="正在分析全片深度范围")
    elif "转换完成并通过验收" in line:
        update_job(job_id, progress=98, stage="正在完成输出验收")


def probe_output(path: Path) -> dict:
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
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": video["avg_frame_rate"],
        "frames": int(video.get("nb_frames") or 0),
        "duration": round(
            float(video.get("duration") or data["format"]["duration"]),
            2,
        ),
        "size": path.stat().st_size,
    }


def remove_uploaded_file(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink()
        parent = path.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass


def run_conversion(job_id: str) -> None:
    with conversion_lock:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                return
            input_path = Path(job["input_path"])
            stem = job["safe_stem"]

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = unique_output_path(stem)
        update_job(
            job_id,
            state="running",
            progress=1,
            stage="正在读取视频",
            output_base=str(output_path),
        )

        environment = os.environ.copy()
        environment.update(
            {
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        command = [
            str(PYTHON_PATH),
            str(CONVERTER_PATH),
            str(input_path),
            str(output_path),
        ]
        process = subprocess.Popen(
            command,
            cwd=str(TOOL_DIR),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        reported_outputs: list[Path] = []
        for line in process.stdout:
            stripped = line.strip()
            if stripped.startswith("OUTPUT_JSON="):
                try:
                    values = json.loads(stripped.removeprefix("OUTPUT_JSON="))
                    reported_outputs = [Path(value).resolve() for value in values]
                except (json.JSONDecodeError, TypeError):
                    reported_outputs = []
                continue
            append_log(job_id, line)
            parse_progress(job_id, line)
        return_code = process.wait()

        if not reported_outputs and output_path.is_file():
            reported_outputs = [output_path.resolve()]
        safe_output_root = OUTPUT_DIR.resolve()
        reported_outputs = [
            path
            for path in reported_outputs
            if path.parent == safe_output_root and path.is_file()
        ]

        if return_code == 0 and reported_outputs:
            outputs = [
                {
                    "path": str(path),
                    "name": path.name,
                    "metadata": probe_output(path),
                }
                for path in reported_outputs
            ]
            update_job(
                job_id,
                state="complete",
                progress=100,
                stage=f"转换完成 · {len(outputs)} 段"
                if len(outputs) > 1
                else "转换完成",
                outputs=outputs,
            )
        else:
            with jobs_lock:
                log = jobs.get(job_id, {}).get("log", [])
            error_message = log[-1] if log else "本地转换进程异常结束"
            update_job(
                job_id,
                state="error",
                stage="转换失败",
                error=error_message,
            )
        remove_uploaded_file(input_path)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": PYTHON_PATH.is_file() and MODEL_CONFIG.is_file(),
            "local": True,
        }
    )


@app.post("/api/convert")
def create_conversion():
    if "video" not in request.files:
        return jsonify({"error": "请选择视频文件"}), 400
    video = request.files["video"]
    if not video.filename:
        return jsonify({"error": "文件名为空"}), 400

    safe_stem, suffix = clean_filename(video.filename)
    job_id = uuid.uuid4().hex
    upload_dir = RUNTIME_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=False)
    input_path = upload_dir / f"input{suffix}"
    try:
        video.save(input_path)
    except Exception:
        remove_uploaded_file(input_path)
        raise

    with jobs_lock:
        ahead = sum(
            1
            for job in jobs.values()
            if job["state"] in {"queued", "running"}
        )
        jobs[job_id] = {
            "id": job_id,
            "state": "queued",
            "progress": 0,
            "stage": "等待处理" if ahead else "准备开始",
            "original_name": Path(video.filename).name,
            "safe_stem": safe_stem,
            "input_path": str(input_path),
            "output_base": None,
            "outputs": [],
            "error": None,
            "log": [],
        }

    worker = threading.Thread(
        target=run_conversion,
        args=(job_id,),
        daemon=True,
        name=f"depth-job-{job_id[:8]}",
    )
    worker.start()
    return jsonify({"job_id": job_id}), 202


@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    payload = public_job(job_id)
    if payload is None:
        abort(404)
    return jsonify(payload)


def completed_output(job_id: str, file_index: int) -> tuple[Path, dict]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None or job["state"] != "complete":
            abort(404)
        outputs = job.get("outputs") or []
        if file_index < 0 or file_index >= len(outputs):
            abort(404)
        item = outputs[file_index]
        path = Path(item["path"])
        output_name = item["name"]
    if not path.is_file():
        abort(404)
    return path, {"output_name": output_name}


@app.get("/api/jobs/<job_id>/files/<int:file_index>/media")
def job_media(job_id: str, file_index: int):
    path, _ = completed_output(job_id, file_index)
    return send_file(path, mimetype="video/mp4", conditional=True)


@app.get("/api/jobs/<job_id>/files/<int:file_index>/download")
def job_download(job_id: str, file_index: int):
    path, info = completed_output(job_id, file_index)
    return send_file(
        path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name=info["output_name"],
        conditional=True,
    )


@app.post("/api/jobs/<job_id>/files/<int:file_index>/reveal")
def job_reveal(job_id: str, file_index: int):
    path, _ = completed_output(job_id, file_index)
    subprocess.Popen(
        ["open", "-R", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return jsonify({"ok": True})


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"error": "视频超过 20 GB，无法上传到本地处理区"}), 413


def available_port(start: int = 7860, stop: int = 7870) -> int:
    for port in range(start, stop + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("7860–7870 端口均被占用")


def main() -> int:
    if not PYTHON_PATH.is_file() or not MODEL_CONFIG.is_file():
        print("本地环境未安装，请先运行 setup.command。")
        return 1
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    port = available_port()
    url = f"http://127.0.0.1:{port}"
    print(f"\n本地深度视频工具已启动：{url}")
    print("关闭这个终端窗口即可停止服务。\n")
    if os.environ.get("DEPTH_WEB_NO_BROWSER") != "1":
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    app.run(
        host="127.0.0.1",
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
