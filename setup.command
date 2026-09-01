#!/bin/zsh
set -euo pipefail

tool_dir="$(cd "$(dirname "$0")" && pwd)"
uv_path="${UV_PATH:-$(command -v uv 2>/dev/null || true)}"
if [[ -z "$uv_path" && -x "$HOME/.local/bin/uv" ]]; then
  uv_path="$HOME/.local/bin/uv"
fi
python_request="${PYTHON_PATH:-3.11}"

if [[ -z "$uv_path" || ! -x "$uv_path" ]]; then
  echo "找不到 uv。请先安装：https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo "找不到 ffmpeg/ffprobe，请先运行：brew install ffmpeg"
  exit 1
fi

echo "1/3 检查独立 Python 环境……"
if [[ ! -x "$tool_dir/.venv/bin/python" ]]; then
  "$uv_path" venv "$tool_dir/.venv" --python "$python_request"
else
  echo "继续使用现有环境：$tool_dir/.venv"
fi

echo "2/3 安装本地推理依赖……"
"$uv_path" pip install \
  --python "$tool_dir/.venv/bin/python" \
  --requirement "$tool_dir/requirements.txt"

echo "3/3 固定深度模型到工具目录……"
if [[ -f "$tool_dir/models/depth-anything-v2-small/config.json" ]]; then
  echo "继续使用现有本地模型。"
else
  "$tool_dir/.venv/bin/python" - "$tool_dir/models/depth-anything-v2-small" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

target = Path(sys.argv[1])
target.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id="depth-anything/Depth-Anything-V2-Small-hf",
    local_dir=target,
)
if not (target / "config.json").exists():
    raise RuntimeError("模型安装后缺少 config.json")
print(f"模型已安装：{target}")
PY
fi

echo
echo "安装完成。以后双击 start.command 即可在网页中离线转换。"
