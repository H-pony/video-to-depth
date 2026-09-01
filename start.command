#!/bin/zsh
set -u

tool_dir="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -x "$tool_dir/.venv/bin/python" || ! -f "$tool_dir/models/depth-anything-v2-small/config.json" ]]; then
  echo "首次运行需要完成本地安装……"
  "$tool_dir/setup.command" || {
    echo
    echo "安装失败。按任意键关闭。"
    read -k 1
    exit 1
  }
fi

PYTORCH_ENABLE_MPS_FALLBACK=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
"$tool_dir/.venv/bin/python" "$tool_dir/web_app.py"
status=$?

if [[ "$status" -ne 0 && "$status" -ne 130 ]]; then
  echo
  echo "本地网页启动失败。按任意键关闭。"
  read -k 1
fi
exit "$status"
