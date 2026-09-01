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

if [[ "$#" -gt 0 ]]; then
  input_path="$1"
else
  input_path="$(/usr/bin/osascript <<'APPLESCRIPT'
set chosenFile to choose file with prompt "选择要转换为黑白深度视频的视频"
POSIX path of chosenFile
APPLESCRIPT
)" || exit 0
fi

echo
echo "开始转换：$input_path"
echo

PYTORCH_ENABLE_MPS_FALLBACK=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
"$tool_dir/.venv/bin/python" "$tool_dir/depth_video.py" "$input_path"
status=$?

if [[ "$status" -eq 0 ]]; then
  /usr/bin/osascript -e 'display dialog "黑白深度视频已经输出到原视频所在文件夹。" with title "转换完成" buttons {"好"} default button "好"' >/dev/null
else
  /usr/bin/osascript -e 'display dialog "转换失败，请查看终端中的错误信息。" with title "视频转深度" buttons {"好"} default button "好" with icon stop' >/dev/null
fi

echo
echo "按任意键关闭。"
read -k 1
exit "$status"
