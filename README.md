# 本地黑白深度视频工具

把普通视频转换成近处偏白、远处偏黑的灰度相对深度视频。全部推理在本机完成，不调用生成式 API，不消耗 Codex 或其他模型 token。

## 网页版用法（推荐）

1. 双击 `start.command`。
2. 默认浏览器会自动打开本地页面。
3. 把视频拖进页面，点击“开始生成深度视频”。
4. 完成后可在页面预览、下载，或点击“在 Finder 中显示”。

页面地址只绑定 `127.0.0.1`，素材不会上传到互联网。网页输出会保存在本工具的 `outputs/` 目录。

## 文件选择器用法

如果不想使用网页，也可以双击 `run.command`，通过系统文件选择器转换。这个入口会把成片保存在原视频目录。

第一次使用会自动安装独立运行环境和 Depth Anything V2 Small 模型。当前机器完成安装后，后续运行可完全离线。

首次安装前请确保系统已有 `uv`、`ffmpeg` 和 `ffprobe`。`uv` 会自动准备 Python 3.11；macOS 可使用 Homebrew 安装 FFmpeg：`brew install ffmpeg`。

## 输出规则

- 保持原画幅，短边为 720 像素。
- 3:4 竖屏素材输出为 720×960。
- H.264 MP4、`yuv420p`、原视频帧率、Web 优化元数据。
- 近处偏白，远处偏黑。
- 不包含音频。
- 视频超过 15 秒时自动均衡拆分，每段不超过 15 秒。
- 例如 17 秒会输出 8 秒和 9 秒两段；超过 30 秒会至少输出三段。
- 如果同名文件已存在，会自动追加数字，不覆盖旧文件。

## 命令行用法

```bash
./run.command "/完整路径/输入视频.mp4"
```

也可以直接调用核心程序：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ./.venv/bin/python ./depth_video.py "/完整路径/输入视频.mp4"
```

## 文件说明

- `start.command`：启动本地网页并自动打开浏览器。
- `web_app.py`：本地网页服务。
- `templates/index.html`：网页界面。
- `run.command`：日常双击入口。
- `setup.command`：重新安装本地环境和模型。
- `depth_video.py`：转换核心。
- `.venv/`：独立 Python 环境。
- `models/`：本地深度模型。
- `outputs/`：网页版生成的成片。
