# 工具箱缺失检测系统说明文档

## 1. 项目简介

本项目用于工具箱缺失检测，支持海康威视工业相机拍照、本地图片导入、YOLO 目标检测、die 编号 OCR 识别以及缺失报警。

系统主要完成以下任务：

- 检测工具箱中 `die`、`tool`、`pin` 等目标是否存在。
- 对 `die` 类目标识别编号，例如 `TH12`、`TH16`、`TH20`、`TH25`、`TH26`。
- 根据配置中的 expected rules 判断数量缺失和编号缺失。
- 在网页中显示输入图像、检测结果图、检测耗时和缺失明细。

## 2. 主要文件说明

| 文件/目录 | 作用 |
| --- | --- |
| `backend/app/main.py` | Web 后端入口，提供拍照检测、上传检测、实时预览、状态查询等接口 |
| `det.py` | 当前后端使用的检测与 OCR 逻辑，替代原来的 `wenzi.py` OCR 主流程 |
| `best_last.pt` | YOLO 检测模型权重 |
| `test_hk_opecv.py` | 海康威视相机单次拍照脚本 |
| `hik_live_worker.py` | 海康威视实时预览独立 worker，用于持续获取海康相机帧 |
| `config/toolbox_workflow.json` | 系统配置文件，包括相机、模型、OCR、规则等 |
| `toolbox-frontend/` | Vue 前端工程 |
| `runtime/` | 运行时输出目录，保存拍照图、检测结果图、OCR crop、日志等 |

## 3. 当前相机配置

配置文件位置：

```text
config/toolbox_workflow.json
```

当前要求只使用海康威视相机：

```json
{
  "camera": {
    "hik_only_mode": true,
    "provider": "hik_sdk",
    "live_provider": "hik_sdk",
    "camera_index": 0
  }
}
```

说明：

- `provider: hik_sdk` 表示拍照检测使用海康 SDK。
- `live_provider: hik_sdk` 表示实时预览也使用海康 SDK。
- `hik_only_mode: true` 表示禁止回退到电脑自带摄像头。

## 4. 启动方式

推荐直接启动 Web 服务：

```powershell
.\.venv\Scripts\python.exe backend\app\main.py --host 127.0.0.1 --port 8000
```

启动后浏览器访问：

```text
http://127.0.0.1:8000
```

常用状态接口：

```text
http://127.0.0.1:8000/api/health
http://127.0.0.1:8000/api/camera-mode
http://127.0.0.1:8000/api/live-status
```

## 5. 检测流程

### 5.1 拍照并检测

点击页面中的“拍照并检测”后，后端执行以下流程：

1. 通过海康 SDK 获取图像。
2. 使用 `best_last.pt` 对图像进行 YOLO 检测。
3. 对检测到的 `die` 目标裁剪局部区域。
4. 调用 `det.py` 中的 OCR 逻辑识别编号。
5. 根据规则判断缺失数量和缺失编号。
6. 返回输入图像、推理结果图、缺失明细和耗时。

### 5.2 本地图片检测

上传本地图片后，流程与拍照检测类似，但输入图像来自上传文件，不调用相机。

## 6. die 编号 OCR 逻辑

当前 OCR 使用 `det.py` 中的 `RapidDigitOCR`。

针对右上角 `TH25`、`TH26` 识别不稳定的问题，已做优化：

- OCR 输入不再缩小到 480 长边，改为放大到约 900 长边，保留小字细节。
- 优先识别 die 下半环、右上环、整环区域。
- 不再因为误读到单个数字就提前停止，例如把 `TH26` 误读成 `9ZH` 后直接放弃。
- 对旋转方向做多角度尝试，提高斜字识别率。

## 7. 实时预览说明

前端左侧实时图像走：

```text
/api/live-stream
```

后端会启动：

```text
hik_live_worker.py
```

该 worker 只连接海康威视相机，并持续写出最新帧：

```text
runtime/preview/hik_live.jpg
```

后端再把该帧作为 MJPEG 流推给前端。

注意：系统已经禁止调用电脑自带摄像头。如果海康 SDK 不能正常枚举设备，页面不会偷偷切换到电脑摄像头。

## 8. 海康实时不刷新排查

如果页面左侧看起来不是实时画面，先访问：

```text
http://127.0.0.1:8000/api/live-status
```

重点看以下字段：

| 字段 | 含义 |
| --- | --- |
| `live_provider` | 应为 `hik_sdk` |
| `hik_live_worker_alive` | 海康实时 worker 是否存活 |
| `hik_live_file_age_ms` | 最新实时帧文件距现在的时间 |
| `cached_frame` | 后端是否拿到可用实时帧 |
| `last_error` | 后端记录的实时流错误 |

如果日志停在：

```text
[LIVE] initializing Hik SDK
[LIVE] enumerating Hik devices
```

说明海康 SDK 卡在设备枚举阶段，程序还没有拿到相机帧。

可按以下顺序排查：

1. 关闭 MVS Viewer、海康官方 Demo 或其他占用相机的软件。
2. 重新插拔海康相机 USB/网线，或重启相机电源。
3. 如果是 GigE 相机，确认电脑网卡 IP 与相机 IP 在同一网段。
4. 用海康 MVS Viewer 确认能看到相机并正常取流。
5. 单独运行拍照脚本测试：

```powershell
.\.venv\Scripts\python.exe test_hk_opecv.py --mode ultrafast --count 1 --auto-exposure --ae-profile day --ae-settle-frames 0 --auto-gain --headless
```

如果该命令也长时间没有输出 `Found devices`，说明不是前端问题，而是海康 SDK 或相机连接层面的问题。

## 9. 运行输出位置

| 路径 | 内容 |
| --- | --- |
| `runtime/captures/` | 拍照得到的原始输入图 |
| `runtime/uploads/` | 上传图片 |
| `runtime/annotated/` | 带检测框的结果图 |
| `runtime/ocr_crops/` | die OCR 裁剪图 |
| `runtime/preview/` | 实时预览帧 |
| `runtime/logs/` | 海康实时 worker 日志 |

## 10. 常见问题

### 10.1 为什么检测准确但左侧不实时？

检测准确说明拍照图和 YOLO/OCR 流程基本正常。左侧不实时通常是海康实时取流没有持续产出帧，需要检查 `/api/live-status` 和 `runtime/logs/hik-live.*.log`。

### 10.2 为什么不能用电脑摄像头？

本项目要求检测工业相机视角，因此配置中启用了：

```json
"hik_only_mode": true
```

后端不会回退到电脑自带摄像头。

### 10.3 TH25、TH26 识别不出来怎么办？

优先检查 OCR crop：

```text
runtime/ocr_crops/
```

如果 crop 中编号太小、反光严重或被裁到边缘，可以调整相机角度、曝光或检测框 padding。当前 `det.py` 已针对 `TH25`、`TH26` 做了多区域、多角度 OCR 增强。

## 11. Git 提交记录参考

近期关键修复：

```text
a443002 fix: improve top die OCR variants
a024611 fix: force Hik camera for live preview
ae44680 fix: add Hik live preview worker
```

