import argparse
import time
from typing import List, Optional, Tuple

import cv2

from paddleocr import PaddleOCR


def build_rtsp_url(
    ip: str,
    username: str,
    password: str,
    port: int,
    channel_id: int,
) -> str:
    return f"rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/{channel_id}"


def parse_roi(roi_text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not roi_text:
        return None
    parts = roi_text.split(",")
    if len(parts) != 4:
        raise ValueError("--roi 格式错误，应为 x,y,w,h")
    x, y, w, h = [int(item.strip()) for item in parts]
    if w <= 0 or h <= 0:
        raise ValueError("--roi 的 w,h 必须大于0")
    return x, y, w, h


def parse_legacy_ocr_with_score(legacy_result, score_thresh: float) -> List[str]:
    texts: List[str] = []
    if not legacy_result:
        return texts

    lines = legacy_result[0] if isinstance(legacy_result, list) and legacy_result else []
    for line in lines:
        if (
            isinstance(line, (list, tuple))
            and len(line) > 1
            and isinstance(line[1], (list, tuple))
            and len(line[1]) > 1
        ):
            text = str(line[1][0]).strip()
            score = float(line[1][1])
            if text and score >= score_thresh:
                texts.append(text)
    return texts


def extract_texts(ocr_engine: PaddleOCR, image, score_thresh: float) -> List[str]:
    texts: List[str] = []

    if hasattr(ocr_engine, "predict"):
        result = list(ocr_engine.predict(image))
        if not result:
            return texts
        first = result[0]
        if isinstance(first, dict):
            rec_texts = first.get("rec_texts") or []
            texts.extend([str(item).strip() for item in rec_texts if str(item).strip()])
            return texts

    legacy_result = ocr_engine.ocr(image, cls=True)
    texts.extend(parse_legacy_ocr_with_score(legacy_result, score_thresh))

    return texts


def open_capture(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(rtsp_url)
    return cap


def main():
    parser = argparse.ArgumentParser(description="海康威视 RTSP 实时文字识别")
    parser.add_argument("--rtsp-url", type=str, default="", help="完整RTSP地址，填了则优先使用")
    parser.add_argument("--ip", type=str, default="192.168.1.64", help="摄像头IP")
    parser.add_argument("--port", type=int, default=554, help="RTSP端口")
    parser.add_argument("--username", type=str, default="admin", help="摄像头用户名")
    parser.add_argument("--password", type=str, default="12345", help="摄像头密码")
    parser.add_argument(
        "--channel-id",
        type=int,
        default=101,
        help="海康通道ID，主码流常用101，子码流常用102",
    )
    parser.add_argument("--lang", type=str, default="ch", help="OCR语言，如 ch/en")
    parser.add_argument("--ocr-interval", type=int, default=8, help="每隔多少帧做一次OCR")
    parser.add_argument("--score-thresh", type=float, default=0.5, help="文本置信度阈值(旧版OCR结果)")
    parser.add_argument("--roi", type=str, default="", help="识别区域 x,y,w,h，留空表示全图")
    parser.add_argument("--reconnect-wait", type=float, default=2.0, help="断流重连等待秒数")
    args = parser.parse_args()

    rtsp_url = args.rtsp_url.strip() or build_rtsp_url(
        ip=args.ip,
        username=args.username,
        password=args.password,
        port=args.port,
        channel_id=args.channel_id,
    )

    roi: Optional[Tuple[int, int, int, int]] = parse_roi(args.roi) if args.roi else None

    print(f"[INFO] 正在连接: {rtsp_url}")
    print("[INFO] 按 q 退出")

    ocr = PaddleOCR(use_textline_orientation=True, lang=args.lang, enable_mkldnn=False)

    cap = open_capture(rtsp_url)
    frame_count = 0
    latest_text = ""
    latest_time = 0.0

    while True:
        if not cap.isOpened():
            print("[WARN] 视频流未打开，尝试重连...")
            time.sleep(args.reconnect_wait)
            cap.release()
            cap = open_capture(rtsp_url)
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] 读取帧失败，尝试重连...")
            time.sleep(args.reconnect_wait)
            cap.release()
            cap = open_capture(rtsp_url)
            continue

        h, w = frame.shape[:2]
        view = frame.copy()
        target = frame

        if roi is not None:
            x, y, rw, rh = roi
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            rw = min(rw, w - x)
            rh = min(rh, h - y)
            target = frame[y : y + rh, x : x + rw]
            cv2.rectangle(view, (x, y), (x + rw, y + rh), (0, 255, 255), 2)

        frame_count += 1
        if frame_count % max(1, args.ocr_interval) == 0:
            texts = extract_texts(ocr, target, args.score_thresh)
            latest_text = " | ".join(texts) if texts else ""
            if latest_text:
                latest_time = time.time()
                print(f"[OCR] {latest_text}")

        text_to_show = latest_text if latest_text else "(no text)"
        if latest_time > 0:
            age = time.time() - latest_time
            text_to_show = f"{text_to_show}  ({age:.1f}s)"

        cv2.putText(
            view,
            text_to_show[:120],
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Hikvision OCR", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
