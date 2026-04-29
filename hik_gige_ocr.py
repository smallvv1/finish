import argparse
import os
import time
from typing import List, Optional

import cv2
import numpy as np
from paddleocr import PaddleOCR


def extract_texts(ocr_engine: PaddleOCR, image) -> List[str]:
    texts: List[str] = []

    if hasattr(ocr_engine, "predict"):
        result = list(ocr_engine.predict(image))
        if result:
            first = result[0]
            if isinstance(first, dict):
                rec_texts = first.get("rec_texts") or []
                texts.extend([str(item).strip() for item in rec_texts if str(item).strip()])
                return texts

    legacy_result = ocr_engine.ocr(image, cls=True)
    if not legacy_result:
        return texts

    lines = legacy_result[0] if isinstance(legacy_result, list) and legacy_result else []
    for line in lines:
        if (
            isinstance(line, (list, tuple))
            and len(line) > 1
            and isinstance(line[1], (list, tuple))
            and len(line[1]) > 0
        ):
            text = str(line[1][0]).strip()
            if text:
                texts.append(text)

    return texts


def _find_device_index_by_ip(device_info_list, ip: str) -> Optional[int]:
    for index, info in enumerate(device_info_list):
        text = str(info)
        if ip in text:
            return index
    return None


def _component_to_bgr(component) -> np.ndarray:
    height = int(component.height)
    width = int(component.width)
    data = component.data

    arr = np.asarray(data)

    if arr.ndim == 1:
        if getattr(component, "num_components_per_pixel", 1) == 1:
            gray = arr.reshape(height, width)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        channels = int(getattr(component, "num_components_per_pixel", 3))
        img = arr.reshape(height, width, channels)
        if channels == 3:
            return img
        if channels == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    if arr.ndim == 3:
        if arr.shape[2] == 3:
            return arr
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    raise RuntimeError("暂不支持的像素格式，请在 MVS 中改为 Mono8 或 RGB8")


def main() -> None:
    parser = argparse.ArgumentParser(description="海康 GigE 工业相机实时OCR（MV-CE120-10GM）")
    parser.add_argument("--ip", type=str, default="169.254.157.82", help="摄像头IP")
    parser.add_argument("--cti-path", type=str, default="", help="GenTL CTI 文件路径（来自 MVS）")
    parser.add_argument("--lang", type=str, default="ch", help="OCR语言，如 ch/en")
    parser.add_argument("--ocr-interval", type=int, default=8, help="每隔多少帧做一次OCR")
    args = parser.parse_args()

    try:
        from harvesters.core import Harvester
    except Exception as error:
        raise RuntimeError(
            "未安装 harvesters，请先执行: D:/python3.11/python.exe -m pip install harvesters"
        ) from error

    cti_path = args.cti_path.strip() or os.environ.get("HIK_GENTL_CTI", "")
    if not cti_path:
        raise RuntimeError(
            "未提供 CTI 路径。请安装海康 MVS 后，传入 --cti-path，"
            "例如: C:/Program Files/MVS/Development/GenTL/Win64/MVGigE_GenTL.cti"
        )

    if not os.path.exists(cti_path):
        raise FileNotFoundError(f"CTI 文件不存在: {cti_path}")

    print(f"[INFO] 使用 CTI: {cti_path}")
    print(f"[INFO] 目标相机IP: {args.ip}")

    h = Harvester()
    h.add_file(cti_path)
    h.update()

    if not h.device_info_list:
        raise RuntimeError("未发现任何 GigE 相机。请检查网线、IP 网段和防火墙")

    print("[INFO] 发现设备:")
    for idx, info in enumerate(h.device_info_list):
        print(f"  [{idx}] {info}")

    device_index = _find_device_index_by_ip(h.device_info_list, args.ip)
    if device_index is None:
        raise RuntimeError(f"未在设备列表中找到 IP={args.ip} 的相机")

    ia = h.create(device_index)
    ia.start()

    ocr = PaddleOCR(use_textline_orientation=True, lang=args.lang, enable_mkldnn=False)
    print("[INFO] 连接成功，按 q 退出")

    frame_count = 0
    latest_text = ""
    latest_time = 0.0

    try:
        while True:
            with ia.fetch(timeout=2.0) as buffer:
                component = buffer.payload.components[0]
                frame = _component_to_bgr(component)

            frame_count += 1
            if frame_count % max(1, args.ocr_interval) == 0:
                texts = extract_texts(ocr, frame)
                latest_text = " | ".join(texts) if texts else ""
                if latest_text:
                    latest_time = time.time()
                    print(f"[OCR] {latest_text}")

            text_to_show = latest_text if latest_text else "(no text)"
            if latest_time > 0:
                age = time.time() - latest_time
                text_to_show = f"{text_to_show} ({age:.1f}s)"

            cv2.putText(
                frame,
                text_to_show[:120],
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Hikrobot GigE OCR", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        ia.stop()
        ia.destroy()
        h.reset()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
