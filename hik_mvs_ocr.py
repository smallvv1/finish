import argparse
import sys
import time
from ctypes import byref, c_ubyte, cast, memset, POINTER, sizeof
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR


def _append_mvs_python_path(sdk_python_root: Path) -> None:
    mv_import = sdk_python_root / "MvImport"
    if not mv_import.exists():
        raise FileNotFoundError(f"未找到 MvImport: {mv_import}")

    root_str = str(sdk_python_root)
    mv_import_str = str(mv_import)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if mv_import_str not in sys.path:
        sys.path.insert(0, mv_import_str)


class MvsCameraClient:
    def __init__(self, sdk_python_root: Path, target_ip: str):
        _append_mvs_python_path(sdk_python_root)

        from MvCameraControl_class import (  # type: ignore
            MV_ACCESS_Exclusive,
            MV_ACCESS_Control,
            MV_ACCESS_Monitor,
            MV_CC_DEVICE_INFO,
            MV_CC_DEVICE_INFO_LIST,
            MV_FRAME_OUT,
            MV_GIGE_DEVICE,
            MV_TRIGGER_MODE_OFF,
            MvCamera,
            PixelType_Gvsp_BGR8_Packed,
            MV_CC_PIXEL_CONVERT_PARAM,
        )

        self.MV_ACCESS_Exclusive = MV_ACCESS_Exclusive
        self.MV_ACCESS_Control = MV_ACCESS_Control
        self.MV_ACCESS_Monitor = MV_ACCESS_Monitor
        self.MV_CC_DEVICE_INFO = MV_CC_DEVICE_INFO
        self.MV_CC_DEVICE_INFO_LIST = MV_CC_DEVICE_INFO_LIST
        self.MV_FRAME_OUT = MV_FRAME_OUT
        self.MV_GIGE_DEVICE = MV_GIGE_DEVICE
        self.MV_TRIGGER_MODE_OFF = MV_TRIGGER_MODE_OFF
        self.MvCamera = MvCamera
        self.PixelType_Gvsp_BGR8_Packed = PixelType_Gvsp_BGR8_Packed
        self.MV_CC_PIXEL_CONVERT_PARAM = MV_CC_PIXEL_CONVERT_PARAM

        self.target_ip = target_ip
        self.cam = self.MvCamera()
        self.device_list = self.MV_CC_DEVICE_INFO_LIST()
        self.device_index: Optional[int] = None
        self.opened_access_mode: Optional[int] = None

    @staticmethod
    def _int_ip_to_str(n_ip: int) -> str:
        return f"{(n_ip >> 24) & 0xFF}.{(n_ip >> 16) & 0xFF}.{(n_ip >> 8) & 0xFF}.{n_ip & 0xFF}"

    @staticmethod
    def _bytes_to_text(byte_array) -> str:
        values = []
        for value in byte_array:
            if value == 0:
                break
            values.append(int(value))
        return bytes(values).decode("utf-8", errors="ignore") if values else ""

    def enumerate_devices(self) -> List[Tuple[int, str, str]]:
        ret = self.MvCamera.MV_CC_EnumDevices(self.MV_GIGE_DEVICE, self.device_list)
        if ret != 0:
            raise RuntimeError(f"枚举设备失败, ret=0x{ret:x}")

        results: List[Tuple[int, str, str]] = []
        for index in range(self.device_list.nDeviceNum):
            dev_info = cast(
                self.device_list.pDeviceInfo[index],
                POINTER(self.MV_CC_DEVICE_INFO),
            ).contents
            if dev_info.nTLayerType != self.MV_GIGE_DEVICE:
                continue

            gig_info = dev_info.SpecialInfo.stGigEInfo
            ip_str = self._int_ip_to_str(gig_info.nCurrentIp)
            model_name = self._bytes_to_text(gig_info.chModelName)
            results.append((index, ip_str, model_name))

        return results

    def connect(self, preferred_access_mode: str = "auto") -> None:
        devices = self.enumerate_devices()
        if not devices:
            raise RuntimeError("未发现任何 GigE 设备")

        for idx, ip, _ in devices:
            if ip == self.target_ip:
                self.device_index = idx
                break

        if self.device_index is None:
            raise RuntimeError(f"未找到目标相机 IP={self.target_ip}")

        st_device = cast(
            self.device_list.pDeviceInfo[self.device_index],
            POINTER(self.MV_CC_DEVICE_INFO),
        ).contents

        ret = self.cam.MV_CC_CreateHandle(st_device)
        if ret != 0:
            raise RuntimeError(f"创建句柄失败, ret=0x{ret:x}")

        access_candidates = {
            "exclusive": [self.MV_ACCESS_Exclusive],
            "control": [self.MV_ACCESS_Control],
            "monitor": [self.MV_ACCESS_Monitor],
            "auto": [self.MV_ACCESS_Exclusive, self.MV_ACCESS_Control, self.MV_ACCESS_Monitor],
        }
        if preferred_access_mode not in access_candidates:
            self.cam.MV_CC_DestroyHandle()
            raise ValueError("access_mode 仅支持: auto/exclusive/control/monitor")

        ret = -1
        for mode in access_candidates[preferred_access_mode]:
            ret = self.cam.MV_CC_OpenDevice(mode, 0)
            if ret == 0:
                self.opened_access_mode = mode
                break

        if ret != 0:
            self.cam.MV_CC_DestroyHandle()
            raise RuntimeError(
                f"打开设备失败, ret=0x{ret:x}。请先关闭 MVS 预览界面或将 --access-mode 设为 monitor"
            )

        if self.opened_access_mode != self.MV_ACCESS_Monitor:
            n_packet_size = self.cam.MV_CC_GetOptimalPacketSize()
            if int(n_packet_size) > 0:
                set_ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", int(n_packet_size))
                if set_ret != 0:
                    print(f"[WARN] 设置 GevSCPSPacketSize 失败, ret=0x{set_ret:x}")

            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", self.MV_TRIGGER_MODE_OFF)
            if ret != 0:
                self.close()
                raise RuntimeError(f"设置 TriggerMode 失败, ret=0x{ret:x}")
        else:
            print("[INFO] 当前为 monitor 模式，跳过可写参数设置")

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.close()
            raise RuntimeError(f"开始取流失败, ret=0x{ret:x}")

    def read_bgr_frame(self, timeout_ms: int = 1000) -> Optional[np.ndarray]:
        st_out = self.MV_FRAME_OUT()
        memset(byref(st_out), 0, sizeof(st_out))

        ret = self.cam.MV_CC_GetImageBuffer(st_out, timeout_ms)
        if ret != 0 or not st_out.pBufAddr:
            return None

        width = int(st_out.stFrameInfo.nWidth)
        height = int(st_out.stFrameInfo.nHeight)
        frame_len = int(st_out.stFrameInfo.nFrameLen)

        dst_size = width * height * 3
        dst_buffer = (c_ubyte * dst_size)()

        st_convert = self.MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(st_convert), 0, sizeof(st_convert))
        st_convert.nWidth = width
        st_convert.nHeight = height
        st_convert.pSrcData = st_out.pBufAddr
        st_convert.nSrcDataLen = frame_len
        st_convert.enSrcPixelType = st_out.stFrameInfo.enPixelType
        st_convert.enDstPixelType = self.PixelType_Gvsp_BGR8_Packed
        st_convert.pDstBuffer = cast(dst_buffer, POINTER(c_ubyte))
        st_convert.nDstBufferSize = dst_size

        convert_ret = self.cam.MV_CC_ConvertPixelType(st_convert)
        self.cam.MV_CC_FreeImageBuffer(st_out)

        if convert_ret != 0:
            print(f"[WARN] 像素转换失败, ret=0x{convert_ret:x}")
            return None

        n_dst_len = int(st_convert.nDstLen)
        if n_dst_len <= 0:
            return None

        frame = np.ctypeslib.as_array(dst_buffer)[:n_dst_len].reshape((height, width, 3))
        return frame

    def close(self) -> None:
        try:
            self.cam.MV_CC_StopGrabbing()
        except Exception:
            pass
        try:
            self.cam.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            self.cam.MV_CC_DestroyHandle()
        except Exception:
            pass


def extract_texts(ocr_engine: PaddleOCR, image: np.ndarray) -> List[str]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="海康 MVS(GigE) 实时 OCR")
    parser.add_argument("--ip", type=str, default="169.254.157.82", help="相机IP")
    parser.add_argument(
        "--sdk-python-root",
        type=str,
        default=r"D:/MVS/Development/Samples/Python",
        help="MVS Python 样例根目录",
    )
    parser.add_argument("--lang", type=str, default="ch", help="OCR 语言，如 ch/en")
    parser.add_argument("--ocr-interval", type=int, default=8, help="每隔多少帧做一次OCR")
    parser.add_argument("--list-only", action="store_true", help="仅枚举设备，不启动OCR")
    parser.add_argument("--probe-once", action="store_true", help="连接后抓取一帧并保存到 outputs/test_images")
    parser.add_argument(
        "--access-mode",
        type=str,
        default="auto",
        choices=["auto", "exclusive", "control", "monitor"],
        help="打开设备模式，MVS占用时建议 monitor",
    )
    args = parser.parse_args()

    sdk_python_root = Path(args.sdk_python_root)

    client = MvsCameraClient(sdk_python_root=sdk_python_root, target_ip=args.ip)
    devices = client.enumerate_devices()
    if not devices:
        raise RuntimeError("未枚举到 GigE 设备，请检查网卡和相机连接")

    print("[INFO] 枚举到的 GigE 设备:")
    for idx, ip, model in devices:
        print(f"  [{idx}] ip={ip}, model={model}")

    if args.list_only:
        print("[INFO] 列表模式完成")
        return

    ocr = PaddleOCR(use_textline_orientation=True, lang=args.lang, enable_mkldnn=False)
    client.connect(preferred_access_mode=args.access_mode)

    frame_count = 0
    latest_text = ""
    latest_time = 0.0

    print("[INFO] 已开始取流，按 q 退出")

    if args.probe_once:
        try:
            frame = None
            for _ in range(20):
                frame = client.read_bgr_frame(timeout_ms=1000)
                if frame is not None:
                    break
            if frame is None:
                raise RuntimeError("抓帧失败：未获取到图像")

            output_dir = Path(__file__).resolve().parent / "outputs" / "test_images"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "mvs_probe_frame.jpg"
            cv2.imwrite(str(output_path), frame)
            print(f"[INFO] 抓帧成功，已保存: {output_path}")
            texts = extract_texts(ocr, frame)
            print(f"[OCR] {' | '.join(texts) if texts else '(no text)'}")
            return
        finally:
            client.close()
            cv2.destroyAllWindows()

    try:
        while True:
            frame = client.read_bgr_frame(timeout_ms=1000)
            if frame is None:
                continue

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

            cv2.imshow("Hik MVS OCR", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        client.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
