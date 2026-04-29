#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime as dt
import os
import sys
import time
from ctypes import POINTER, byref, c_ubyte, cast

import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from MvCameraControl_class import *
from CameraParams_header import *


MODE_CFG = {
    "ultrafast": {"exposure": 22000.0, "gain": 10.0, "frames": 1},
    "fast": {"exposure": 10000.0, "gain": 7.0, "frames": 3},
    "bright": {"exposure": 26000.0, "gain": 12.0, "frames": 8},
    "quality": {"exposure": 32000.0, "gain": 10.0, "frames": 12},
}

AE_PROFILE_TARGET = {
    "day": 105,
    "night": 155,
}


def apply_auto_exposure_target(camera, target_brightness: int) -> bool:
    target = int(max(30, min(220, int(target_brightness))))
    candidates = [
        "AutoExposureTargetBrightness",
        "AutoExposureTargetGrayValue",
        "TargetBrightness",
        "TargetGrayValue",
    ]
    for node_name in candidates:
        ret = camera.MV_CC_SetIntValue(node_name, target)
        if ret == 0:
            print(f"[INFO] Auto exposure target: {target} (node={node_name})")
            return True
    print(f"[WARN] Auto exposure target unsupported on this camera (target={target})")
    return False


def convert_frame(st_frame_info, data_buf):
    frame = np.frombuffer(data_buf, dtype=np.uint8, count=st_frame_info.nFrameLen)
    w = int(st_frame_info.nWidth)
    h = int(st_frame_info.nHeight)
    pixel_type = int(st_frame_info.enPixelType)

    if pixel_type == 17301505:  # Mono8
        if len(frame) != w * h:
            return None
        frame = frame.reshape((h, w))
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    if pixel_type == 17301513:  # BayerRG8
        if len(frame) != w * h:
            return None
        frame = frame.reshape((h, w))
        frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
        return frame

    if pixel_type == 17301514:  # BayerGB8
        if len(frame) != w * h:
            return None
        frame = frame.reshape((h, w))
        frame = cv2.cvtColor(frame, cv2.COLOR_BayerGB2BGR)
        return frame

    if len(frame) == w * h * 3:
        frame = frame.reshape((h, w, 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    return None


def main():
    parser = argparse.ArgumentParser(description="HK camera capture tool")
    parser.add_argument("--mode", choices=["ultrafast", "fast", "bright", "quality"], default="fast")
    parser.add_argument("--count", type=int, default=1, help="capture count")
    parser.add_argument("--headless", action="store_true", help="compat arg for web mode")
    parser.add_argument("--auto-exposure", dest="auto_exposure", action="store_true", default=True,
                        help="enable camera auto exposure (default: on)")
    parser.add_argument("--manual-exposure", dest="auto_exposure", action="store_false",
                        help="disable auto exposure and use manual exposure")
    parser.add_argument("--auto-gain", dest="auto_gain", action="store_true", default=True,
                        help="enable camera auto gain (default: on)")
    parser.add_argument("--manual-gain", dest="auto_gain", action="store_false",
                        help="disable auto gain and use manual gain")
    parser.add_argument("--exposure", type=float, default=None,
                        help="manual exposure time in us, default uses mode preset")
    parser.add_argument("--gain", type=float, default=None,
                        help="manual gain value, default uses mode preset")
    parser.add_argument("--ae-settle-frames", type=int, default=6,
                        help="discard N warmup frames to let auto exposure settle")
    parser.add_argument("--ae-profile", choices=["day", "night", "custom"], default="day",
                        help="auto exposure profile: day/night/custom")
    parser.add_argument("--ae-target-brightness", type=int, default=120,
                        help="target brightness for custom profile")
    args = parser.parse_args()

    cfg = MODE_CFG[args.mode]
    manual_exposure = float(args.exposure) if args.exposure is not None else float(cfg["exposure"])
    manual_gain = float(args.gain) if args.gain is not None else float(cfg["gain"])
    ae_target = int(args.ae_target_brightness) if args.ae_profile == "custom" else int(AE_PROFILE_TARGET[args.ae_profile])

    camera = None
    initialized = False

    try:
        if hasattr(MvCamera, "MV_CC_Initialize"):
            ret = MvCamera.MV_CC_Initialize()
            if ret != 0:
                print(f"[ERROR] SDK init failed: ret={ret}")
                sys.exit(1)
            initialized = True

        device_list = MV_CC_DEVICE_INFO_LIST()
        n_layer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
        gntl_cameralink_device = globals().get("MV_GENTL_CAMERALINK_DEVICE")
        if gntl_cameralink_device is not None:
            n_layer_type |= gntl_cameralink_device
        mv_cameralink_device = globals().get("MV_CAMERALINK_DEVICE")
        if mv_cameralink_device is not None:
            n_layer_type |= mv_cameralink_device

        ret = MvCamera.MV_CC_EnumDevices(n_layer_type, device_list)
        if ret != 0:
            print(f"[ERROR] Enum devices failed: ret={ret}")
            sys.exit(1)

        print(f"Found devices: {device_list.nDeviceNum}")
        if device_list.nDeviceNum == 0:
            print("[ERROR] No camera found")
            sys.exit(1)

        st_device = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        camera = MvCamera()

        ret = camera.MV_CC_CreateHandle(st_device)
        if ret != 0:
            print(f"[ERROR] Create handle failed: ret={ret}")
            sys.exit(1)

        open_ret = -1
        for _ in range(5):
            open_ret = camera.MV_CC_OpenDevice()
            if open_ret == 0:
                break
            time.sleep(0.3)
        if open_ret != 0:
            print(f"[ERROR] Open device failed: ret={open_ret}")
            sys.exit(1)

        if args.auto_exposure:
            camera.MV_CC_SetEnumValue("ExposureAuto", 2)
            apply_auto_exposure_target(camera, ae_target)
        else:
            camera.MV_CC_SetEnumValue("ExposureAuto", 0)
            camera.MV_CC_SetFloatValue("ExposureTime", manual_exposure)

        if args.auto_gain:
            camera.MV_CC_SetEnumValue("GainAuto", 2)
        else:
            camera.MV_CC_SetEnumValue("GainAuto", 0)
            camera.MV_CC_SetFloatValue("Gain", manual_gain)

        print(
            f"[INFO] Exposure={'auto' if args.auto_exposure else f'manual({manual_exposure:.1f}us)'} | "
            f"Gain={'auto' if args.auto_gain else f'manual({manual_gain:.1f})'}"
        )

        ret = camera.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"[ERROR] Start grabbing failed: ret={ret}")
            sys.exit(1)

        st_param = MVCC_INTVALUE()
        ret = camera.MV_CC_GetIntValue("PayloadSize", st_param)
        if ret != 0:
            print(f"[ERROR] Get payload size failed: ret={ret}")
            sys.exit(1)

        payload_size = int(st_param.nCurValue)
        st_frame_info = MV_FRAME_OUT_INFO_EX()

        settle_frames = max(0, int(args.ae_settle_frames)) if (args.auto_exposure or args.auto_gain) else 0
        for _ in range(settle_frames):
            data_buf = (c_ubyte * payload_size)()
            camera.MV_CC_GetOneFrameTimeout(byref(data_buf), payload_size, st_frame_info, 500)

        for i in range(args.count):
            best = None
            for _ in range(int(cfg["frames"])):
                data_buf = (c_ubyte * payload_size)()
                ret = camera.MV_CC_GetOneFrameTimeout(byref(data_buf), payload_size, st_frame_info, 1200)
                if ret != 0:
                    continue
                frame = convert_frame(st_frame_info, data_buf)
                if frame is not None:
                    best = frame
                    break

            if best is None:
                print(f"[ERROR] Capture failed at index {i + 1}")
                sys.exit(1)

            filename = f"capture_clear_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            out_path = os.path.join(CURRENT_DIR, filename)
            ok = cv2.imwrite(out_path, best, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                print(f"[ERROR] Save image failed: {out_path}")
                sys.exit(1)
            print(f"[OK] Saved: {out_path}")

        print("[OK] Capture completed")

    finally:
        if camera is not None:
            try:
                camera.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                camera.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                camera.MV_CC_DestroyHandle()
            except Exception:
                pass

        if initialized and hasattr(MvCamera, "MV_CC_Finalize"):
            try:
                MvCamera.MV_CC_Finalize()
            except Exception:
                pass


if __name__ == "__main__":
    main()
