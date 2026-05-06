#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from ctypes import POINTER, byref, c_ubyte, cast
from pathlib import Path

import cv2

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import test_hk_opecv as hk


def _open_camera(device_index: int, args):
    initialized = False
    print(f"[LIVE] initializing Hik SDK, camera_index={device_index}", flush=True)
    if hasattr(hk.MvCamera, "MV_CC_Initialize"):
        ret = hk.MvCamera.MV_CC_Initialize()
        if ret != 0:
            raise RuntimeError(f"SDK init failed: ret={ret}")
        initialized = True

    print("[LIVE] enumerating Hik devices", flush=True)
    device_list = hk.MV_CC_DEVICE_INFO_LIST()
    n_layer_type = hk.MV_GIGE_DEVICE | hk.MV_USB_DEVICE
    gntl = getattr(hk, "MV_GENTL_CAMERALINK_DEVICE", None)
    if gntl is not None:
        n_layer_type |= gntl
    cameralink = getattr(hk, "MV_CAMERALINK_DEVICE", None)
    if cameralink is not None:
        n_layer_type |= cameralink

    ret = hk.MvCamera.MV_CC_EnumDevices(n_layer_type, device_list)
    if ret != 0:
        raise RuntimeError(f"Enum devices failed: ret={ret}")
    if int(device_list.nDeviceNum) <= 0:
        raise RuntimeError("No Hikvision camera found")
    if device_index < 0 or device_index >= int(device_list.nDeviceNum):
        raise RuntimeError(f"Camera index out of range: {device_index}, found={device_list.nDeviceNum}")
    print(f"[LIVE] found {device_list.nDeviceNum} Hik device(s), opening index {device_index}", flush=True)

    st_device = cast(device_list.pDeviceInfo[device_index], POINTER(hk.MV_CC_DEVICE_INFO)).contents
    camera = hk.MvCamera()
    ret = camera.MV_CC_CreateHandle(st_device)
    if ret != 0:
        raise RuntimeError(f"Create handle failed: ret={ret}")

    open_ret = -1
    for _ in range(5):
        open_ret = camera.MV_CC_OpenDevice()
        if open_ret == 0:
            break
        time.sleep(0.2)
    if open_ret != 0:
        camera.MV_CC_DestroyHandle()
        raise RuntimeError(f"Open device failed: ret={open_ret}")
    print("[LIVE] device opened", flush=True)

    if args.auto_exposure:
        camera.MV_CC_SetEnumValue("ExposureAuto", 2)
        target = int(args.ae_target_brightness)
        hk.apply_auto_exposure_target(camera, target)
    else:
        camera.MV_CC_SetEnumValue("ExposureAuto", 0)
        camera.MV_CC_SetFloatValue("ExposureTime", float(args.exposure))

    if args.auto_gain:
        camera.MV_CC_SetEnumValue("GainAuto", 2)
    else:
        camera.MV_CC_SetEnumValue("GainAuto", 0)
        camera.MV_CC_SetFloatValue("Gain", float(args.gain))

    try:
        packet = int(camera.MV_CC_GetOptimalPacketSize())
        if packet > 0:
            camera.MV_CC_SetIntValue("GevSCPSPacketSize", packet)
    except Exception:
        pass

    ret = camera.MV_CC_StartGrabbing()
    if ret != 0:
        camera.MV_CC_CloseDevice()
        camera.MV_CC_DestroyHandle()
        raise RuntimeError(f"Start grabbing failed: ret={ret}")
    print("[LIVE] grabbing started", flush=True)

    st_param = hk.MVCC_INTVALUE()
    ret = camera.MV_CC_GetIntValue("PayloadSize", st_param)
    if ret != 0:
        camera.MV_CC_StopGrabbing()
        camera.MV_CC_CloseDevice()
        camera.MV_CC_DestroyHandle()
        raise RuntimeError(f"Get payload size failed: ret={ret}")
    print(f"[LIVE] payload_size={int(st_param.nCurValue)}", flush=True)

    return camera, initialized, int(st_param.nCurValue)


def _close_camera(camera, initialized: bool) -> None:
    if camera is not None:
        for name in ("MV_CC_StopGrabbing", "MV_CC_CloseDevice", "MV_CC_DestroyHandle"):
            try:
                getattr(camera, name)()
            except Exception:
                pass
    if initialized and hasattr(hk.MvCamera, "MV_CC_Finalize"):
        try:
            hk.MvCamera.MV_CC_Finalize()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent Hikvision live preview worker")
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--auto-exposure", dest="auto_exposure", action="store_true", default=True)
    parser.add_argument("--manual-exposure", dest="auto_exposure", action="store_false")
    parser.add_argument("--auto-gain", dest="auto_gain", action="store_true", default=True)
    parser.add_argument("--manual-gain", dest="auto_gain", action="store_false")
    parser.add_argument("--exposure", type=float, default=10000.0)
    parser.add_argument("--gain", type=float, default=7.0)
    parser.add_argument("--ae-target-brightness", type=int, default=105)
    parser.add_argument("--ae-settle-frames", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    args = parser.parse_args()

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    interval = 1.0 / max(1.0, float(args.fps))

    camera = None
    initialized = False
    try:
        camera, initialized, payload_size = _open_camera(int(args.camera_index), args)
        frame_info = hk.MV_FRAME_OUT_INFO_EX()
        settle = max(0, int(args.ae_settle_frames)) if (args.auto_exposure or args.auto_gain) else 0
        for _ in range(settle):
            buf = (c_ubyte * payload_size)()
            camera.MV_CC_GetOneFrameTimeout(byref(buf), payload_size, frame_info, 500)

        frame_count = 0
        while True:
            t0 = time.perf_counter()
            buf = (c_ubyte * payload_size)()
            ret = camera.MV_CC_GetOneFrameTimeout(byref(buf), payload_size, frame_info, 500)
            if ret == 0:
                frame = hk.convert_frame(frame_info, buf)
                if frame is not None:
                    ok = cv2.imwrite(str(tmp), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
                    if ok:
                        os.replace(str(tmp), str(out))
                        if frame_count == 0:
                            print(f"[LIVE] first frame saved: {out}", flush=True)
                        frame_count += 1
            dt = time.perf_counter() - t0
            if dt < interval:
                time.sleep(interval - dt)
    finally:
        _close_camera(camera, initialized)


if __name__ == "__main__":
    main()
