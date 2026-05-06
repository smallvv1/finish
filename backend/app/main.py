#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime as dt
import importlib
import importlib.util
import os
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from ctypes import POINTER, byref, c_ubyte, cast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from workflow import (
    detect_and_ocr,
    evaluate_missing,
    load_config,
    _resolve_path,
)

# Speed up Paddle initialization in offline environments.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

try:
    import winsound
except ImportError:
    winsound = None

from ultralytics import YOLO
from det import RapidDigitOCR, detect_and_ocr_with_det

try:
    from MvCameraControl_class import *  # noqa: F403
    from CameraParams_header import *  # noqa: F403

    HK_SDK_AVAILABLE = True
except Exception:
    HK_SDK_AVAILABLE = False


HIK_AE_PROFILE_TARGET = {
    "day": 105,
    "night": 155,
}
SLOT_CODE_ORDER = ("TH26", "TH25", "TH20", "TH16", "TH12")
SLOT_ORDER_INDEX = {code: idx for idx, code in enumerate(SLOT_CODE_ORDER)}


def _sort_slot_codes(codes: List[Any]) -> List[str]:
    unique = {str(code).strip().upper() for code in codes if str(code).strip()}
    return sorted(unique, key=lambda code: (SLOT_ORDER_INDEX.get(code, len(SLOT_CODE_ORDER)), code))


class CaptureRequest(BaseModel):
    camera_index: Optional[int] = None


class WorkflowService:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.cfg = load_config(config_path)

        self.model_path = _resolve_path(self.cfg["model"]["path"])
        self.conf_threshold = float(self.cfg["model"].get("conf_threshold", 0.4))
        class_conf_cfg = self.cfg["model"].get("class_conf_thresholds", {})
        self.class_conf_thresholds: Dict[str, float] = {}
        if isinstance(class_conf_cfg, dict):
            for k, v in class_conf_cfg.items():
                try:
                    self.class_conf_thresholds[str(k).strip().lower()] = float(v)
                except Exception:
                    continue

        ocr_cfg = self.cfg.get("ocr", {})
        self.ocr_enabled = bool(ocr_cfg.get("enabled", True))
        self.allow_ocr_failure = bool(ocr_cfg.get("allow_failure", True))
        self.ocr_lang = str(ocr_cfg.get("lang", "en"))
        self.ocr_score_thresh = float(ocr_cfg.get("score_threshold", 0.5))
        self.ocr_use_textline_orientation = bool(ocr_cfg.get("use_textline_orientation", False))

        rules_cfg = self.cfg.get("rules", {})
        self.code_pattern = str(rules_cfg.get("code_pattern", r"TH\d{2}"))
        self.expected_tools = rules_cfg.get("expected_tools", [])
        self.expected_die_codes: List[str] = []
        for item in self.expected_tools:
            class_name = str(item.get("class_name", "")).strip().lower()
            if class_name != "die":
                continue
            required_codes = item.get("required_codes", [])
            if isinstance(required_codes, list):
                self.expected_die_codes = [
                    str(code).strip().upper()
                    for code in required_codes
                    if str(code).strip()
                ]
            break

        output_cfg = self.cfg.get("output", {})
        runtime_dir = _resolve_path(output_cfg.get("runtime_dir", "runtime"))
        self.runtime_dir = runtime_dir
        self.capture_dir = runtime_dir / "captures"
        self.upload_dir = runtime_dir / "uploads"
        self.annotated_dir = runtime_dir / "annotated"
        self.report_dir = runtime_dir / "reports"
        self.preview_dir = runtime_dir / "preview"
        self.tmp_dir = runtime_dir / "tmp"
        for d in [
            self.runtime_dir,
            self.capture_dir,
            self.upload_dir,
            self.annotated_dir,
            self.report_dir,
            self.preview_dir,
            self.tmp_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        os.environ["TEMP"] = str(self.tmp_dir)
        os.environ["TMP"] = str(self.tmp_dir)

        self.camera_index_default = int(self.cfg.get("camera", {}).get("camera_index", 0))
        self.camera_width = int(self.cfg.get("camera", {}).get("width", 0))
        self.camera_height = int(self.cfg.get("camera", {}).get("height", 0))
        self.camera_provider = str(self.cfg.get("camera", {}).get("provider", "opencv")).strip().lower()
        self.camera_live_provider = str(self.cfg.get("camera", {}).get("live_provider", "")).strip().lower()
        self.hik_only_mode = bool(self.cfg.get("camera", {}).get("hik_only_mode", True))
        if not self.camera_live_provider:
            # In hik_script mode, prefer SDK stream for realtime preview.
            self.camera_live_provider = "hik_sdk" if self.camera_provider == "hik_script" else self.camera_provider
        # `hik_script` launches an external process and cannot provide realtime preview.
        # Normalize it to SDK stream mode and let preview logic fallback to OpenCV if needed.
        if self.camera_live_provider == "hik_script":
            self.camera_live_provider = "hik_sdk"
        if self.hik_only_mode:
            # In Hik-only mode both capture/detect and realtime preview must use
            # Hikvision paths; do not fall back to the host webcam.
            if self.camera_provider not in {"hik_script", "hik_sdk"}:
                self.camera_provider = "hik_sdk"
            if self.camera_live_provider not in {"hik_script", "hik_sdk"}:
                self.camera_live_provider = "hik_sdk"
        self.camera_source = str(self.cfg.get("camera", {}).get("source", "")).strip()
        self.camera_source_prefer = bool(self.cfg.get("camera", {}).get("source_prefer", True))
        self.camera_source_fallback_to_index = bool(
            self.cfg.get("camera", {}).get("source_fallback_to_index", True)
        )
        self.hik_auto_exposure = bool(self.cfg.get("camera", {}).get("hik_auto_exposure", True))
        self.hik_auto_gain = bool(self.cfg.get("camera", {}).get("hik_auto_gain", True))
        self.hik_exposure_time = float(self.cfg.get("camera", {}).get("hik_exposure_time", 10000.0))
        self.hik_gain = float(self.cfg.get("camera", {}).get("hik_gain", 7.0))
        self.hik_ae_profile = str(self.cfg.get("camera", {}).get("hik_ae_profile", "day")).strip().lower()
        if self.hik_ae_profile not in {"day", "night", "custom"}:
            self.hik_ae_profile = "day"
        self.hik_ae_target_brightness = int(self.cfg.get("camera", {}).get("hik_ae_target_brightness", 120))
        self.hik_ae_settle_frames = int(self.cfg.get("camera", {}).get("hik_ae_settle_frames", 6))
        self.hik_python_path = str(self.cfg.get("camera", {}).get("hik_python_path", "")).strip()
        self.hik_mvs_home = str(self.cfg.get("camera", {}).get("hik_mvs_home", "")).strip()
        self.hik_dll_path = str(self.cfg.get("camera", {}).get("hik_dll_path", "")).strip()
        capture_cfg = self.cfg.get("capture", {})
        self.capture_script = str(capture_cfg.get("script", "test_hk_opecv.py"))
        self.capture_python = str(capture_cfg.get("python_executable", "")).strip()
        self.capture_mode = str(capture_cfg.get("mode", "fast"))
        self.capture_preview_mode = str(capture_cfg.get("preview_mode", "ultrafast"))
        self.capture_headless = bool(capture_cfg.get("headless", True))
        self.capture_script_timeout_sec = float(capture_cfg.get("script_timeout_sec", 8.0))
        self.preview_interval_sec = float(capture_cfg.get("preview_interval_sec", 0.033))
        self.preview_fps = float(capture_cfg.get("preview_fps", 30.0))
        wenzi_cfg = self.cfg.get("wenzi", {})
        self.use_wenzi_pipeline = bool(wenzi_cfg.get("enabled", True))
        self.wenzi_min_box_area = int(wenzi_cfg.get("min_box_area", 900))
        self.wenzi_max_die_boxes = int(wenzi_cfg.get("max_die_boxes", 0))
        self.wenzi_two_stage = bool(wenzi_cfg.get("two_stage", True))
        self.wenzi_fast_imgsz = int(wenzi_cfg.get("fast_imgsz", 512))
        self.wenzi_full_imgsz = int(wenzi_cfg.get("full_imgsz", 640))
        self.wenzi_two_stage_max_missing_items = int(wenzi_cfg.get("two_stage_max_missing_items", 1))
        self.wenzi_stable_mode = bool(wenzi_cfg.get("stable_mode", True))
        self.wenzi_ocr_fallback_full = bool(wenzi_cfg.get("ocr_fallback_full", False))
        self.wenzi_ocr_fast_max_side = int(wenzi_cfg.get("ocr_fast_max_side", 256))
        self.wenzi_ocr_timeout_ms = int(wenzi_cfg.get("ocr_timeout_ms", 8000))
        self.wenzi_max_ocr_items = int(wenzi_cfg.get("max_ocr_items", 0))
        if self.wenzi_max_die_boxes <= 0:
            # Auto-limit OCR workload on noisy frames:
            # only cap die boxes (tool/pin never run OCR in wenzi pipeline).
            expected_die_count = 0
            for item in self.expected_tools:
                if str(item.get("class_name", "")).strip().lower() == "die":
                    expected_die_count = int(item.get("required_count", 0))
                    break
            if expected_die_count > 0:
                self.wenzi_max_die_boxes = max(expected_die_count + 1, 6)

        self.model: Optional[YOLO] = None
        self.ocr_engine: Optional[Any] = None
        self.ocr_status = "not_initialized"
        self._ocr_attempted = False
        self._load_lock = threading.Lock()
        self._hik_sdk_loaded = HK_SDK_AVAILABLE
        self._preview_lock = threading.Lock()
        self._preview_last_ts = 0.0
        self._live_frame_lock = threading.Lock()
        self._latest_live_frame: Dict[int, np.ndarray] = {}
        self._latest_live_frame_ts: Dict[int, float] = {}
        self._stream_lock = threading.Lock()
        self._stream_camera = None
        self._stream_initialized = False
        self._stream_payload_size = 0
        self._stream_device_index = None
        self._stream_opening = False
        self._stream_last_error = ""
        self._stream_last_error_ts = 0.0
        self._preview_worker_stop = threading.Event()
        self._preview_worker_thread = None
        self._cv_stream_lock = threading.Lock()
        self._cv_stream_cap = None
        self._cv_stream_key = None
        self._analyze_lock = threading.Lock()
        self._die_position_prior_attempted = False
        self.die_position_prior: Dict[str, Tuple[float, float]] = {}

    def _open_hik_device_with_fallback(self, cam) -> int:
        # Keep first-frame latency low: try fast/common modes first.
        # Broad fallback modes can each block in SDK and make page-open very slow.
        access_exclusive = int(globals().get("MV_ACCESS_Exclusive", 1))
        access_control = int(globals().get("MV_ACCESS_Control", 3))
        access_modes = [
            ("Exclusive", access_exclusive),
            ("Control", access_control),
        ]
        last_ret = -1
        mode_logs = []
        for mode_name, mode in access_modes:
            ret = cam.MV_CC_OpenDevice(mode, 0)
            mode_logs.append(f"{mode_name}:{ret}")
            if ret == 0:
                return 0
            last_ret = ret
            time.sleep(0.15)
        hint = ""
        if int(last_ret) == 2147484163:
            hint = " (access denied: device is occupied by another process)"
        raise RuntimeError(
            f"Hik open device failed: ret={last_ret}{hint}, tries=[{', '.join(mode_logs)}]"
        )

    def _get_hik_ae_target(self) -> int:
        if self.hik_ae_profile == "custom":
            target = int(self.hik_ae_target_brightness)
        else:
            target = int(HIK_AE_PROFILE_TARGET.get(self.hik_ae_profile, 105))
        return int(max(30, min(220, target)))

    def _apply_hik_auto_exposure_target(self, cam) -> bool:
        target = self._get_hik_ae_target()
        candidates = [
            "AutoExposureTargetBrightness",
            "AutoExposureTargetGrayValue",
            "TargetBrightness",
            "TargetGrayValue",
        ]
        for node_name in candidates:
            try:
                ret = cam.MV_CC_SetIntValue(node_name, target)
                if ret == 0:
                    return True
            except Exception:
                continue
        return False

    def _apply_hik_exposure_gain(self, cam) -> None:
        if self.hik_auto_exposure:
            cam.MV_CC_SetEnumValue("ExposureAuto", 2)
            self._apply_hik_auto_exposure_target(cam)
        else:
            cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            cam.MV_CC_SetFloatValue("ExposureTime", float(self.hik_exposure_time))

        if self.hik_auto_gain:
            cam.MV_CC_SetEnumValue("GainAuto", 2)
        else:
            cam.MV_CC_SetEnumValue("GainAuto", 0)
            cam.MV_CC_SetFloatValue("Gain", float(self.hik_gain))

    def ensure_model(self) -> None:
        with self._load_lock:
            if self.model is None:
                self.model = YOLO(str(self.model_path))

    def _find_die_reference_image(self) -> Optional[Path]:
        if self.model is None:
            return None
        preferred = self.capture_dir / "20260428_163012.jpg"
        candidates: List[Path] = []
        if preferred.exists():
            candidates.append(preferred)
        for base_dir in [self.capture_dir, self.upload_dir]:
            if not base_dir.exists():
                continue
            files = sorted(
                [p for p in base_dir.glob("*.jpg") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            candidates.extend(files[:20])

        seen = set()
        unique_candidates: List[Path] = []
        for path in candidates:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(path)

        for path in unique_candidates:
            try:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                results = self.model(img, conf=self.conf_threshold, verbose=False, imgsz=960)
                result = results[0]
                names = result.names if hasattr(result, "names") else {}
                die_count = 0
                for box in result.boxes:
                    class_id = int(box.cls.item()) if box.cls is not None else -1
                    class_name = str(names.get(class_id, class_id)).lower()
                    if class_name == "die":
                        die_count += 1
                if die_count >= 5:
                    return path
            except Exception:
                continue
        return None

    def ensure_ocr(self) -> None:
        with self._load_lock:
            if self._ocr_attempted:
                return
            self._ocr_attempted = True

            if not self.ocr_enabled:
                self.ocr_status = "disabled"
                self.ocr_engine = None
                return

            try:
                self.ocr_engine = RapidDigitOCR(text_score=self.ocr_score_thresh)
                self.ocr_status = "rapidocr_ready"
            except Exception as e:
                self.ocr_engine = None
                self.ocr_status = f"failed: {e}"
                if not self.allow_ocr_failure:
                    raise

    def warmup_runtime(self) -> None:
        """Warm up model and OCR once to reduce first-request latency."""
        try:
            self.ensure_model()
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy, conf=self.conf_threshold, verbose=False, imgsz=640)
        except Exception:
            pass
        try:
            self.ensure_ocr()
            if self.ocr_engine is not None:
                self.ocr_engine.warmup()
        except Exception:
            pass

    def _capture_by_index(self, idx: int):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {idx}")
        if self.camera_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        if self.camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame from camera index {idx}")
        return frame

    def _capture_by_source(self, source: str):
        # For Hikvision RTSP streams we prefer FFmpeg backend, then fallback to default.
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")
        if self.camera_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        if self.camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame from camera source: {source}")
        return frame

    def _capture_by_hik_sdk(self, device_index: int):
        self._ensure_hik_sdk()

        if not self._hik_sdk_loaded:
            raise RuntimeError("Hikvision SDK not available. Missing MVS Python bindings.")

        initialized = False
        camera = None
        try:
            if hasattr(MvCamera, "MV_CC_Initialize"):  # noqa: F405
                ret = MvCamera.MV_CC_Initialize()  # noqa: F405
                if ret != 0:
                    raise RuntimeError(f"Hik SDK init failed: {ret}")
                initialized = True

            device_list = MV_CC_DEVICE_INFO_LIST()  # noqa: F405
            n_layer_type = MV_GIGE_DEVICE | MV_USB_DEVICE  # noqa: F405
            gntl_cameralink_device = globals().get("MV_GENTL_CAMERALINK_DEVICE")
            if gntl_cameralink_device is not None:
                n_layer_type |= gntl_cameralink_device
            mv_cameralink_device = globals().get("MV_CAMERALINK_DEVICE")
            if mv_cameralink_device is not None:
                n_layer_type |= mv_cameralink_device

            ret = MvCamera.MV_CC_EnumDevices(n_layer_type, device_list)  # noqa: F405
            if ret != 0:
                raise RuntimeError(f"Hik enum devices failed: ret={ret}")
            if device_list.nDeviceNum == 0:
                raise RuntimeError(
                    f"No Hikvision device found (layer_type={n_layer_type}). "
                    "Please check: camera power/network, MVS can see device, and no app is occupying the camera."
                )

            if device_index < 0 or device_index >= int(device_list.nDeviceNum):
                raise RuntimeError(f"Hik device index out of range: {device_index}, found={device_list.nDeviceNum}")

            st_device = cast(  # noqa: F405
                device_list.pDeviceInfo[device_index], POINTER(MV_CC_DEVICE_INFO)  # noqa: F405
            ).contents

            camera = MvCamera()  # noqa: F405
            ret = camera.MV_CC_CreateHandle(st_device)
            if ret != 0:
                raise RuntimeError(f"Hik create handle failed: {ret}")

            self._open_hik_device_with_fallback(camera)

            if self.camera_width > 0:
                camera.MV_CC_SetIntValue("Width", int(self.camera_width))
            if self.camera_height > 0:
                camera.MV_CC_SetIntValue("Height", int(self.camera_height))

            self._apply_hik_exposure_gain(camera)

            ret = camera.MV_CC_StartGrabbing()
            if ret != 0:
                raise RuntimeError(f"Hik start grabbing failed: {ret}")

            st_param = MVCC_INTVALUE()  # noqa: F405
            ret = camera.MV_CC_GetIntValue("PayloadSize", st_param)
            if ret != 0:
                raise RuntimeError(f"Hik get payload size failed: {ret}")
            payload_size = int(st_param.nCurValue)

            data_buf = (c_ubyte * payload_size)()  # noqa: F405
            st_frame_info = MV_FRAME_OUT_INFO_EX()  # noqa: F405

            settle_frames = max(0, int(self.hik_ae_settle_frames)) if (self.hik_auto_exposure or self.hik_auto_gain) else 0
            for _ in range(settle_frames):
                camera.MV_CC_GetOneFrameTimeout(byref(data_buf), payload_size, st_frame_info, 500)  # noqa: F405

            ret = camera.MV_CC_GetOneFrameTimeout(byref(data_buf), payload_size, st_frame_info, 1500)  # noqa: F405
            if ret != 0:
                raise RuntimeError(f"Hik get frame failed: {ret}")

            frame = np.frombuffer(data_buf, dtype=np.uint8, count=st_frame_info.nFrameLen)
            w = int(st_frame_info.nWidth)
            h = int(st_frame_info.nHeight)
            pixel_type = int(st_frame_info.enPixelType)

            if pixel_type == 17301505:  # Mono8
                if len(frame) != w * h:
                    raise RuntimeError("Hik frame size mismatch (Mono8)")
                frame = frame.reshape((h, w))
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif pixel_type == 17301513:  # BayerRG8
                if len(frame) != w * h:
                    raise RuntimeError("Hik frame size mismatch (BayerRG8)")
                frame = frame.reshape((h, w))
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
            elif pixel_type == 17301514:  # BayerGB8
                if len(frame) != w * h:
                    raise RuntimeError("Hik frame size mismatch (BayerGB8)")
                frame = frame.reshape((h, w))
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerGB2BGR)
            elif len(frame) == w * h * 3:
                frame = frame.reshape((h, w, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                raise RuntimeError(f"Unsupported Hik pixel type: {pixel_type}")

            return frame
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
            if initialized and hasattr(MvCamera, "MV_CC_Finalize"):  # noqa: F405
                try:
                    MvCamera.MV_CC_Finalize()  # noqa: F405
                except Exception:
                    pass

    def _decode_hik_frame(self, st_frame_info, data_buf):
        frame = np.frombuffer(data_buf, dtype=np.uint8, count=st_frame_info.nFrameLen)
        w = int(st_frame_info.nWidth)
        h = int(st_frame_info.nHeight)
        pixel_type = int(st_frame_info.enPixelType)

        if pixel_type == 17301505:  # Mono8
            if len(frame) != w * h:
                raise RuntimeError("Hik frame size mismatch (Mono8)")
            frame = frame.reshape((h, w))
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if pixel_type == 17301513:  # BayerRG8
            if len(frame) != w * h:
                raise RuntimeError("Hik frame size mismatch (BayerRG8)")
            frame = frame.reshape((h, w))
            return cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
        if pixel_type == 17301514:  # BayerGB8
            if len(frame) != w * h:
                raise RuntimeError("Hik frame size mismatch (BayerGB8)")
            frame = frame.reshape((h, w))
            return cv2.cvtColor(frame, cv2.COLOR_BayerGB2BGR)
        if len(frame) == w * h * 3:
            frame = frame.reshape((h, w, 3))
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        raise RuntimeError(f"Unsupported Hik pixel type: {pixel_type}")

    def _close_hik_stream_unlocked(self):
        cam = self._stream_camera
        self._stream_camera = None
        self._stream_payload_size = 0
        self._stream_device_index = None
        if cam is not None:
            try:
                cam.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                cam.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                cam.MV_CC_DestroyHandle()
            except Exception:
                pass
        if self._stream_initialized and hasattr(MvCamera, "MV_CC_Finalize"):  # noqa: F405
            try:
                MvCamera.MV_CC_Finalize()  # noqa: F405
            except Exception:
                pass
        self._stream_initialized = False

    def _close_hik_stream(self):
        with self._stream_lock:
            self._close_hik_stream_unlocked()

    def _close_cv_stream_unlocked(self):
        cap = self._cv_stream_cap
        self._cv_stream_cap = None
        self._cv_stream_key = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _close_cv_stream(self):
        with self._cv_stream_lock:
            self._close_cv_stream_unlocked()

    def _open_cv_stream_unlocked(self, camera_index: Optional[int], provider: str):
        idx = self.camera_index_default if camera_index is None else int(camera_index)
        p = str(provider or "").strip().lower()
        if p in {"hik_sdk", "hik_script"}:
            raise RuntimeError(f"Provider {p} is not a cv stream provider")

        use_source_first = self.camera_source_prefer and bool(self.camera_source)
        source = self.camera_source if use_source_first else idx
        key = (str(p), str(source))
        if self._cv_stream_cap is not None and self._cv_stream_key == key:
            return self._cv_stream_cap

        self._close_cv_stream_unlocked()
        cap = None
        if isinstance(source, str):
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(int(source))

        if not cap.isOpened():
            if use_source_first and self.camera_source_fallback_to_index:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(idx)
                key = (str(p), str(idx))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open live stream source: {source}")

        if self.camera_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        if self.camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._cv_stream_cap = cap
        self._cv_stream_key = key
        return cap

    def _capture_cv_stream_frame(self, camera_index: Optional[int], provider: str):
        with self._cv_stream_lock:
            cap = self._open_cv_stream_unlocked(camera_index, provider)
            ok, frame = cap.read()
            if not ok or frame is None:
                # Reopen once to recover from transient stream errors.
                self._close_cv_stream_unlocked()
                cap = self._open_cv_stream_unlocked(camera_index, provider)
                ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Failed to read frame from live stream")
            return frame

    def _open_hik_stream(self, device_index: int):
        self._ensure_hik_sdk()
        if not self._hik_sdk_loaded:
            raise RuntimeError("Hikvision SDK not available")

        with self._stream_lock:
            # reuse existing stream if same device
            if self._stream_camera is not None and self._stream_device_index == device_index and self._stream_payload_size > 0:
                return

            self._close_hik_stream_unlocked()

            if hasattr(MvCamera, "MV_CC_Initialize"):  # noqa: F405
                ret = MvCamera.MV_CC_Initialize()  # noqa: F405
                if ret != 0:
                    raise RuntimeError(f"Hik SDK init failed: {ret}")
                self._stream_initialized = True

        device_list = MV_CC_DEVICE_INFO_LIST()  # noqa: F405
        n_layer_type = MV_GIGE_DEVICE | MV_USB_DEVICE  # noqa: F405
        gntl_cameralink_device = globals().get("MV_GENTL_CAMERALINK_DEVICE")
        if gntl_cameralink_device is not None:
            n_layer_type |= gntl_cameralink_device
        mv_cameralink_device = globals().get("MV_CAMERALINK_DEVICE")
        if mv_cameralink_device is not None:
            n_layer_type |= mv_cameralink_device

        ret = MvCamera.MV_CC_EnumDevices(n_layer_type, device_list)  # noqa: F405
        if ret != 0:
            raise RuntimeError(f"Hik enum devices failed: ret={ret}")
        if int(device_list.nDeviceNum) == 0:
            raise RuntimeError("No Hikvision device found")
        if device_index < 0 or device_index >= int(device_list.nDeviceNum):
            raise RuntimeError(f"Hik device index out of range: {device_index}, found={device_list.nDeviceNum}")

        st_device = cast(device_list.pDeviceInfo[device_index], POINTER(MV_CC_DEVICE_INFO)).contents  # noqa: F405
        cam = MvCamera()  # noqa: F405
        ret = cam.MV_CC_CreateHandle(st_device)
        if ret != 0:
            raise RuntimeError(f"Hik create handle failed: {ret}")
        try:
            self._open_hik_device_with_fallback(cam)
        except Exception:
            cam.MV_CC_DestroyHandle()
            raise

        if self.camera_width > 0:
            cam.MV_CC_SetIntValue("Width", int(self.camera_width))
        if self.camera_height > 0:
            cam.MV_CC_SetIntValue("Height", int(self.camera_height))

        self._apply_hik_exposure_gain(cam)

        # For GigE cameras, set optimal packet size to reduce first-frame latency
        # and packet resend pressure on Windows NIC drivers.
        try:
            optimal_packet_size = int(cam.MV_CC_GetOptimalPacketSize())
            if optimal_packet_size > 0:
                cam.MV_CC_SetIntValue("GevSCPSPacketSize", optimal_packet_size)
        except Exception:
            pass

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            raise RuntimeError(f"Hik start grabbing failed: {ret}")

        st_param = MVCC_INTVALUE()  # noqa: F405
        ret = cam.MV_CC_GetIntValue("PayloadSize", st_param)
        if ret != 0:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            raise RuntimeError(f"Hik get payload size failed: {ret}")
        with self._stream_lock:
            self._stream_camera = cam
            self._stream_payload_size = int(st_param.nCurValue)
            self._stream_device_index = int(device_index)
            self._stream_last_error = ""
            self._stream_last_error_ts = 0.0

        settle_frames = max(0, int(self.hik_ae_settle_frames)) if (self.hik_auto_exposure or self.hik_auto_gain) else 0
        if settle_frames > 0:
            with self._stream_lock:
                warm_cam = self._stream_camera
                warm_payload = int(self._stream_payload_size)
            if warm_cam is not None and warm_payload > 0:
                warm_buf = (c_ubyte * warm_payload)()  # noqa: F405
                warm_info = MV_FRAME_OUT_INFO_EX()  # noqa: F405
                for _ in range(settle_frames):
                    warm_cam.MV_CC_GetOneFrameTimeout(byref(warm_buf), warm_payload, warm_info, 300)  # noqa: F405

    def _capture_hik_stream_frame(self, device_index: int):
        self._open_hik_stream(device_index)
        with self._stream_lock:
            cam = self._stream_camera
            payload_size = self._stream_payload_size
            if cam is None or payload_size <= 0:
                raise RuntimeError("Hik stream is not ready")
            data_buf = (c_ubyte * payload_size)()  # noqa: F405
            st_frame_info = MV_FRAME_OUT_INFO_EX()  # noqa: F405
            # Keep timeout short so preview endpoint fails fast and retries,
            # instead of blocking the browser for a long time.
            ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), payload_size, st_frame_info, 220)  # noqa: F405
            if ret != 0:
                raise RuntimeError(f"Hik stream get frame failed: {ret}")
            return self._decode_hik_frame(st_frame_info, data_buf)

    def warmup_camera_stream(self, device_index: Optional[int] = None) -> None:
        if self.camera_live_provider != "hik_sdk":
            return
        if self._stream_opening:
            return
        self._stream_opening = True
        try:
            idx = int(self.camera_index_default if device_index is None else device_index)
            self._open_hik_stream(idx)
            # Prime one frame so first browser request is more likely instant.
            try:
                frame = self._capture_hik_stream_frame(idx)
                self._cache_live_frame(idx, frame)
            except Exception:
                pass
        except Exception as e:
            # Non-fatal: stream endpoint will retry on demand.
            with self._stream_lock:
                self._stream_last_error = str(e)
                self._stream_last_error_ts = time.time()
        finally:
            self._stream_opening = False

    def _ensure_hik_stream_async(self, device_index: int) -> None:
        if self._stream_opening:
            return
        t = threading.Thread(target=self.warmup_camera_stream, args=(device_index,), daemon=True)
        t.start()

    def start_preview_worker(self) -> None:
        if self.camera_live_provider != "hik_sdk":
            return
        if self._preview_worker_thread is not None and self._preview_worker_thread.is_alive():
            return
        self._preview_worker_stop.clear()
        self._preview_worker_thread = threading.Thread(target=self._preview_worker_loop, daemon=True)
        self._preview_worker_thread.start()

    def stop_preview_worker(self) -> None:
        try:
            self._preview_worker_stop.set()
        except Exception:
            pass

    def _preview_worker_loop(self) -> None:
        idx = int(self.camera_index_default)
        while not self._preview_worker_stop.is_set():
            try:
                if not self._hik_stream_ready(idx):
                    self.warmup_camera_stream(idx)
                if self._hik_stream_ready(idx):
                    frame = self._capture_hik_stream_frame(idx)
                    self._cache_live_frame(idx, frame)
                    time.sleep(0.05)
                    continue
            except Exception as e:
                with self._stream_lock:
                    self._stream_last_error = str(e)
                    self._stream_last_error_ts = time.time()
            time.sleep(0.5)

    def _hik_stream_ready(self, device_index: int) -> bool:
        with self._stream_lock:
            return (
                self._stream_camera is not None
                and self._stream_device_index == int(device_index)
                and self._stream_payload_size > 0
            )

    def _ensure_hik_sdk(self):
        if self._hik_sdk_loaded:
            return

        if self.hik_mvs_home:
            os.environ["MVS_HOME"] = self.hik_mvs_home
        if self.hik_dll_path:
            os.environ["MVS_DLL_PATH"] = self.hik_dll_path
            dll_dir = str(Path(self.hik_dll_path).resolve().parent)
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(dll_dir)
                except Exception:
                    pass

        if self.hik_python_path:
            py_path = str(Path(self.hik_python_path).resolve())
            if py_path not in sys.path:
                sys.path.append(py_path)
            mv_import = Path(py_path) / "MvImport"
            if mv_import.exists():
                mv_import_path = str(mv_import.resolve())
                if mv_import_path not in sys.path:
                    sys.path.append(mv_import_path)
            if hasattr(os, "add_dll_directory"):
                for p in [Path(py_path), Path(py_path) / "MvImport"]:
                    if p.exists():
                        try:
                            os.add_dll_directory(str(p.resolve()))
                        except Exception:
                            pass

        try:
            # Prefer official MVS Python wrappers when hik_python_path is provided.
            selected_mv = None
            selected_hdr = None

            if self.hik_python_path:
                py_root = Path(self.hik_python_path).resolve()
                candidates = [py_root, py_root / "MvImport"]
                for base in candidates:
                    mv_file = base / "MvCameraControl_class.py"
                    hdr_file = base / "CameraParams_header.py"
                    if mv_file.exists() and hdr_file.exists():
                        base_str = str(base)
                        if base_str not in sys.path:
                            sys.path.append(base_str)
                        selected_mv = mv_file
                        selected_hdr = hdr_file
                        break

            if selected_mv is None:
                # Fallback to local wrappers in this project.
                local_mv = PROJECT_ROOT / "MvCameraControl_class.py"
                local_hdr = PROJECT_ROOT / "CameraParams_header.py"
                if not local_mv.exists():
                    raise RuntimeError(f"Local wrapper not found: {local_mv}")
                selected_mv = local_mv
                selected_hdr = local_hdr if local_hdr.exists() else None

            mv_spec = importlib.util.spec_from_file_location("MvCameraControl_class", str(selected_mv))
            if mv_spec is None or mv_spec.loader is None:
                raise RuntimeError(f"Cannot create import spec for: {selected_mv}")
            mv_mod = importlib.util.module_from_spec(mv_spec)
            sys.modules["MvCameraControl_class"] = mv_mod
            mv_spec.loader.exec_module(mv_mod)

            # Load matching header from the same wrapper source first.
            if selected_hdr is not None and selected_hdr.exists():
                hdr_spec = importlib.util.spec_from_file_location("CameraParams_header", str(selected_hdr))
                if hdr_spec is None or hdr_spec.loader is None:
                    raise RuntimeError(f"Cannot create import spec for: {selected_hdr}")
                hdr_mod = importlib.util.module_from_spec(hdr_spec)
                sys.modules["CameraParams_header"] = hdr_mod
                hdr_spec.loader.exec_module(hdr_mod)
            else:
                hdr_mod = importlib.import_module("CameraParams_header")

            globals()["MvCamera"] = mv_mod.MvCamera
            globals()["MV_CC_DEVICE_INFO_LIST"] = hdr_mod.MV_CC_DEVICE_INFO_LIST
            globals()["MV_GIGE_DEVICE"] = hdr_mod.MV_GIGE_DEVICE
            globals()["MV_USB_DEVICE"] = hdr_mod.MV_USB_DEVICE
            globals()["MV_CC_DEVICE_INFO"] = hdr_mod.MV_CC_DEVICE_INFO
            globals()["MVCC_INTVALUE"] = hdr_mod.MVCC_INTVALUE
            globals()["MV_FRAME_OUT_INFO_EX"] = hdr_mod.MV_FRAME_OUT_INFO_EX
            self._hik_sdk_loaded = True
        except Exception as e:
            self._hik_sdk_loaded = False
            raise RuntimeError(f"Hikvision SDK import failed: {e}")

    def debug_hik_devices(self) -> Dict:
        self._ensure_hik_sdk()
        device_list = MV_CC_DEVICE_INFO_LIST()  # noqa: F405
        n_layer_type = MV_GIGE_DEVICE | MV_USB_DEVICE  # noqa: F405
        gntl_cameralink_device = globals().get("MV_GENTL_CAMERALINK_DEVICE")
        if gntl_cameralink_device is not None:
            n_layer_type |= gntl_cameralink_device
        mv_cameralink_device = globals().get("MV_CAMERALINK_DEVICE")
        if mv_cameralink_device is not None:
            n_layer_type |= mv_cameralink_device
        ret = MvCamera.MV_CC_EnumDevices(n_layer_type, device_list)  # noqa: F405
        return {
            "sdk_loaded": bool(self._hik_sdk_loaded),
            "layer_type": int(n_layer_type),
            "enum_ret": int(ret),
            "device_count": int(getattr(device_list, "nDeviceNum", 0)),
        }

    def capture_image(self, camera_index: Optional[int]) -> Path:
        idx = self.camera_index_default if camera_index is None else int(camera_index)

        # Always prefer latest cached preview frame to minimize click-to-result latency.
        cached = self._get_cached_live_frame(idx, max_age_sec=300.0)
        if cached is not None and self._is_usable_frame(cached):
            filename = f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = self.capture_dir / filename
            cv2.imwrite(str(path), cached)
            return path

        # The realtime Hik SDK path can block for a long time when the device is busy.
        # For explicit capture, use the bounded script path so requests fail by timeout
        # instead of occupying the detection lock indefinitely.
        if str(self.camera_provider).strip().lower() == "hik_sdk":
            fallback = self._copy_latest_saved_image_to_capture_dir(max_age_sec=300.0)
            if fallback is not None:
                return fallback
            try:
                return self._capture_by_hik_script(self.capture_mode)
            except Exception as e:
                fallback = self._copy_latest_saved_image_to_capture_dir()
                if fallback is not None:
                    return fallback
                raise RuntimeError(str(e))

        try:
            cached = self._get_live_frame_by_provider(idx, self.camera_provider)
            if self._is_usable_frame(cached):
                self._cache_live_frame(idx, cached)
            else:
                cached = None
        except Exception as e:
            raise RuntimeError(
                f"No recent live frame for camera_index={idx}, and on-demand capture failed: {e}"
            )
        if cached is None:
            fallback = self._copy_latest_saved_image_to_capture_dir()
            if fallback is not None:
                return fallback
            raise RuntimeError(f"No usable frame for camera_index={idx}")

        filename = f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = self.capture_dir / filename
        cv2.imwrite(str(path), cached)
        return path

    def _copy_latest_saved_image_to_capture_dir(self, max_age_sec: Optional[float] = None) -> Optional[Path]:
        candidates: List[Path] = []
        for base in [self.preview_dir, self.capture_dir, self.upload_dir]:
            if not base.exists():
                continue
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                candidates.extend([p for p in base.glob(pattern) if p.is_file()])
        if not candidates:
            return None
        now = time.time()
        for latest in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
            if max_age_sec is not None and (now - latest.stat().st_mtime) > float(max_age_sec):
                continue
            img = cv2.imread(str(latest))
            if img is None or not self._is_usable_frame(img):
                continue
            out = self.capture_dir / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}{latest.suffix.lower()}"
            ok = cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            return out if ok else None
        return None

    def _cache_live_frame(self, camera_index: Optional[int], frame: np.ndarray) -> None:
        if not self._is_usable_frame(frame):
            return
        idx = self.camera_index_default if camera_index is None else int(camera_index)
        with self._live_frame_lock:
            self._latest_live_frame[idx] = frame.copy()
            self._latest_live_frame_ts[idx] = time.time()

    def _is_usable_frame(self, frame: Optional[np.ndarray]) -> bool:
        if frame is None or getattr(frame, "size", 0) == 0:
            return False
        try:
            mean = float(frame.mean())
            max_value = int(frame.max())
        except Exception:
            return False
        return mean >= 8.0 and max_value >= 40

    def _get_cached_live_frame(self, camera_index: Optional[int], max_age_sec: float) -> Optional[np.ndarray]:
        idx = self.camera_index_default if camera_index is None else int(camera_index)
        with self._live_frame_lock:
            ts = self._latest_live_frame_ts.get(idx)
            frame = self._latest_live_frame.get(idx)
            if ts is None or frame is None:
                return None
            if (time.time() - ts) > max_age_sec:
                return None
            return frame.copy()

    def _capture_by_hik_script(self, mode: Optional[str] = None) -> Path:
        script_path = _resolve_path(self.capture_script)
        if not script_path.exists():
            raise RuntimeError(f"Capture script not found: {script_path}")

        py_exec = self.capture_python or sys.executable
        script_dir = script_path.parent
        before = set(script_dir.glob("capture_clear_*.jpg"))
        mode_name = str(mode or self.capture_mode or "fast").strip()
        cmd_base = [py_exec, str(script_path), "--mode", mode_name, "--count", "1"]
        if self.hik_auto_exposure:
            cmd_base.extend(["--auto-exposure", "--ae-profile", self.hik_ae_profile])
            if self.hik_ae_profile == "custom":
                cmd_base.extend(["--ae-target-brightness", str(int(self.hik_ae_target_brightness))])
            cmd_base.extend(["--ae-settle-frames", str(max(0, int(self.hik_ae_settle_frames)))])
        else:
            cmd_base.extend(["--manual-exposure", "--exposure", str(float(self.hik_exposure_time))])

        if self.hik_auto_gain:
            cmd_base.append("--auto-gain")
        else:
            cmd_base.extend(["--manual-gain", "--gain", str(float(self.hik_gain))])
        cmd = list(cmd_base)
        if self.capture_headless:
            cmd.append("--headless")
        run_kwargs = {
            "cwd": str(script_dir),
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "ignore",
        }
        if os.name == "nt":
            run_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            proc = subprocess.run(cmd, timeout=max(1.0, float(self.capture_script_timeout_sec)), **run_kwargs)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Hik capture script timed out after {self.capture_script_timeout_sec:.1f}s: {' '.join(cmd)}"
            )

        # Some third-party scripts do not accept --headless; retry automatically.
        if (
            proc.returncode != 0
            and self.capture_headless
            and "unrecognized arguments: --headless" in ((proc.stderr or "") + (proc.stdout or ""))
        ):
            cmd = list(cmd_base)
            try:
                proc = subprocess.run(
                    cmd,
                    timeout=max(1.0, float(self.capture_script_timeout_sec)),
                    **run_kwargs,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"Hik capture script timed out after {self.capture_script_timeout_sec:.1f}s: {' '.join(cmd)}"
                )

        if proc.returncode != 0:
            msg = (
                f"Hik capture script failed: returncode={proc.returncode}\n"
                f"cmd={' '.join(cmd)}\n"
                f"stdout:\n{(proc.stdout or '').strip()}\n"
                f"stderr:\n{(proc.stderr or '').strip()}"
            )
            raise RuntimeError(msg)
        after = set(script_dir.glob("capture_clear_*.jpg"))
        new_files = list(after - before)

        candidate: Optional[Path] = None
        if new_files:
            candidate = max(new_files, key=lambda p: p.stat().st_mtime)
        else:
            all_files = list(script_dir.glob("capture_clear_*.jpg"))
            if all_files:
                candidate = max(all_files, key=lambda p: p.stat().st_mtime)

        if candidate is None or not candidate.exists():
            raise RuntimeError("Hik capture script finished but no image was produced")

        filename = f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}{candidate.suffix.lower()}"
        out_path = self.capture_dir / filename
        data = cv2.imread(str(candidate))
        if data is None:
            raise RuntimeError(f"Captured file is not a valid image: {candidate}")
        cv2.imwrite(str(out_path), data)
        return out_path

    def _get_live_frame_by_provider(self, camera_index: Optional[int], provider: str):
        idx = self.camera_index_default if camera_index is None else int(camera_index)
        p = str(provider or "").strip().lower()
        if p == "hik_sdk":
            return self._capture_hik_stream_frame(idx)
        if p == "hik_script":
            src = self._capture_by_hik_script(self.capture_mode)
            img = cv2.imread(str(src))
            try:
                src.unlink(missing_ok=True)
            except Exception:
                pass
            if img is None:
                raise RuntimeError("Failed to decode frame from hik_script capture")
            return img
        if self.hik_only_mode:
            raise RuntimeError(
                f"Provider {p} is disabled in hik_only_mode; only hik_sdk/hik_script are allowed"
            )

        use_source_first = self.camera_source_prefer and bool(self.camera_source)
        if use_source_first:
            try:
                return self._capture_by_source(self.camera_source)
            except Exception:
                if not self.camera_source_fallback_to_index:
                    raise
        return self._capture_by_index(idx)

    def _get_live_frame(self, camera_index: Optional[int]):
        try:
            return self._get_live_frame_by_provider(camera_index, self.camera_provider)
        except Exception:
            # Hard fallback for unstable Hik SDK stream:
            # if script capture exists, use it to guarantee a frame.
            if str(self.camera_provider).lower() == "hik_sdk":
                try:
                    src = self._capture_by_hik_script(self.capture_preview_mode or self.capture_mode)
                    img = cv2.imread(str(src))
                    try:
                        src.unlink(missing_ok=True)
                    except Exception:
                        pass
                    if img is not None:
                        return img
                except Exception:
                    pass
            raise

    def _get_preview_frame(self, camera_index: Optional[int]):
        # Realtime preview can use a different provider than capture/detect.
        idx = self.camera_index_default if camera_index is None else int(camera_index)
        if self.camera_live_provider == "hik_sdk":
            # Do not block preview HTTP responses on SDK open/enum or script fallback.
            if self._hik_stream_ready(idx):
                return self._capture_hik_stream_frame(idx)
            cached = self._get_cached_live_frame(idx, max_age_sec=2.0)
            if cached is not None:
                return cached
            self._ensure_hik_stream_async(idx)
            with self._stream_lock:
                last_error = (self._stream_last_error or "").strip()
            if last_error and not self._stream_opening:
                raise RuntimeError(f"Hik realtime stream open failed: {last_error}")
            raise RuntimeError("Hik realtime stream is initializing")
        if str(self.camera_live_provider).lower() not in {"hik_script", "hik_sdk"}:
            if self.hik_only_mode:
                raise RuntimeError(
                    f"live_provider={self.camera_live_provider} is disabled in hik_only_mode"
                )
            return self._capture_cv_stream_frame(camera_index, self.camera_live_provider)
        try:
            return self._get_live_frame_by_provider(camera_index, self.camera_live_provider)
        except Exception:
            if self.camera_live_provider != self.camera_provider:
                if self.hik_only_mode:
                    raise
                if str(self.camera_provider).lower() not in {"hik_script", "hik_sdk"}:
                    return self._capture_cv_stream_frame(camera_index, self.camera_provider)
                return self._get_live_frame_by_provider(camera_index, self.camera_provider)
            raise

    def capture_live_preview(self, camera_index: Optional[int], force: bool = False) -> Path:
        out = self.preview_dir / "live.jpg"
        now = time.time()

        # Reuse recent preview to avoid expensive repeated camera grabs.
        if (not force) and out.exists() and (now - self._preview_last_ts) < self.preview_interval_sec:
            return out

    def ensure_die_position_prior(self) -> None:
        return

    def _apply_die_position_prior(self, report: Dict[str, Any], image_path: Path) -> None:
        return

        with self._preview_lock:
            now = time.time()
            if (not force) and out.exists() and (now - self._preview_last_ts) < self.preview_interval_sec:
                return out

            img = self._get_live_frame(camera_index)
            ok = cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                raise RuntimeError("Preview write failed")
            self._preview_last_ts = time.time()
            return out

    def analyze(self, image_path: Path, skip_ocr: bool = False) -> Dict:
        self.ensure_model()
        if not skip_ocr:
            self.ensure_ocr()

        ocr_engine = None if skip_ocr else self.ocr_engine

        with self._analyze_lock:
            if self.use_wenzi_pipeline:
                image_reports = detect_and_ocr_with_det(
                    image_paths=[image_path],
                    model=self.model,
                    ocr_engine=ocr_engine,
                    conf_threshold=self.conf_threshold,
                    class_conf_thresholds=self.class_conf_thresholds,
                    code_pattern=self.code_pattern,
                    annotated_dir=self.annotated_dir,
                    min_box_area=self.wenzi_min_box_area,
                    max_die_boxes=self.wenzi_max_die_boxes,
                    ocr_timeout_ms=self.wenzi_ocr_timeout_ms,
                    max_ocr_items=self.wenzi_max_ocr_items,
                    expected_die_codes=self.expected_die_codes,
                    detect_imgsz=self.wenzi_full_imgsz,
                )
                missing_eval = evaluate_missing(image_reports, self.expected_tools)
                # Accuracy-preserving fallback: rerun with full resolution only when
                # fast path indicates missing items (count issue), not code-only gaps.
                missing_items = int(missing_eval.get("total_missing_items", 0))
                if (
                    (not skip_ocr)
                    and (not self.wenzi_stable_mode)
                    and self.wenzi_two_stage
                    and 0 < missing_items <= self.wenzi_two_stage_max_missing_items
                ):
                    image_reports = detect_and_ocr_with_det(
                        image_paths=[image_path],
                        model=self.model,
                        ocr_engine=ocr_engine,
                        conf_threshold=self.conf_threshold,
                        class_conf_thresholds=self.class_conf_thresholds,
                        code_pattern=self.code_pattern,
                        annotated_dir=self.annotated_dir,
                        min_box_area=self.wenzi_min_box_area,
                        max_die_boxes=self.wenzi_max_die_boxes,
                        ocr_timeout_ms=self.wenzi_ocr_timeout_ms,
                        max_ocr_items=self.wenzi_max_ocr_items,
                        expected_die_codes=self.expected_die_codes,
                        detect_imgsz=self.wenzi_full_imgsz,
                    )
                    missing_eval = evaluate_missing(image_reports, self.expected_tools)
            else:
                image_reports = detect_and_ocr(
                    image_paths=[image_path],
                    model=self.model,
                    ocr_engine=ocr_engine,
                    conf_threshold=self.conf_threshold,
                    class_conf_thresholds=self.class_conf_thresholds,
                    score_thresh=self.ocr_score_thresh,
                    code_pattern=self.code_pattern,
                    annotated_dir=self.annotated_dir,
                )
                missing_eval = evaluate_missing(image_reports, self.expected_tools)
        report = image_reports[0] if image_reports else {}

        missing_parts = []
        for item in missing_eval.get("details", []):
            if item.get("missing_count", 0) > 0 or item.get("missing_codes"):
                missing_code_hint = ""
                if item.get("missing_count", 0) > 0 and not item.get("missing_codes"):
                    class_name = str(item.get("class_name", "")).strip().lower()
                    if class_name in {"tool", "pin"}:
                        missing_code_hint = "无编号"
                missing_parts.append(
                    {
                        "class_name": item.get("class_name"),
                        "missing_count": int(item.get("missing_count", 0)),
                        "missing_codes": _sort_slot_codes(item.get("missing_codes", [])),
                        "missing_code_hint": missing_code_hint,
                    }
                )

        # Build per-part OCR summary for frontend display.
        detections = report.get("detections", [])
        detections_by_class: Dict[str, List[Dict]] = {}
        for det in detections:
            class_name = str(det.get("class_name", "unknown"))
            detections_by_class.setdefault(class_name, []).append(det)

        part_summaries: List[Dict] = []
        for detail in missing_eval.get("details", []):
            class_name = str(detail.get("class_name", "unknown"))
            class_dets = detections_by_class.get(class_name, [])
            ocr_texts = []
            ocr_codes = []
            for det in class_dets:
                ocr_texts.extend(det.get("ocr_texts", []))
                ocr_codes.extend(det.get("ocr_codes", []))
            # Keep stable and unique.
            raw_ocr_texts = [str(x).strip() for x in ocr_texts if str(x).strip()]
            ocr_codes = _sort_slot_codes(ocr_codes)
            ocr_texts = []
            seen_texts = set()
            for code in ocr_codes:
                for text in (code[2:], code):
                    if text not in seen_texts:
                        seen_texts.add(text)
                        ocr_texts.append(text)
            for text in raw_ocr_texts:
                if text not in seen_texts:
                    seen_texts.add(text)
                    ocr_texts.append(text)
            part_summaries.append(
                {
                    "class_name": class_name,
                    "required_count": int(detail.get("required_count", 0)),
                    "found_count": int(detail.get("found_count", 0)),
                    "missing_count": int(detail.get("missing_count", 0)),
                    "found_codes": _sort_slot_codes(detail.get("found_codes", [])),
                    "missing_codes": _sort_slot_codes(detail.get("missing_codes", [])),
                    "ocr_texts": ocr_texts,
                    "ocr_codes": ocr_codes,
                }
            )

        die_by_slot: Dict[str, Dict] = {}
        for det in detections:
            if str(det.get("class_name", "")).lower() != "die":
                continue
            slot_code = str(det.get("expected_slot_code", "") or "").strip().upper()
            if not slot_code:
                for code in _sort_slot_codes(det.get("ocr_codes", [])):
                    slot_code = code
                    break
            if slot_code:
                die_by_slot.setdefault(slot_code, det)

        slot_results = []
        for idx, code in enumerate(SLOT_CODE_ORDER):
            det = die_by_slot.get(code)
            recognized_codes = _sort_slot_codes(det.get("ocr_codes", [])) if det else []
            recognized_code = str(det.get("recognized_code", "") or "").strip().upper() if det else ""
            if not recognized_code and recognized_codes:
                recognized_code = recognized_codes[0]
            slot_results.append(
                {
                    "slot_index": idx,
                    "slot_code": code,
                    "expected_code": code,
                    "found": det is not None,
                    "recognized_code": recognized_code,
                    "recognized_codes": recognized_codes,
                    "slot_match": bool(det is not None and recognized_code == code),
                    "detection": det,
                }
            )
        slot_mismatches = [
            {
                "slot_index": item["slot_index"],
                "expected_code": item["expected_code"],
                "recognized_code": item["recognized_code"],
            }
            for item in slot_results
            if item["found"] and not item["slot_match"]
        ]

        ocr_status = "skipped_fast_stage" if skip_ocr else self.ocr_status

        return {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "ocr_status": ocr_status,
            "capture_image_path": str(image_path),
            "annotated_image_path": report.get("annotated_image_path"),
            "timings_ms": report.get("timings_ms", {}),
            "detection_count": report.get("detection_count", 0),
            "missing_status": missing_eval.get("status", "unknown"),
            "missing_total_items": int(missing_eval.get("total_missing_items", 0)),
            "missing_total_codes": int(missing_eval.get("total_missing_codes", 0)),
            "missing_parts": missing_parts,
            "part_summaries": part_summaries,
            "slot_code_order": list(SLOT_CODE_ORDER),
            "recognized_codes_by_slot": [item["recognized_code"] for item in slot_results],
            "slot_mismatches": slot_mismatches,
            "slot_results": slot_results,
            "detections": detections,
        }


    def save_uploaded_image(self, upload: UploadFile) -> Path:
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            ext = ".jpg"
        filename = f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        output = self.upload_dir / filename
        data = upload.file.read()
        if not data:
            raise RuntimeError("Uploaded file is empty")
        with output.open("wb") as f:
            f.write(data)

        # Validate decode to avoid non-image uploads.
        img = cv2.imread(str(output))
        if img is None:
            output.unlink(missing_ok=True)
            raise RuntimeError("Uploaded file is not a valid image")
        return output


def path_to_url(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    p = Path(path_str).resolve()
    try:
        rel = p.relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    return "/" + rel.as_posix()


def trigger_alarm() -> None:
    def _beep():
        if winsound is not None:
            for _ in range(3):
                winsound.Beep(1800, 350)
        else:
            print("\a")

    t = threading.Thread(target=_beep, daemon=True)
    t.start()


def create_app(config_path: Path) -> FastAPI:
    service = WorkflowService(config_path)
    async_jobs: Dict[str, Dict[str, Any]] = {}
    async_jobs_lock = threading.Lock()
    detection_route_lock = threading.Lock()

    def _cleanup_async_jobs(max_age_sec: float = 900.0) -> None:
        now = time.time()
        with async_jobs_lock:
            stale = [
                k
                for k, v in async_jobs.items()
                if (now - float(v.get("updated_ts", v.get("created_ts", now)))) > max_age_sec
            ]
            for k in stale:
                async_jobs.pop(k, None)

    def _finalize_result_urls(result: Dict[str, Any], elapsed_ms: int) -> Dict[str, Any]:
        result["capture_image_url"] = path_to_url(result.get("capture_image_path"))
        result["annotated_image_url"] = path_to_url(result.get("annotated_image_path"))
        for det in result.get("detections", []) or []:
            det["ocr_crop_url"] = path_to_url(det.get("ocr_crop_path"))
        for slot in result.get("slot_results", []) or []:
            det = slot.get("detection")
            if isinstance(det, dict):
                det["ocr_crop_url"] = path_to_url(det.get("ocr_crop_path"))
        result["processing_time_ms"] = int(elapsed_ms)
        result["processing_time_sec"] = round(float(elapsed_ms) / 1000.0, 3)
        return result

    def _latest_nonblocking_image() -> Optional[Path]:
        candidates: List[Path] = []
        for base in [service.preview_dir, service.capture_dir, service.upload_dir]:
            try:
                for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                    candidates.extend([p for p in base.glob(pattern) if p.is_file()])
            except Exception:
                continue
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _start_async_full_ocr(task_id: str, image_path: Path) -> None:
        def _worker() -> None:
            t0 = time.perf_counter()
            try:
                full = service.analyze(image_path, skip_ocr=False)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                _finalize_result_urls(full, elapsed_ms)
                with async_jobs_lock:
                    async_jobs[task_id]["status"] = "done"
                    async_jobs[task_id]["result"] = full
                    async_jobs[task_id]["updated_ts"] = time.time()
            except Exception as e:
                with async_jobs_lock:
                    async_jobs[task_id]["status"] = "error"
                    async_jobs[task_id]["error"] = str(e)
                    async_jobs[task_id]["updated_ts"] = time.time()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Avoid startup warmups here: camera SDK/OCR initialization can block the
        # single web process and make even /api/health wait until frontend timeout.
        # Model and det.py OCR are initialized lazily by the detection request.
        try:
            yield
        finally:
            service.stop_preview_worker()
            try:
                service._close_hik_stream()
            except Exception:
                pass
            try:
                service._close_cv_stream()
            except Exception:
                pass

    app = FastAPI(title="Toolbox Missing Detection Web", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    vue_dist_dir = PROJECT_ROOT / "toolbox-frontend" / "dist"
    legacy_ui_dir = PROJECT_ROOT / "frontend" / "legacy"

    has_vue_dist = vue_dist_dir.exists() and (vue_dist_dir / "index.html").exists()

    def _no_cache_headers() -> Dict[str, str]:
        return {
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        }

    app.mount("/runtime", StaticFiles(directory=str(PROJECT_ROOT / "runtime")), name="runtime")
    if not has_vue_dist:
        app.mount("/web_ui", StaticFiles(directory=str(legacy_ui_dir)), name="web_ui")
    if has_vue_dist:
        assets_dir = vue_dist_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    def index():
        if has_vue_dist:
            return FileResponse(str(vue_dist_dir / "index.html"), headers=_no_cache_headers())
        return FileResponse(str(legacy_ui_dir / "index.html"), headers=_no_cache_headers())

    @app.get("/web_ui")
    @app.get("/web_ui/")
    @app.get("/web_ui/index.html")
    def legacy_redirect():
        if has_vue_dist:
            return RedirectResponse(url="/", status_code=307)
        return FileResponse(str(legacy_ui_dir / "index.html"), headers=_no_cache_headers())

    @app.post("/api/capture-and-detect")
    def capture_and_detect(req: CaptureRequest):
        if not detection_route_lock.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="Detection is already running. Please retry shortly.")
        t0 = time.perf_counter()
        try:
            t_capture0 = time.perf_counter()
            image_path = service.capture_image(req.camera_index)
            t_capture1 = time.perf_counter()
            result = service.analyze(image_path)
            t_analyze1 = time.perf_counter()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            detection_route_lock.release()

        if result["missing_status"] == "missing_detected":
            trigger_alarm()

        result["capture_image_url"] = path_to_url(result["capture_image_path"])
        result["annotated_image_url"] = path_to_url(result.get("annotated_image_path"))
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        result["processing_time_ms"] = elapsed_ms
        result["processing_time_sec"] = round(elapsed_ms / 1000.0, 3)
        result["pipeline_time_ms"] = {
            "capture": int((t_capture1 - t_capture0) * 1000),
            "analyze": int((t_analyze1 - t_capture1) * 1000),
            "total": elapsed_ms,
        }
        return result

    @app.post("/api/capture-and-detect-fast")
    def capture_and_detect_fast(req: CaptureRequest):
        if not detection_route_lock.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="Detection is already running. Please retry shortly.")
        _cleanup_async_jobs()
        t0 = time.perf_counter()
        try:
            t_capture0 = time.perf_counter()
            image_path = service.capture_image(req.camera_index)
            t_capture1 = time.perf_counter()
            # Stage-1: fast result without OCR for instant feedback.
            result = service.analyze(image_path, skip_ocr=True)
            t_analyze1 = time.perf_counter()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            detection_route_lock.release()

        result["capture_image_url"] = path_to_url(result["capture_image_path"])
        result["annotated_image_url"] = path_to_url(result.get("annotated_image_path"))
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        result["processing_time_ms"] = elapsed_ms
        result["processing_time_sec"] = round(elapsed_ms / 1000.0, 3)
        result["pipeline_time_ms"] = {
            "capture": int((t_capture1 - t_capture0) * 1000),
            "analyze_fast": int((t_analyze1 - t_capture1) * 1000),
            "total": elapsed_ms,
        }
        result["result_stage"] = "fast"

        result["async_task_id"] = ""
        return result

    @app.post("/api/detect-upload")
    def detect_upload(image: UploadFile = File(...)):
        if not detection_route_lock.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="Detection is already running. Please retry shortly.")
        t0 = time.perf_counter()
        try:
            t_save0 = time.perf_counter()
            image_path = service.save_uploaded_image(image)
            t_save1 = time.perf_counter()
            result = service.analyze(image_path)
            t_analyze1 = time.perf_counter()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            detection_route_lock.release()

        if result["missing_status"] == "missing_detected":
            trigger_alarm()

        result["capture_image_url"] = path_to_url(result["capture_image_path"])
        result["annotated_image_url"] = path_to_url(result.get("annotated_image_path"))
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        result["processing_time_ms"] = elapsed_ms
        result["processing_time_sec"] = round(elapsed_ms / 1000.0, 3)
        result["pipeline_time_ms"] = {
            "save_upload": int((t_save1 - t_save0) * 1000),
            "analyze": int((t_analyze1 - t_save1) * 1000),
            "total": elapsed_ms,
        }
        return result

    @app.post("/api/detect-upload-fast")
    def detect_upload_fast(image: UploadFile = File(...)):
        if not detection_route_lock.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="Detection is already running. Please retry shortly.")
        _cleanup_async_jobs()
        t0 = time.perf_counter()
        try:
            t_save0 = time.perf_counter()
            image_path = service.save_uploaded_image(image)
            t_save1 = time.perf_counter()
            # Stage-1: fast result without OCR for instant feedback.
            result = service.analyze(image_path, skip_ocr=True)
            t_analyze1 = time.perf_counter()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            detection_route_lock.release()

        result["capture_image_url"] = path_to_url(result["capture_image_path"])
        result["annotated_image_url"] = path_to_url(result.get("annotated_image_path"))
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        result["processing_time_ms"] = elapsed_ms
        result["processing_time_sec"] = round(elapsed_ms / 1000.0, 3)
        result["pipeline_time_ms"] = {
            "save_upload": int((t_save1 - t_save0) * 1000),
            "analyze_fast": int((t_analyze1 - t_save1) * 1000),
            "total": elapsed_ms,
        }
        result["result_stage"] = "fast"

        result["async_task_id"] = ""
        return result

    @app.get("/api/async-result/{task_id}")
    def get_async_result(task_id: str):
        _cleanup_async_jobs()
        with async_jobs_lock:
            item = async_jobs.get(task_id)
        if item is None:
            raise HTTPException(status_code=404, detail="task not found")
        status = str(item.get("status", "pending"))
        if status == "done":
            result = item.get("result") or {}
            if result.get("missing_status") == "missing_detected":
                trigger_alarm()
            return {"status": "done", "result": result}
        if status == "error":
            return {"status": "error", "error": str(item.get("error", "async task failed"))}
        return {"status": "pending"}

    @app.post("/api/live-preview")
    def live_preview(req: CaptureRequest):
        idx = service.camera_index_default if req.camera_index is None else int(req.camera_index)
        try:
            # Capture one frame from the realtime preview path.
            out = service.preview_dir / "live.jpg"
            try:
                img = service._get_preview_frame(req.camera_index)
            except Exception as e:
                cached = service._get_cached_live_frame(idx, max_age_sec=30.0)
                if cached is not None:
                    img = cached
                else:
                    latest = _latest_nonblocking_image()
                    if latest is not None:
                        return {
                            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                            "capture_image_path": str(latest),
                            "capture_image_url": path_to_url(str(latest)),
                            "stale": True,
                            "preview_status": f"live unavailable: {e}",
                        }
                    raise HTTPException(status_code=503, detail=f"live preview unavailable: {e}")
            ok = cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                raise RuntimeError("Preview write failed")
            image_path = out
        except HTTPException:
            # Preserve original status codes such as 503 (stream initializing/unavailable).
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "capture_image_path": str(image_path),
            "capture_image_url": path_to_url(str(image_path)),
        }

    @app.get("/api/live-stream")
    def live_stream(camera_index: Optional[int] = None):
        interval = 1.0 / max(service.preview_fps, 1.0)
        idx = int(service.camera_index_default if camera_index is None else camera_index)
        service.start_preview_worker()

        def gen():
            fail_count = 0
            while True:
                try:
                    if service.camera_live_provider == "hik_sdk":
                        frame = service._get_cached_live_frame(idx, max_age_sec=2.0)
                        if frame is None:
                            raise RuntimeError("live frame cache is not ready")
                    else:
                        frame = service._get_preview_frame(idx)
                    fail_count = 0
                    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if not ok:
                        time.sleep(interval)
                        continue
                    jpg = encoded.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Cache-Control: no-cache\r\n\r\n" + jpg + b"\r\n"
                    )
                except Exception:
                    fail_count += 1
                    # Keep stream alive: fallback to cached frame when realtime grab fails.
                    cached = service._get_cached_live_frame(idx, max_age_sec=30.0)
                    if cached is not None:
                        ok, encoded = cv2.imencode(".jpg", cached, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                        if ok:
                            jpg = encoded.tobytes()
                            yield (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                b"Cache-Control: no-cache\r\n\r\n" + jpg + b"\r\n"
                            )
                            time.sleep(interval)
                            continue
                    # No cache available yet: keep retrying for a while instead of immediate close.
                    if fail_count >= 20:
                        break
                    time.sleep(0.2)
                time.sleep(interval)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/live-status")
    def live_status(camera_index: Optional[int] = None):
        idx = int(service.camera_index_default if camera_index is None else camera_index)
        with service._live_frame_lock:
            ts = service._latest_live_frame_ts.get(idx)
        with service._stream_lock:
            last_error = service._stream_last_error
            last_error_ts = service._stream_last_error_ts
        age_ms = None
        if ts:
            age_ms = int((time.time() - ts) * 1000)
        return {
            "provider": service.camera_provider,
            "live_provider": service.camera_live_provider,
            "camera_index": idx,
            "stream_ready": service._hik_stream_ready(idx),
            "cached_frame": bool(service._get_cached_live_frame(idx, max_age_sec=2.0) is not None),
            "cached_age_ms": age_ms,
            "last_error": last_error,
            "last_error_time": dt.datetime.fromtimestamp(last_error_ts).isoformat(timespec="seconds")
            if last_error_ts
            else None,
        }

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "time": dt.datetime.now().isoformat(timespec="seconds")}

    @app.get("/api/camera-mode")
    def camera_mode():
        return {
            "provider": service.camera_provider,
            "live_provider": service.camera_live_provider,
            "use_snapshot_preview": bool(service.camera_live_provider == "hik_script"),
            "camera_index_default": service.camera_index_default,
        }

    @app.get("/api/camera-debug")
    def camera_debug():
        try:
            return service.debug_hik_devices()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Vue history fallback for non-API routes.
    @app.get("/{full_path:path}")
    def vue_fallback(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("runtime/"):
            raise HTTPException(status_code=404, detail="Not Found")
        if has_vue_dist:
            target = vue_dist_dir / full_path
            if target.is_file():
                return FileResponse(str(target))
            return FileResponse(str(vue_dist_dir / "index.html"), headers=_no_cache_headers())
        # Keep old static fallback behavior when Vue is not built yet.
        if not full_path:
            return FileResponse(str(legacy_ui_dir / "index.html"), headers=_no_cache_headers())
        target = legacy_ui_dir / full_path
        if target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(legacy_ui_dir / "index.html"), headers=_no_cache_headers())

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Web server for toolbox missing detection")
    parser.add_argument("--config", default="config/toolbox_workflow.json", help="Config path")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    app = create_app(_resolve_path(args.config))
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

