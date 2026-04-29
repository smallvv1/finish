#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
海康相机实时预览 + 按需拍照检测
功能：
- 实时显示摄像头视频流
- 按 SPACE 键拍照并保存
- 按 O 键对最后拍摄的图像进行 OCR 文字识别
- 按 Q 键退出
"""

import os
import sys
import time
import datetime
import argparse
import re
import cv2
import numpy as np
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.absolute()

# 添加 MVS SDK 路径
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from MvCameraControl_class import *
from CameraParams_header import *

try:
    from paddleocr import PaddleOCR
    from ultralytics import YOLO
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[WARN] PaddleOCR 或 YOLO 未安装，将仅提供预览功能")


TH_CONFUSION_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "H": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}

AE_PROFILE_TARGET = {
    "day": 105,
    "night": 155,
}


def _correct_th_token(token):
    token = token.strip().upper()
    if len(token) < 4 or not token.startswith("TH"):
        return ""

    suffix = token[2:4]
    corrected_digits = []
    for char in suffix:
        if char.isdigit():
            corrected_digits.append(char)
            continue
        mapped = TH_CONFUSION_TO_DIGIT.get(char)
        if mapped is None:
            return ""
        corrected_digits.append(mapped)
    return f"TH{''.join(corrected_digits)}"


def extract_best_th_code(raw_text):
    if not raw_text:
        return ""

    merged = raw_text.upper().replace(" ", "")
    candidates = re.findall(r"TH[A-Z0-9]{2}", merged)
    for candidate in candidates:
        corrected = _correct_th_token(candidate)
        if corrected:
            return corrected
    return ""


def normalize_th_codes_in_text(raw_text):
    if not raw_text:
        return ""

    normalized = re.sub(r"\s+", " ", raw_text.strip().upper())

    def _replace_token(match):
        token = match.group(0)
        corrected = _correct_th_token(token)
        return corrected if corrected else token

    return re.sub(r"TH[A-Z0-9]{2}", _replace_token, normalized)


class CameraLiveOCR:
    def __init__(
        self,
        exposure_time=10000,
        gain=7.0,
        auto_exposure=True,
        auto_gain=True,
        ae_target_brightness=120,
        ae_settle_frames=6,
    ):
        self.camera = None
        self.exposure_time = exposure_time
        self.gain = gain
        self.auto_exposure = auto_exposure
        self.auto_gain = auto_gain
        self.ae_target_brightness = int(max(30, min(220, int(ae_target_brightness))))
        self.ae_settle_frames = max(0, int(ae_settle_frames))
        self.is_running = False
        self.last_capture_path = None
        self.frame_count = 0
        
        if OCR_AVAILABLE:
            self.ocr = PaddleOCR(use_textline_orientation=True, lang="en", enable_mkldnn=False)
            self.model = YOLO(r"D:\1\EzYOLO-main\runs\detect\runs\train\exp_2\weights\best.pt")
        
    def init_camera(self):
        """初始化海康相机"""
        print("[INFO] 初始化摄像头...")
        
        if hasattr(MvCamera, "MV_CC_Initialize"):
            ret = MvCamera.MV_CC_Initialize()
            if ret != 0:
                print(f"[ERROR] 初始化SDK失败，错误码: {ret}")
                return False
        
        # 枚举设备
        deviceList = MV_CC_DEVICE_INFO_LIST()
        n_layer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
        gntl_cameralink_device = globals().get("MV_GENTL_CAMERALINK_DEVICE")
        if gntl_cameralink_device is not None:
            n_layer_type |= gntl_cameralink_device
        
        ret = MvCamera.MV_CC_EnumDevices(n_layer_type, deviceList)
        if ret != 0 or deviceList.nDeviceNum == 0:
            print(f"[ERROR] 枚举设备失败或未找到设备")
            return False
        
        print(f"[INFO] 找到 {deviceList.nDeviceNum} 台设备")
        
        stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        
        self.camera = MvCamera()
        ret = self.camera.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print(f"[ERROR] 创建句柄失败，错误码: {ret}")
            return False
        
        # 打开设备
        for try_index in range(5):
            ret = self.camera.MV_CC_OpenDevice()
            if ret == 0:
                break
            time.sleep(0.4)
        
        if ret != 0:
            print(f"[ERROR] 打开设备失败，错误码: {ret}")
            print("[HINT] 请关闭 MVS 客户端或其他占用相机的软件")
            return False
        
        # 配置相机
        self._configure_camera()
        print("[INFO] 摄像头初始化成功 ✓")
        return True
    
    def _configure_camera(self):
        """配置相机参数"""
        # 设置分辨率
        stParam = MVCC_INTVALUE()
        self.camera.MV_CC_GetIntValue("Width", stParam)
        width = stParam.nCurValue
        self.camera.MV_CC_GetIntValue("Height", stParam)
        height = stParam.nCurValue
        print(f"[INFO] 分辨率: {width}x{height}")
        
        # 曝光模式：自动或手动
        if self.auto_exposure:
            ret_exp = self.camera.MV_CC_SetEnumValue("ExposureAuto", 2)
            if ret_exp != 0:
                print(f"[WARN] 开启自动曝光失败，错误码: {ret_exp}，将回退为手动曝光")
                self.camera.MV_CC_SetEnumValue("ExposureAuto", 0)
                self.camera.MV_CC_SetFloatValue("ExposureTime", self.exposure_time)
            else:
                print("[INFO] 自动曝光: 已开启")
                self._apply_auto_exposure_target()
        else:
            self.camera.MV_CC_SetEnumValue("ExposureAuto", 0)
            ret_time = self.camera.MV_CC_SetFloatValue("ExposureTime", self.exposure_time)
            if ret_time != 0:
                print(f"[WARN] 设置手动曝光时间失败，错误码: {ret_time}")
            else:
                print(f"[INFO] 手动曝光: {self.exposure_time} us")

        # 增益模式：自动或手动
        if self.auto_gain:
            ret_gain_auto = self.camera.MV_CC_SetEnumValue("GainAuto", 2)
            if ret_gain_auto != 0:
                print(f"[WARN] 开启自动增益失败，错误码: {ret_gain_auto}，将回退为手动增益")
                self.camera.MV_CC_SetEnumValue("GainAuto", 0)
                self.camera.MV_CC_SetFloatValue("Gain", self.gain)
            else:
                print("[INFO] 自动增益: 已开启")
        else:
            self.camera.MV_CC_SetEnumValue("GainAuto", 0)
            ret_gain = self.camera.MV_CC_SetFloatValue("Gain", self.gain)
            if ret_gain != 0:
                print(f"[WARN] 设置手动增益失败，错误码: {ret_gain}")
            else:
                print(f"[INFO] 手动增益: {self.gain}")
        
        # 开始抓图
        ret = self.camera.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"[ERROR] 开始抓图失败，错误码: {ret}")
            return

        if (self.auto_exposure or self.auto_gain) and self.ae_settle_frames > 0:
            print(f"[INFO] 等待自动曝光/增益稳定: {self.ae_settle_frames} 帧...")
            for _ in range(self.ae_settle_frames):
                self.get_frame()

    def _apply_auto_exposure_target(self):
        """设置自动曝光目标亮度（不同型号节点名可能不同）"""
        target = int(self.ae_target_brightness)
        candidates = [
            "AutoExposureTargetBrightness",
            "AutoExposureTargetGrayValue",
            "TargetBrightness",
            "TargetGrayValue",
        ]

        for node_name in candidates:
            ret = self.camera.MV_CC_SetIntValue(node_name, target)
            if ret == 0:
                print(f"[INFO] 自动曝光目标亮度: {target} (节点: {node_name})")
                return True

        print(f"[WARN] 当前相机不支持设置自动曝光目标亮度（期望值: {target}）")
        return False
    
    def get_frame(self):
        """获取一帧图像"""
        stParam = MVCC_INTVALUE()
        self.camera.MV_CC_GetIntValue("PayloadSize", stParam)
        payload_size = stParam.nCurValue
        
        data_buf = (c_ubyte * payload_size)()
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        
        ret = self.camera.MV_CC_GetOneFrameTimeout(
            byref(data_buf),
            payload_size,
            stFrameInfo,
            1000
        )
        
        if ret != 0:
            return None
        
        frame = np.frombuffer(data_buf, dtype=np.uint8, count=stFrameInfo.nFrameLen)
        actual_width = stFrameInfo.nWidth
        actual_height = stFrameInfo.nHeight
        pixel_type = stFrameInfo.enPixelType
        
        # 像素格式转换
        if pixel_type == 17301505:  # Mono8
            expected_size = actual_width * actual_height
            if len(frame) != expected_size:
                return None
            frame = frame.reshape((actual_height, actual_width))
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif pixel_type == 17301513:  # BayerRG8
            expected_size = actual_width * actual_height
            if len(frame) != expected_size:
                return None
            frame = frame.reshape((actual_height, actual_width))
            frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
        elif pixel_type == 17301514:  # BayerGB8
            expected_size = actual_width * actual_height
            if len(frame) != expected_size:
                return None
            frame = frame.reshape((actual_height, actual_width))
            frame = cv2.cvtColor(frame, cv2.COLOR_BayerGB2BGR)
        elif len(frame) == actual_width * actual_height * 3:
            frame = frame.reshape((actual_height, actual_width, 3))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            return None
        
        return frame
    
    def capture_frame(self):
        """捕获当前帧并保存"""
        frame = self.get_frame()
        if frame is None:
            print("[ERROR] 获取帧失败")
            return None
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"capture_clear_{timestamp}.jpg"
        filepath = CURRENT_DIR / filename
        
        # 确保文件被保存
        success = cv2.imwrite(str(filepath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        if success:
            self.last_capture_path = str(filepath)
            print(f"[✓] 已保存: {filename} ({filepath})")
            # 验证文件是否存在
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"[✓] 文件大小: {file_size} bytes")
            return frame
        else:
            print(f"[ERROR] 保存失败: {filepath}")
            return None
    
    def perform_ocr_on_last_capture(self):
        """对最后拍摄的图像进行 OCR 识别"""
        if not OCR_AVAILABLE:
            print("[ERROR] OCR 功能未可用（缺少依赖）")
            return
        
        if self.last_capture_path is None:
            print("[WARN] 没有拍摄过图像")
            return
        
        if not os.path.exists(self.last_capture_path):
            print(f"[ERROR] 文件不存在: {self.last_capture_path}")
            return
        
        print(f"\n[OCR] 开始识别: {os.path.basename(self.last_capture_path)}")
        print(f"[OCR] 文件路径: {self.last_capture_path}")
        
        try:
            img = cv2.imread(self.last_capture_path)
            if img is None:
                print(f"[ERROR] 无法读取图像")
                return
            
            print(f"[OCR] 图像尺寸: {img.shape}")
            print(f"[OCR] 运行目标检测模型...")
            
            results = self.model(self.last_capture_path, conf=0.4, verbose=False)
            
            print(f"[OCR] 检测到 {len(results[0].boxes)} 个目标")
            
            output_log = CURRENT_DIR / "ocr_results.txt"
            with open(output_log, "a", encoding="utf-8") as log_file:
                log_file.write(f"\n{'='*60}\n")
                log_file.write(f"文件: {os.path.basename(self.last_capture_path)}\n")
                log_file.write(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"{'='*60}\n")
                
                detection_count = 0
                for idx, box in enumerate(results[0].boxes):
                    try:
                        class_id = int(box.cls.item()) if box.cls is not None else -1
                        cls_name = results[0].names.get(class_id, str(class_id)) if class_id >= 0 else "unknown"
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img.shape[1], x2)
                        y2 = min(img.shape[0], y2)
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        detection_count += 1
                        crop = img[y1:y2, x1:x2]
                        
                        print(f"[OCR] 检测框 #{detection_count}: {cls_name} ({x1},{y1})-({x2},{y2})")
                        
                        # OCR 识别
                        print(f"[OCR] 正在识别文字...")
                        res = list(self.ocr.predict(crop))
                        text = ""
                        if res:
                            first = res[0]
                            if isinstance(first, dict) and first.get("rec_texts"):
                                text = " ".join(first["rec_texts"])
                            elif hasattr(first, "rec_texts") and getattr(first, "rec_texts"):
                                text = " ".join(first.rec_texts)
                            elif isinstance(first, list):
                                text = " ".join([line[1][0] for line in first if len(line) > 1 and len(line[1]) > 0])

                        normalized_text = normalize_th_codes_in_text(text)
                        corrected_th = extract_best_th_code(normalized_text)
                        if corrected_th:
                            result_text = f"[{cls_name} #{detection_count}] 识别文字: {normalized_text} | TH编码: {corrected_th}"
                        else:
                            result_text = f"[{cls_name} #{detection_count}] 识别文字: {normalized_text}"
                        print(f"[OCR] {result_text}")
                        log_file.write(result_text + "\n")
                    
                    except Exception as e:
                        print(f"[ERROR] 处理检测框失败: {e}")
                        continue
                
                if detection_count == 0:
                    print("[INFO] 未检测到任何目标")
                    log_file.write("未检测到任何目标\n")
            
            print(f"[✓] 识别完成，结果已保存到 ocr_results.txt\n")
        
        except Exception as e:
            print(f"[ERROR] OCR 识别异常: {e}")
            import traceback
            traceback.print_exc()
    
    def run_live_preview(self):
        """运行实时预览"""
        if not self.init_camera():
            return
        
        self.is_running = True
        cv2.namedWindow("Camera Live Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Live Preview", 960, 720)
        
        print("\n" + "="*60)
        print("实时摄像头预览已启动")
        print("="*60)
        print("按键说明:")
        print("  SPACE - 拍照、检测并保存")
        print("  Q     - 退出")
        print("="*60 + "\n")
        
        try:
            while self.is_running:
                frame = self.get_frame()
                if frame is None:
                    print("[WARN] 获取帧失败，重试...")
                    continue
                
                self.frame_count += 1
                
                # 在帧上显示信息
                info_text = f"Frame: {self.frame_count} | Press SPACE to capture & detect, Q to quit"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                if self.last_capture_path:
                    status_text = f"Last: {os.path.basename(self.last_capture_path)}"
                    cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 255), 1)
                
                cv2.imshow("Camera Live Preview", frame)
                
                # 使用 waitKey 获取按键，timeout 设置为 30ms
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:  # 27 是 ESC 键
                    print("\n[INFO] 按下退出键，正在关闭...")
                    self.is_running = False
                elif key == ord(' '):  # SPACE 键
                    print("\n[ACTION] 拍照中...")
                    captured_frame = self.capture_frame()
                    if captured_frame is not None:
                        print("[ACTION] 正在进行文字识别...")
                        self.perform_ocr_on_last_capture()
        
        except KeyboardInterrupt:
            print("\n[INFO] 收到中断信号，正在关闭...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        cv2.destroyAllWindows()
        
        if self.camera:
            self.camera.MV_CC_StopGrabbing()
            self.camera.MV_CC_CloseDevice()
            self.camera.MV_CC_DestroyHandle()
        
        if hasattr(MvCamera, "MV_CC_Finalize"):
            MvCamera.MV_CC_Finalize()
        
        print("[INFO] 摄像头已关闭")


def main():
    parser = argparse.ArgumentParser(
        description="海康相机实时预览 + 按需 OCR 识别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python camera_live_ocr.py
  python camera_live_ocr.py --exposure 15000 --gain 8.0
    python camera_live_ocr.py --ae-profile day
    python camera_live_ocr.py --ae-profile night
    python camera_live_ocr.py --ae-profile custom --ae-target-brightness 135
        """
    )
    parser.add_argument("--exposure", type=float, default=10000, help="曝光时间（微秒）")
    parser.add_argument("--gain", type=float, default=7.0, help="增益值")
    parser.add_argument("--auto-exposure", action="store_true", help="开启自动曝光（默认开启）")
    parser.add_argument("--manual-exposure", action="store_true", help="关闭自动曝光，使用 --exposure")
    parser.add_argument("--auto-gain", action="store_true", help="开启自动增益（默认开启）")
    parser.add_argument("--manual-gain", action="store_true", help="关闭自动增益，使用 --gain")
    parser.add_argument("--ae-profile", choices=["day", "night", "custom"], default="day",
                        help="自动曝光亮度档位：day(默认)/night/custom")
    parser.add_argument("--ae-target-brightness", type=int, default=120,
                        help="自动曝光目标亮度(仅 custom 档位生效，建议 90~170)")
    parser.add_argument("--ae-settle-frames", type=int, default=6,
                        help="自动曝光/增益稳定帧数，默认 6")
    
    args = parser.parse_args()
    
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + "海康相机实时预览 + 文字识别".center(58) + "║")
    print("╚" + "="*58 + "╝")
    print()
    
    auto_exposure = True
    if args.manual_exposure:
        auto_exposure = False
    elif args.auto_exposure:
        auto_exposure = True

    auto_gain = True
    if args.manual_gain:
        auto_gain = False
    elif args.auto_gain:
        auto_gain = True

    if args.ae_profile == "custom":
        ae_target_brightness = int(args.ae_target_brightness)
    else:
        ae_target_brightness = AE_PROFILE_TARGET[args.ae_profile]

    camera = CameraLiveOCR(
        exposure_time=args.exposure,
        gain=args.gain,
        auto_exposure=auto_exposure,
        auto_gain=auto_gain,
        ae_target_brightness=ae_target_brightness,
        ae_settle_frames=args.ae_settle_frames,
    )
    camera.run_live_preview()


if __name__ == "__main__":
    main()
