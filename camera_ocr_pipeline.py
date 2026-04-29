#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
海康相机 + 文字识别 完整管道
运行流程：
1. test_hk_opecv.py - 调用相机拍摄图像并保存为 capture_clear_*.jpg
2. wenzi.py - 对拍摄结果进行目标检测和文字识别
"""

import os
import sys
import subprocess
import argparse
import importlib.util
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.absolute()

def load_module_from_file(module_name, file_path):
    """动态加载 Python 模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_camera_capture(mode="fast", count=1):
    """
    运行相机拍摄
    Args:
        mode: 拍摄模式 (ultrafast/fast/bright/quality)
        count: 拍摄次数
    """
    print("=" * 60)
    print("[1/2] 开始调用海康相机拍摄...")
    print("=" * 60)
    
    script_path = CURRENT_DIR / "test_hk_opecv.py"
    cmd = [sys.executable, str(script_path), "--mode", mode, "--count", str(count)]
    
    try:
        result = subprocess.run(cmd, cwd=str(CURRENT_DIR), check=True)
        print(f"✓ 拍摄完成 (exit code: {result.returncode})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 拍摄失败 (exit code: {e.returncode})")
        return False
    except Exception as e:
        print(f"✗ 执行相机脚本异常: {e}")
        return False


def run_ocr_recognition():
    """
    运行文字识别（直接调用 wenzi 模块中的函数）
    """
    print("\n" + "=" * 60)
    print("[2/2] 开始进行目标检测和文字识别...")
    print("=" * 60)
    
    try:
        # 直接加载并调用 wenzi.py 中的函数
        wenzi_path = CURRENT_DIR / "wenzi.py"
        wenzi = load_module_from_file("wenzi", str(wenzi_path))
        
        # 调用 process_images_from_camera 函数
        if hasattr(wenzi, 'process_images_from_camera'):
            wenzi.process_images_from_camera()
            print(f"✓ 识别完成")
            return True
        else:
            print(f"✗ 未找到 process_images_from_camera 函数")
            return False
    except Exception as e:
        print(f"✗ 识别过程异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="海康相机拍摄 + 文字识别完整管道",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 快速模式拍摄1张，进行文字识别
  python camera_ocr_pipeline.py --mode fast --count 1

  # 高质量模式拍摄3张
  python camera_ocr_pipeline.py --mode quality --count 3
        """
    )
    parser.add_argument("--mode", choices=["ultrafast", "fast", "bright", "quality"], 
                        default="fast", help="拍摄模式")
    parser.add_argument("--count", type=int, default=1, help="拍摄张数")
    parser.add_argument("--skip-capture", action="store_true", 
                        help="跳过拍摄步骤，仅进行识别")
    parser.add_argument("--skip-ocr", action="store_true", 
                        help="跳过识别步骤，仅进行拍摄")
    
    args = parser.parse_args()
    
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + "海康相机 + 文字识别完整管道".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    success = True
    
    # 步骤1：拍摄
    if not args.skip_capture:
        success = run_camera_capture(mode=args.mode, count=args.count)
        if not success:
            print("\n[ERROR] 拍摄失败，停止处理")
            sys.exit(1)
    
    # 步骤2：识别
    if not args.skip_ocr:
        success = run_ocr_recognition()
        if not success:
            print("\n[ERROR] 识别失败")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ 完整管道执行完成！")
    print(f"✓ 识别结果已保存到: {CURRENT_DIR / 'inference_log.txt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
