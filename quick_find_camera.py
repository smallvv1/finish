#!/usr/bin/env python3
"""快速找海康摄像头 - 直接扫描网线连接段"""
import socket
import threading
import sys

def scan_ip(ip, port=554, timeout=0.3):
    """测试单个IP的RTSP端口"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        result = s.connect_ex((ip, port))
        s.close()
        return result == 0
    except:
        return False

def main():
    print("[*] 扫描 169.254.x.x 网段中的海康摄像头...")
    print("[*] 这可能需要几分钟...")
    
    found = []
    found_lock = threading.Lock()
    
    def worker(start, end):
        for i in range(start, end):
            ip = f"169.254.{i//256}.{i%256}"
            if scan_ip(ip, 554):
                with found_lock:
                    found.append(ip)
                    print(f"[+] 找到: {ip}")
            # 进度显示
            if i % 100 == 0:
                print(f"    进度: 169.254.{i//256}.{i%256}", end='\r')
    
    # 多线程扫描
    threads = []
    step = 65536 // 8  # 8个线程
    for i in range(8):
        start = i * step
        end = (i+1) * step if i < 7 else 65536
        t = threading.Thread(target=worker, args=(start, end))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    print("\n" + "="*60)
    if found:
        print(f"[✓] 找到 {len(found)} 个摄像头:")
        for ip in sorted(found):
            print(f"    {ip}")
            print(f"\n    连接命令:")
            print(f"    python hikvision_ocr.py --ip {ip} --username admin --password <密码> --lang ch")
    else:
        print("[!] 未找到摄像头，请检查:")
        print("    1. 摄像头电源是否打开")
        print("    2. 网线是否正确连接")
        print("    3. 是否需要配置摄像头的IP地址")

if __name__ == "__main__":
    main()
