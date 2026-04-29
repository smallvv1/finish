import socket
import sys

# 海康摄像头直连网线通常在169.254段的默认IP
candidates = []

# 常见的海康摄像头默认IP（单网线直连模式）
test_ips = [
    "169.254.1.1",
    "169.254.1.2", 
    "169.254.52.1",
    "169.254.52.2",
    "169.254.100.1",
    "169.254.0.1",
]

for ip in test_ips:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        result = s.connect_ex((ip, 554))  # RTSP端口
        s.close()
        if result == 0:
            candidates.append(ip)
            print(f"[OK] {ip} 端口554开放")
        else:
            # 尝试HTTP端口8000
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            result = s.connect_ex((ip, 8000))
            s.close()
            if result == 0:
                candidates.append(ip)
                print(f"[OK] {ip} 端口8000开放")
    except Exception as e:
        pass

if candidates:
    print(f"\n找到 {len(candidates)} 个候选摄像头:")
    for ip in candidates:
        print(f"  {ip}")
else:
    print("未找到")
