import socket
import struct

# 海康威视设备发现协议 (ISAPI Discovery)
# 发送广播包，让摄像头响应
ip = "169.254.157.81"

print(f"[*] 尝试 UDP 发现协议与摄像头通信...")

# HIKVISION ISAPI Discovery 包
discovery_packet = (
    b"\x48\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
)

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2)
    sock.sendto(discovery_packet, (ip, 37777))
    data, _ = sock.recvfrom(1024)
    print(f"[+] 收到响应: {len(data)} bytes")
    print(f"    响应内容: {data[:100]}")
except socket.timeout:
    print(f"[-] UDP 端口 37777 无响应")
except Exception as e:
    print(f"[-] 错误: {e}")
finally:
    sock.close()

# 另一个发现方式: ONVIF Discovery (WS-Discovery)
print(f"\n[*] 尝试 ONVIF 发现...")
onvif_ip = ip
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1)
    sock.sendto(b"test", (onvif_ip, 3702))
    print(f"[-] ONVIF 端口无响应")
except:
    print(f"[-] ONVIF 不可用")
