import socket
import time

ip = "169.254.157.81"
# 尝试更多的RTSP端口和HTTP管理端口
ports = [554, 8554, 1554, 80, 8000, 8080, 8088, 37777, 37778]

print(f"[*] 扫描 {ip} 的开放端口...")
open_ports = []
for port in ports:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.8)
    result = s.connect_ex((ip, port))
    s.close()
    if result == 0:
        print(f"[+] 端口 {port} 开放")
        open_ports.append(port)
    else:
        print(f"[-] 端口 {port} 关闭")

if open_ports:
    print(f"\n找到开放端口: {open_ports}")
else:
    print("\n未找到开放端口")
