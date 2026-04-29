import cv2

ips = ['192.168.1.64', '192.0.0.64', '10.0.0.64']
urls = []
for ip in ips:
    urls.append(f'rtsp://admin:12345@{ip}:554/Streaming/Channels/101')
    urls.append(f'rtsp://admin:12345@{ip}:554/Streaming/Channels/102')

for url in urls:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    ok, frame = cap.read()
    print(url, 'OK' if (ok and frame is not None) else 'FAIL')
    cap.release()
