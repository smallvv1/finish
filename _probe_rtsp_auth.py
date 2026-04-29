import cv2

ips = ['192.168.1.64', '192.0.0.64', '10.0.0.64']
creds = [
    ('admin', 'admin'),
    ('admin', '12345'),
    ('admin', '123456'),
    ('admin', '12345678'),
    ('admin', '888888'),
]
paths = [
    '/Streaming/Channels/101',
    '/Streaming/Channels/102',
    '/h264/ch1/main/av_stream',
    '/h264/ch1/sub/av_stream',
]

found = False
for ip in ips:
    for user, pwd in creds:
        for path in paths:
            url = f'rtsp://{user}:{pwd}@{ip}:554{path}'
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                print('SUCCESS', url, frame.shape)
                found = True
                raise SystemExit(0)

print('NO_SUCCESS')
