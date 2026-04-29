import cv2

ip = '169.254.157.82'
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

ok_any = False
for user, pwd in creds:
    for path in paths:
        url = f'rtsp://{user}:{pwd}@{ip}:554{path}'
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        ok, frame = cap.read()
        cap.release()
        print(('OK' if (ok and frame is not None) else 'FAIL'), user, pwd, path)
        if ok and frame is not None:
            print('SUCCESS_URL', url, frame.shape)
            ok_any = True
            raise SystemExit(0)

if not ok_any:
    print('NO_RTSP_SUCCESS')
