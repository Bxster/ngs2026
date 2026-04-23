import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
from picamera2.devices.imx500.postprocess import COCODrawer

RPK_PATH = "/home/pi/best_nano_v2_merged.rpk"
VIDEO_PATH = "/home/pi/Downloads/YTDown.com_YouTube_Traffic-Intersection-Camera-Car-Narrowly_Media_FjTs7d3JFoE_001_240p.mp4"
CLASSES = ["bus", "car", "pickup", "truck", "van", "motorcycle"]
CONF_THRESHOLD = 0.3

imx500 = IMX500(RPK_PATH)
intrinsics = imx500.network_intrinsics

cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/home/pi/test_nano_output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 5 != 0:
        out.write(frame)
        continue

    resized = cv2.resize(frame, (320, 320))
    outputs = imx500.run_inference(resized)

    if outputs is not None:
        boxes, scores, classes, num = outputs
        for i in range(int(num[0])):
            if scores[0][i] < CONF_THRESHOLD:
                continue
            cls_id = int(classes[0][i])
            if cls_id >= len(CLASSES):
                continue
            h, w = frame.shape[:2]
            y1 = int(boxes[0][i][0] * h)
            x1 = int(boxes[0][i][1] * w)
            y2 = int(boxes[0][i][2] * h)
            x2 = int(boxes[0][i][3] * w)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            label = CLASSES[cls_id] + ' ' + str(round(float(scores[0][i]), 2))
            cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            print(f"Frame {frame_count}: {label}")

    out.write(frame)

cap.release()
out.release()
print("Output salvato in: /home/pi/test_nano_output.mp4")