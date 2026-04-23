import hailo_platform as hp
import numpy as np
import cv2
import os

HEF_PATH = '/home/pi/yolov11s.hef'
IMAGES_DIR = '/home/pi/test_images'
OUTPUT_DIR = '/home/pi/test_output'
CLASSES = ['bus', 'car', 'pickup', 'truck', 'van', 'motorcycle']
COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]
CONF_THRESHOLD = 0.3

os.makedirs(OUTPUT_DIR, exist_ok=True)

hef = hp.HEF(HEF_PATH)
target = hp.VDevice()
network_group = target.configure(hef)[0]
input_vstreams_params = hp.InputVStreamParams.make(network_group, format_type=hp.FormatType.UINT8)
output_vstreams_params = hp.OutputVStreamParams.make(network_group, format_type=hp.FormatType.FLOAT32)

images = [f for f in sorted(os.listdir(IMAGES_DIR)) if f.lower().endswith(('.jpg','.jpeg','.png'))]

with network_group.activate():
    with hp.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as pipeline:
        for img_file in images:
            img = cv2.imread(os.path.join(IMAGES_DIR, img_file))
            orig_h, orig_w = img.shape[:2]
            resized = cv2.resize(img, (640, 640))
            frame = np.expand_dims(resized.astype(np.uint8), axis=0)
            output = pipeline.infer({'yolov11s/input_layer1': frame})
            detections = output['yolov11s/yolov8_nms_postprocess'][0]
            for cls_id, dets in enumerate(detections):
                if len(dets) == 0:
                    continue
                for det in dets:
                    conf = float(det[4])
                    threshold = 0.15 if cls_id == 5 else CONF_THRESHOLD
                    if conf < threshold:
                        continue
                    y1 = int(det[0] * orig_h)
                    x1 = int(det[1] * orig_w)
                    y2 = int(det[2] * orig_h)
                    x2 = int(det[3] * orig_w)
                    color = COLORS[cls_id]
                    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                    label = CLASSES[cls_id] + ' ' + str(round(conf, 2))
                    cv2.putText(img, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    print(img_file + ': ' + label)
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_file), img)

print('Test completato. Risultati in: ' + OUTPUT_DIR)
