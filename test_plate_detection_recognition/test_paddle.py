import hailo_platform as hp
import numpy as np
import cv2
import os
import easyocr

PLATE_HEF = '/home/pi/Downloads/tiny_yolov4_license_plates.hef'
IMAGES_DIR = '/home/pi/test_images'
OUTPUT_DIR = '/home/pi/test_paddle_output'
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4
INPUT_SIZE = 416

ANCHORS_13 = [(81,82), (135,169), (344,319)]
ANCHORS_26 = [(23,27), (37,58), (81,82)]

os.makedirs(OUTPUT_DIR, exist_ok=True)

reader = easyocr.Reader(['en'], gpu=False)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def decode_predictions(raw, anchors, input_size, orig_h, orig_w, conf_thresh):
    raw = raw[0]
    grid_h, grid_w = raw.shape[0], raw.shape[1]
    detections = []
    for row in range(grid_h):
        for col in range(grid_w):
            for a, (aw, ah) in enumerate(anchors):
                offset = a * 6
                tx, ty, tw, th = raw[row, col, offset:offset+4]
                obj = sigmoid(raw[row, col, offset+4])
                cls = sigmoid(raw[row, col, offset+5])
                conf = obj * cls
                if conf < conf_thresh:
                    continue
                bx = (sigmoid(tx) + col) / grid_w
                by = (sigmoid(ty) + row) / grid_h
                bw = (aw * np.exp(tw)) / input_size
                bh = (ah * np.exp(th)) / input_size
                x1 = int((bx - bw/2) * orig_w)
                y1 = int((by - bh/2) * orig_h)
                x2 = int((bx + bw/2) * orig_w)
                y2 = int((by + bh/2) * orig_h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                detections.append([x1, y1, x2, y2, float(conf)])
    return detections

def nms(detections, iou_thresh):
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [d for d in detections if iou(best, d) < iou_thresh]
    return keep

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

plate_hef = hp.HEF(PLATE_HEF)
target = hp.VDevice()
plate_ng = target.configure(plate_hef)[0]
plate_in_params = hp.InputVStreamParams.make(plate_ng, format_type=hp.FormatType.UINT8)
plate_out_params = hp.OutputVStreamParams.make(plate_ng, format_type=hp.FormatType.FLOAT32)
plate_input_info = plate_hef.get_input_vstream_infos()[0]

images = [f for f in sorted(os.listdir(IMAGES_DIR)) if f.lower().endswith(('.jpg','.jpeg','.png'))]

all_crops = []
with plate_ng.activate():
    with hp.InferVStreams(plate_ng, plate_in_params, plate_out_params) as plate_pipeline:
        for img_file in images:
            img = cv2.imread(os.path.join(IMAGES_DIR, img_file))
            orig_h, orig_w = img.shape[:2]
            resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
            frame = np.expand_dims(resized.astype(np.uint8), axis=0)
            output = plate_pipeline.infer({plate_input_info.name: frame})
            raw13 = output['tiny_yolov4_license_plates/conv19']
            raw26 = output['tiny_yolov4_license_plates/conv21']
            dets = decode_predictions(raw13, ANCHORS_13, INPUT_SIZE, orig_h, orig_w, CONF_THRESHOLD)
            dets += decode_predictions(raw26, ANCHORS_26, INPUT_SIZE, orig_h, orig_w, CONF_THRESHOLD)
            dets = nms(dets, IOU_THRESHOLD)
            for i, det in enumerate(dets):
                x1, y1, x2, y2, conf = det
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                all_crops.append((img_file, i, conf, crop))

for img_file, i, conf, crop in all_crops:
    result = reader.readtext(crop)
    text = ' '.join([r[1] for r in result]) if result else ''
    print(img_file + ' plate ' + str(round(conf,2)) + ': ' + text)
    cv2.putText(crop, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, img_file.replace('.', f'_plate{i}.')), crop)

print('Done. Output in: ' + OUTPUT_DIR)
