import hailo_platform as hp
import numpy as np
import cv2
import os

PLATE_HEF = '/home/pi/Downloads/tiny_yolov4_license_plates.hef'
LPR_HEF = '/home/pi/Downloads/lprnet/lprnet.hef'
IMAGES_DIR = '/home/pi/test_images'
OUTPUT_DIR = '/home/pi/test_lpr_output'
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4
INPUT_SIZE = 416

ANCHORS_13 = [(81,82), (135,169), (344,319)]
ANCHORS_26 = [(23,27), (37,58), (81,82)]

CHARS = '0123456789'

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def decode_lpr(output):
    logits = list(output)[0]
    if hasattr(logits, 'shape') and len(logits.shape) == 3:
        indices = np.argmax(logits[0], axis=1)
    else:
        indices = np.argmax(logits, axis=1)
    chars = []
    prev = -1
    for idx in np.array(indices).flatten().tolist():
        if idx != prev and idx < len(CHARS):
            chars.append(CHARS[idx])
        prev = idx
    return ''.join(chars)

plate_hef = hp.HEF(PLATE_HEF)
lpr_hef = hp.HEF(LPR_HEF)
target = hp.VDevice()

plate_ng = target.configure(plate_hef)[0]
lpr_ng = target.configure(lpr_hef)[0]

plate_in_params = hp.InputVStreamParams.make(plate_ng, format_type=hp.FormatType.UINT8)
plate_out_params = hp.OutputVStreamParams.make(plate_ng, format_type=hp.FormatType.FLOAT32)
lpr_in_params = hp.InputVStreamParams.make(lpr_ng, format_type=hp.FormatType.UINT8)
lpr_out_params = hp.OutputVStreamParams.make(lpr_ng, format_type=hp.FormatType.FLOAT32)

plate_input_info = plate_hef.get_input_vstream_infos()[0]
lpr_input_info = lpr_hef.get_input_vstream_infos()[0]

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
                crop_resized = cv2.resize(crop, (300, 75))
                all_crops.append((img_file, i, conf, crop_resized))

with lpr_ng.activate():
    with hp.InferVStreams(lpr_ng, lpr_in_params, lpr_out_params) as lpr_pipeline:
        for img_file, i, conf, crop_resized in all_crops:
            lpr_frame = np.expand_dims(crop_resized.astype(np.uint8), axis=0)
            lpr_output = lpr_pipeline.infer({lpr_input_info.name: lpr_frame})
            text = decode_lpr(list(lpr_output.values()))
            print(img_file + ' plate ' + str(round(conf,2)) + ': ' + text)
            cv2.putText(crop_resized, text, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_file.replace('.', f'_plate{i}.')), crop_resized)

print('Done. Output in: ' + OUTPUT_DIR)
