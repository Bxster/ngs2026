import hailo_platform as hp
import numpy as np
import cv2
import os
import json
import easyocr
from collections import defaultdict

VIDEO_PATH = '/home/pi/Downloads/video1.mp4'
OUTPUT_DIR = '/home/pi/Desktop/test_video_pipeline'
SMALL_HEF = '/home/pi/yolov11s.hef'
PLATE_HEF = '/home/pi/Downloads/tiny_yolov4_license_plates.hef'

VEHICLE_CLASSES = ['bus', 'car', 'pickup', 'truck', 'van', 'motorcycle']
VEHICLE_COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]
PLATE_COLOR = (0, 200, 255)

VEHICLE_CONF = 0.3
PLATE_CONF = 0.3
IOU_THRESH = 0.4
PLATE_INPUT_SIZE = 416

ANCHORS_13 = [(81,82), (135,169), (344,319)]
ANCHORS_26 = [(23,27), (37,58), (81,82)]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def decode_yolov4(raw, anchors, input_size, orig_h, orig_w, conf_thresh):
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

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

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

print('Loading EasyOCR...')
reader = easyocr.Reader(['en'], gpu=False)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Video: {total_frames} frames, {fps:.1f} fps, {width}x{height}')

# PASSATA 1: leggi tutti i frame in memoria
print('Reading all frames...')
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print(f'Loaded {len(frames)} frames')

# PASSATA 2: vehicle detection con small
print('Phase 1: vehicle detection (small)...')
small_hef = hp.HEF(SMALL_HEF)
target = hp.VDevice()
small_ng = target.configure(small_hef)[0]
small_in_params = hp.InputVStreamParams.make(small_ng, format_type=hp.FormatType.UINT8)
small_out_params = hp.OutputVStreamParams.make(small_ng, format_type=hp.FormatType.FLOAT32)

frame_vehicles = []
with small_ng.activate():
    with hp.InferVStreams(small_ng, small_in_params, small_out_params) as pipeline:
        for idx, frame in enumerate(frames):
            orig_h, orig_w = frame.shape[:2]
            resized = cv2.resize(frame, (640, 640))
            input_data = {'yolov11s/input_layer1': np.expand_dims(resized.astype(np.uint8), axis=0)}
            output = pipeline.infer(input_data)
            detections = output['yolov11s/yolov8_nms_postprocess'][0]
            vehicles = []
            for cls_id, dets in enumerate(detections):
                if len(dets) == 0:
                    continue
                for det in dets:
                    conf = float(det[4])
                    if conf < VEHICLE_CONF:
                        continue
                    y1 = int(det[0] * orig_h)
                    x1 = int(det[1] * orig_w)
                    y2 = int(det[2] * orig_h)
                    x2 = int(det[3] * orig_w)
                    vehicles.append({'cls_id': cls_id, 'class': VEHICLE_CLASSES[cls_id],
                                     'bbox': [x1,y1,x2,y2], 'conf': conf})
            frame_vehicles.append(vehicles)
            if idx % 30 == 0:
                print(f'  frame {idx}/{len(frames)}')

del small_ng
del target

# PASSATA 3: plate detection sui crop dei veicoli
print('Phase 2: plate detection (tiny_yolov4)...')
plate_hef = hp.HEF(PLATE_HEF)
target = hp.VDevice()
plate_ng = target.configure(plate_hef)[0]
plate_in_params = hp.InputVStreamParams.make(plate_ng, format_type=hp.FormatType.UINT8)
plate_out_params = hp.OutputVStreamParams.make(plate_ng, format_type=hp.FormatType.FLOAT32)
plate_input_info = plate_hef.get_input_vstream_infos()[0]

with plate_ng.activate():
    with hp.InferVStreams(plate_ng, plate_in_params, plate_out_params) as pipeline:
        for idx, vehicles in enumerate(frame_vehicles):
            frame = frames[idx]
            for v in vehicles:
                x1, y1, x2, y2 = v['bbox']
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    v['plate_bbox'] = None
                    continue
                ch, cw = crop.shape[:2]
                resized = cv2.resize(crop, (PLATE_INPUT_SIZE, PLATE_INPUT_SIZE))
                input_data = {plate_input_info.name: np.expand_dims(resized.astype(np.uint8), axis=0)}
                output = pipeline.infer(input_data)
                raw13 = output['tiny_yolov4_license_plates/conv19']
                raw26 = output['tiny_yolov4_license_plates/conv21']
                dets = decode_yolov4(raw13, ANCHORS_13, PLATE_INPUT_SIZE, ch, cw, PLATE_CONF)
                dets += decode_yolov4(raw26, ANCHORS_26, PLATE_INPUT_SIZE, ch, cw, PLATE_CONF)
                dets = nms(dets, IOU_THRESH)
                if dets:
                    best = dets[0]
                    px1, py1, px2, py2, pconf = best
                    v['plate_bbox'] = [x1+px1, y1+py1, x1+px2, y1+py2]
                    v['plate_conf'] = pconf
                else:
                    v['plate_bbox'] = None
            if idx % 30 == 0:
                print(f'  frame {idx}/{len(frame_vehicles)}')

del plate_ng
del target

# PASSATA 4: OCR sui crop targa
print('Phase 3: OCR (EasyOCR)...')
for idx, vehicles in enumerate(frame_vehicles):
    frame = frames[idx]
    for v in vehicles:
        if v.get('plate_bbox') is None:
            v['plate_text'] = ''
            continue
        px1, py1, px2, py2 = v['plate_bbox']
        plate_crop = frame[py1:py2, px1:px2]
        if plate_crop.size == 0:
            v['plate_text'] = ''
            continue
        result = reader.readtext(plate_crop)
        v['plate_text'] = ' '.join([r[1] for r in result]) if result else ''
    if idx % 30 == 0:
        print(f'  frame {idx}/{len(frame_vehicles)}')

# Disegna bbox e salva video
print('Saving output video...')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, 'output.mp4'), fourcc, fps, (width, height))

json_data = []
for idx, vehicles in enumerate(frame_vehicles):
    frame = frames[idx].copy()
    frame_record = {'frame': idx, 'vehicles': []}
    for v in vehicles:
        x1, y1, x2, y2 = v['bbox']
        color = VEHICLE_COLORS[v['cls_id']]
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        label = v['class'] + ' ' + str(round(v['conf'],2))
        cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        record = {'class': v['class'], 'bbox': v['bbox'], 'conf': v['conf']}
        if v.get('plate_bbox'):
            px1, py1, px2, py2 = v['plate_bbox']
            cv2.rectangle(frame, (px1,py1), (px2,py2), PLATE_COLOR, 2)
            text = v.get('plate_text', '')
            if text:
                cv2.putText(frame, text, (px1, py2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PLATE_COLOR, 2)
            record['plate_bbox'] = v['plate_bbox']
            record['plate_text'] = text
        frame_record['vehicles'].append(record)
    json_data.append(frame_record)
    out.write(frame)

out.release()

with open(os.path.join(OUTPUT_DIR, 'detections.json'), 'w') as f:
    json.dump(json_data, f, indent=2)

print(f'\nDone. Output in: {OUTPUT_DIR}')
print(f'  - output.mp4')
print(f'  - detections.json')
