import hailo_platform as hp
import numpy as np
import cv2
import os
import json
import easyocr
import supervision as sv
from collections import defaultdict, Counter
import re

# ============ CONFIG ============
VIDEO_PATH = '/home/pi/Downloads/video3.mp4'
OUTPUT_DIR = '/home/pi/Desktop/test_pipeline_full'
SMALL_HEF = '/home/pi/yolov11s.hef'
PLATE_HEF = '/home/pi/Downloads/tiny_yolov4_license_plates.hef'

# Modalità: 'sequential' (prima tutta la small, poi tutta la tiny) | 'alternating' (per ogni frame switch)
EXECUTION_MODE = 'sequential'

ITALIAN_PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{3}[A-Z]{2}$')

APPLY_ITALIAN_FORMAT_FILTER = False  # True per filtrare voto solo su targhe formato AA000AA

VEHICLE_CLASSES = ['bus', 'car', 'pickup', 'truck', 'van', 'motorcycle']
VEHICLE_COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]
PLATE_COLOR = (0, 200, 255)

VEHICLE_CONF = 0.3
PLATE_CONF = 0.3
IOU_THRESH = 0.4
PLATE_INPUT_SIZE = 416

ANCHORS_13 = [(81,82), (135,169), (344,319)]
ANCHORS_26 = [(23,27), (37,58), (81,82)]

OCR_ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ HELPERS ============
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

def run_small_inference(pipeline, frame, input_name='yolov11s/input_layer1'):
    orig_h, orig_w = frame.shape[:2]
    resized = cv2.resize(frame, (640, 640))
    input_data = {input_name: np.expand_dims(resized.astype(np.uint8), axis=0)}
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
    return vehicles

def run_plate_inference(pipeline, frame, vehicles, input_name):
    for v in vehicles:
        x1, y1, x2, y2 = v['bbox']
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            v['plate_bbox'] = None
            continue
        ch, cw = crop.shape[:2]
        resized = cv2.resize(crop, (PLATE_INPUT_SIZE, PLATE_INPUT_SIZE))
        input_data = {input_name: np.expand_dims(resized.astype(np.uint8), axis=0)}
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
    return vehicles

# ============ MAIN ============
print('Loading EasyOCR...')
reader = easyocr.Reader(['en'], gpu=False)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Video: {total_frames} frames, {fps:.1f} fps, {width}x{height}')
print(f'Mode: {EXECUTION_MODE}')

print('Reading all frames...')
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print(f'Loaded {len(frames)} frames')

frame_vehicles = [None] * len(frames)

# ============ INFERENCE ============
if EXECUTION_MODE == 'sequential':
    # PASSATA 1: small su tutti i frame
    print('Phase 1: vehicle detection (small)...')
    small_hef = hp.HEF(SMALL_HEF)
    target = hp.VDevice()
    small_ng = target.configure(small_hef)[0]
    small_in = hp.InputVStreamParams.make(small_ng, format_type=hp.FormatType.UINT8)
    small_out = hp.OutputVStreamParams.make(small_ng, format_type=hp.FormatType.FLOAT32)
    with small_ng.activate():
        with hp.InferVStreams(small_ng, small_in, small_out) as pipeline:
            for idx, frame in enumerate(frames):
                frame_vehicles[idx] = run_small_inference(pipeline, frame)
                if idx % 30 == 0:
                    print(f'  frame {idx}/{len(frames)}')
    del small_ng, target

    # PASSATA 2: tiny_yolov4 sui crop
    print('Phase 2: plate detection (tiny_yolov4)...')
    plate_hef = hp.HEF(PLATE_HEF)
    target = hp.VDevice()
    plate_ng = target.configure(plate_hef)[0]
    plate_in = hp.InputVStreamParams.make(plate_ng, format_type=hp.FormatType.UINT8)
    plate_out = hp.OutputVStreamParams.make(plate_ng, format_type=hp.FormatType.FLOAT32)
    plate_input_name = plate_hef.get_input_vstream_infos()[0].name
    with plate_ng.activate():
        with hp.InferVStreams(plate_ng, plate_in, plate_out) as pipeline:
            for idx, vehicles in enumerate(frame_vehicles):
                run_plate_inference(pipeline, frames[idx], vehicles, plate_input_name)
                if idx % 30 == 0:
                    print(f'  frame {idx}/{len(frame_vehicles)}')
    del plate_ng, target

elif EXECUTION_MODE == 'alternating':
    # Per ogni frame: configura small, inferenza, deconfigura, configura tiny, inferenza, deconfigura
    print('Phase 1+2 alternating...')
    small_hef = hp.HEF(SMALL_HEF)
    plate_hef = hp.HEF(PLATE_HEF)
    plate_input_name = plate_hef.get_input_vstream_infos()[0].name

    for idx, frame in enumerate(frames):
        target = hp.VDevice()
        small_ng = target.configure(small_hef)[0]
        small_in = hp.InputVStreamParams.make(small_ng, format_type=hp.FormatType.UINT8)
        small_out = hp.OutputVStreamParams.make(small_ng, format_type=hp.FormatType.FLOAT32)
        with small_ng.activate():
            with hp.InferVStreams(small_ng, small_in, small_out) as pipeline:
                vehicles = run_small_inference(pipeline, frame)
        del small_ng, target

        target = hp.VDevice()
        plate_ng = target.configure(plate_hef)[0]
        plate_in = hp.InputVStreamParams.make(plate_ng, format_type=hp.FormatType.UINT8)
        plate_out = hp.OutputVStreamParams.make(plate_ng, format_type=hp.FormatType.FLOAT32)
        with plate_ng.activate():
            with hp.InferVStreams(plate_ng, plate_in, plate_out) as pipeline:
                vehicles = run_plate_inference(pipeline, frame, vehicles, plate_input_name)
        del plate_ng, target

        frame_vehicles[idx] = vehicles
        if idx % 10 == 0:
            print(f'  frame {idx}/{len(frames)}')

else:
    raise ValueError(f'Unknown mode: {EXECUTION_MODE}')

# ============ TRACKING (ByteTrack) ============
print('Phase 3: tracking (ByteTrack)...')
tracker = sv.ByteTrack()
frame_tracked = []
for idx, vehicles in enumerate(frame_vehicles):
    if not vehicles:
        frame_tracked.append([])
        continue
    xyxy = np.array([v['bbox'] for v in vehicles], dtype=np.float32)
    conf = np.array([v['conf'] for v in vehicles], dtype=np.float32)
    cls = np.array([v['cls_id'] for v in vehicles], dtype=int)
    detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
    detections = tracker.update_with_detections(detections)
    tracked = []
    for i in range(len(detections)):
        tid = int(detections.tracker_id[i]) if detections.tracker_id[i] is not None else -1
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        cls_id = int(detections.class_id[i])
        # ritrova il plate_bbox associato (matching per IoU)
        best_v = None
        best_iou = 0
        for v in vehicles:
            i_iou = iou([x1,y1,x2,y2], v['bbox'])
            if i_iou > best_iou:
                best_iou = i_iou
                best_v = v
        plate_bbox = best_v.get('plate_bbox') if best_v else None
        tracked.append({
            'id': tid,
            'cls_id': cls_id,
            'class': VEHICLE_CLASSES[cls_id],
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'conf': float(detections.confidence[i]),
            'plate_bbox': plate_bbox,
        })
    frame_tracked.append(tracked)

# ============ OCR sui crop targa ============
print('Phase 4: OCR (EasyOCR)...')
for idx, tracked in enumerate(frame_tracked):
    frame = frames[idx]
    for v in tracked:
        if v.get('plate_bbox') is None:
            v['plate_text'] = ''
            continue
        px1, py1, px2, py2 = v['plate_bbox']
        plate_crop = frame[py1:py2, px1:px2]
        if plate_crop.size == 0:
            v['plate_text'] = ''
            continue
        result = reader.readtext(plate_crop, allowlist=OCR_ALLOWLIST)
        v['plate_text'] = ' '.join([r[1] for r in result]) if result else ''
    if idx % 30 == 0:
        print(f'  frame {idx}/{len(frame_tracked)}')

# ============ Voto maggioranza per ID ============
print('Phase 5: majority voting...')
id_classes = defaultdict(list)
id_plates = defaultdict(list)
for tracked in frame_tracked:
    for v in tracked:
        if v['id'] == -1:
            continue
        id_classes[v['id']].append(v['class'])
        if v.get('plate_text'):
            plate_clean = v['plate_text'].replace(' ', '').upper()
            if APPLY_ITALIAN_FORMAT_FILTER:
                if ITALIAN_PLATE_REGEX.match(plate_clean):
                    id_plates[v['id']].append(plate_clean)
            else:
                id_plates[v['id']].append(plate_clean)

id_summary = {}
for tid, classes in id_classes.items():
    cls_vote = Counter(classes).most_common(1)[0][0]
    plates = id_plates.get(tid, [])
    plate_vote = Counter(plates).most_common(1)[0][0] if plates else ''
    id_summary[tid] = {'class': cls_vote, 'plate': plate_vote, 'samples': len(classes)}

# ============ Output video + JSON ============
print('Saving output video...')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, 'output3.mp4'), fourcc, fps, (width, height))

json_data = {'frames': [], 'summary': id_summary}
for idx, tracked in enumerate(frame_tracked):
    frame = frames[idx].copy()
    frame_record = {'frame': idx, 'vehicles': []}
    for v in tracked:
        x1, y1, x2, y2 = v['bbox']
        color = VEHICLE_COLORS[v['cls_id']]
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        # Usa la classe e targa votate (più stabili)
        final_class = id_summary.get(v['id'], {}).get('class', v['class'])
        final_plate = id_summary.get(v['id'], {}).get('plate', '')
        label = f"ID{v['id']} {final_class} {round(v['conf'],2)}"
        cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        record = {'id': v['id'], 'class': v['class'], 'bbox': v['bbox'], 'conf': v['conf']}
        if v.get('plate_bbox'):
            px1, py1, px2, py2 = v['plate_bbox']
            cv2.rectangle(frame, (px1,py1), (px2,py2), PLATE_COLOR, 2)
            text = v.get('plate_text', '')
            display = final_plate if final_plate else text
            if display:
                cv2.putText(frame, display, (px1, py2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PLATE_COLOR, 2)
            record['plate_bbox'] = v['plate_bbox']
            record['plate_text'] = text
        frame_record['vehicles'].append(record)
    json_data['frames'].append(frame_record)
    out.write(frame)

out.release()

with open(os.path.join(OUTPUT_DIR, 'detections3.json'), 'w') as f:
    json.dump(json_data, f, indent=2)

print(f'\nDone. Output in: {OUTPUT_DIR}')
print(f'  - output.mp4')
print(f'  - detections.json')
print(f'\nVehicles tracked: {len(id_summary)}')
for tid, info in sorted(id_summary.items()):
    print(f'  ID{tid}: {info["class"]} {"(" + info["plate"] + ")" if info["plate"] else ""} [{info["samples"]} frames]')
