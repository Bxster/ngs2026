#!/usr/bin/env python3
"""
NGS2026 Pipeline v2 — yolo11s + license_plate_finetune_v1n + fast-plate-ocr + ByteTrack

Architettura:
  - Stage 1: yolo11s.hef rileva veicoli (6 classi)
  - Stage 2: license_plate_finetune_v1n.hef rileva targhe SUI CROP dei veicoli
  - Stage 3: ByteTrack assegna ID stabili ai veicoli
  - Stage 4: fast-plate-ocr cct-xs-v1-global-model legge le targhe
  - Stage 5: majority voting per ID (a fine video, dopo che ogni tracker_id e' sparito)

Lancio:
  source ~/hailo-apps/setup_env.sh
  PYTHONIOENCODING=utf-8 LANG=C.UTF-8 python3 ngs_pipeline_v2.py
"""

import os
import json
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import cv2
import supervision as sv

import hailo_platform as hp
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface,
    ConfigureParams, InputVStreamParams, OutputVStreamParams,
    InferVStreams, FormatType,
)

from fast_plate_ocr import LicensePlateRecognizer


# ============================================================================
# CONFIG
# ============================================================================

VIDEO_PATH = '/home/pi/Downloads/video2.mp4'
OUTPUT_DIR = '/home/pi/Desktop/ngs_pipeline_v2_out'

SMALL_HEF = '/home/pi/yolov11s.hef'
PLATE_HEF = '/home/pi/Downloads/license_plate_finetune_v1n.hef'

OCR_MODEL_NAME = "cct-xs-v1-global-model"

# Merge by plate: due tracker_id si uniscono solo se appaiono entro questo gap temporale
MERGE_MAX_GAP_SECONDS = 15.0

# 'sequential' = prima tutto yolo11s su tutti i frame, poi tutto v1n sui crop
# 'alternating' = per ogni frame, configura small, infer, deconfigura, configura v1n, infer, deconfigura
EXECUTION_MODE = 'sequential'

VEHICLE_CLASSES = ['bus', 'car', 'pickup', 'truck', 'van', 'motorcycle']
VEHICLE_COLORS = [
    (255, 0, 0),     # bus - blu
    (0, 255, 0),     # car - verde
    (0, 0, 255),     # pickup - rosso
    (255, 255, 0),   # truck - giallo
    (0, 255, 255),   # van - ciano
    (255, 0, 255),   # motorcycle - magenta
]
PLATE_COLOR = (0, 200, 255)

VEHICLE_CONF = 0.3
PLATE_CONF = 0.3

# Input shape v1n
PLATE_INPUT_H, PLATE_INPUT_W = 640, 640


# ============================================================================
# UTILITY GENERICHE
# ============================================================================

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


# ============================================================================
# STAGE 1 — YOLO11S VEHICLE DETECTION
# ============================================================================

def run_small_inference(pipeline, frame, input_name='yolov11s/input_layer1'):
    """Inferenza yolo11s su frame intero. Ritorna lista di veicoli."""
    orig_h, orig_w = frame.shape[:2]
    resized = cv2.resize(frame, (640, 640))
    input_data = {input_name: np.expand_dims(resized.astype(np.uint8), axis=0)}
    output = pipeline.infer(input_data)
    # NMS integrato: [batch][lista di 6 classi][lista di detection per classe]
    detections = output['yolov11s/yolov8_nms_postprocess'][0]
    vehicles = []
    for cls_id, dets in enumerate(detections):
        if len(dets) == 0:
            continue
        for det in dets:
            conf = float(det[4])
            if conf < VEHICLE_CONF:
                continue
            # Hailo restituisce y1, x1, y2, x2 normalizzati
            y1 = int(det[0] * orig_h)
            x1 = int(det[1] * orig_w)
            y2 = int(det[2] * orig_h)
            x2 = int(det[3] * orig_w)
            vehicles.append({
                'cls_id': cls_id,
                'class': VEHICLE_CLASSES[cls_id],
                'bbox': [x1, y1, x2, y2],
                'conf': conf,
            })
    return vehicles


# ============================================================================
# STAGE 2 — V1N PLATE DETECTION SUI CROP VEICOLO
# ============================================================================

def letterbox(img, target_h=640, target_w=640, pad_value=114):
    """Resize con preservazione aspect ratio + padding."""
    h0, w0 = img.shape[:2]
    scale = min(target_h / h0, target_w / w0)
    new_h, new_w = int(round(h0 * scale)), int(round(w0 * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    out = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                              cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
    return out, scale, (pad_left, pad_top)


def unletterbox_bbox(bbox_norm, scale, pad, orig_w, orig_h):
    """Riporta bbox normalizzate (letterbox 640x640) alle coordinate originali del crop."""
    x1n, y1n, x2n, y2n = bbox_norm
    x1 = x1n * PLATE_INPUT_W - pad[0]
    y1 = y1n * PLATE_INPUT_H - pad[1]
    x2 = x2n * PLATE_INPUT_W - pad[0]
    y2 = y2n * PLATE_INPUT_H - pad[1]
    x1 /= scale; x2 /= scale
    y1 /= scale; y2 /= scale
    return (max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2))


def parse_v1n_output(nms_output, conf_thres=0.3):
    """Parsing output NMS v1n (singola classe license-plate)."""
    detections = []
    try:
        n = len(nms_output)
    except TypeError:
        return detections
    for class_id in range(n):
        class_dets = nms_output[class_id]
        if class_dets is None or not hasattr(class_dets, '__len__'):
            continue
        if len(class_dets) == 0:
            continue
        for det in class_dets:
            det_arr = np.asarray(det).ravel()
            if det_arr.size < 5:
                continue
            y1n, x1n, y2n, x2n, score = (float(det_arr[i]) for i in range(5))
            if score < conf_thres:
                continue
            detections.append({'score': score, 'bbox_norm': (x1n, y1n, x2n, y2n)})
    return detections


def run_plate_inference(pipeline, frame, vehicles, input_name):
    """
    Per ogni veicolo del frame, fa plate detection sul crop e aggiunge plate_bbox in coord assolute.
    Sceglie la detection con confidence piu' alta per veicolo.
    """
    for v in vehicles:
        x1, y1, x2, y2 = v['bbox']
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            v['plate_bbox'] = None
            continue
        ch, cw = crop.shape[:2]

        crop_lb, scale, pad = letterbox(crop, PLATE_INPUT_H, PLATE_INPUT_W)
        crop_rgb = cv2.cvtColor(crop_lb, cv2.COLOR_BGR2RGB)
        input_data = {input_name: np.expand_dims(crop_rgb, 0).astype(np.uint8)}

        output = pipeline.infer(input_data)
        out_array = list(output.values())[0]
        nms_out = out_array
        # Scendi nelle liste annidate fino al livello classi
        while isinstance(nms_out, list) and len(nms_out) == 1 and isinstance(nms_out[0], list):
            nms_out = nms_out[0]

        detections = parse_v1n_output(nms_out, PLATE_CONF)
        if not detections:
            v['plate_bbox'] = None
            continue

        # Scelta: massima confidence
        best = max(detections, key=lambda d: d['score'])
        bbox_crop = unletterbox_bbox(best['bbox_norm'], scale, pad, cw, ch)
        px1, py1, px2, py2 = bbox_crop
        # Riporta in coord assolute del frame
        v['plate_bbox'] = [int(x1 + px1), int(y1 + py1), int(x1 + px2), int(y1 + py2)]
        v['plate_conf'] = best['score']
    return vehicles


# ============================================================================
# STAGE 3 — TRACKING (BYTETRACK)
# ============================================================================

def apply_tracking(frame_vehicles):
    """Applica ByteTrack ai veicoli di ogni frame. Ritorna lista per-frame con tracker_id."""
    tracker = sv.ByteTrack()
    frame_tracked = []
    for vehicles in frame_vehicles:
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
            # Ritrova il plate_bbox associato (matching IoU sul veicolo originale)
            best_v = None
            best_iou = 0
            for v in vehicles:
                i_iou = iou([x1, y1, x2, y2], v['bbox'])
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
    return frame_tracked


# ============================================================================
# STAGE 4 — OCR FAST-PLATE-OCR
# ============================================================================

def run_ocr_on_plates(frame_tracked, frames, ocr_model):
    """Per ogni plate_bbox nei tracked vehicles, esegue OCR con fast-plate-ocr."""
    for idx, tracked in enumerate(frame_tracked):
        frame = frames[idx]
        for v in tracked:
            if v.get('plate_bbox') is None:
                v['plate_text'] = ''
                continue
            px1, py1, px2, py2 = v['plate_bbox']
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                v['plate_text'] = ''
                continue
            # fast-plate-ocr cct-xs-v1 vuole RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            try:
                result = ocr_model.run(crop_rgb)
                if isinstance(result, list) and len(result) > 0:
                    pred_obj = result[0]
                    text = pred_obj.plate if hasattr(pred_obj, 'plate') else str(pred_obj)
                else:
                    text = str(result)
            except Exception as e:
                text = ''
                print(f'  WARN OCR frame {idx} id={v["id"]}: {e}')
            v['plate_text'] = text.upper().strip()
        if idx % 30 == 0:
            print(f'  OCR frame {idx}/{len(frame_tracked)}')
    return frame_tracked


# ============================================================================
# STAGE 5 — MAJORITY VOTING A FINE VIDEO
# ============================================================================

def majority_voting(frame_tracked):
    """
    Per ogni tracker_id, raccoglie tutte le letture (classe + targa) e applica voto maggioritario.
    Ritorna dict: {tid: {'class': str, 'plate': str_or_unknown, 'n_frames': int, 'n_plate_reads': int}}
    """
    id_classes = defaultdict(list)
    id_plates = defaultdict(list)

    for tracked in frame_tracked:
        for v in tracked:
            if v['id'] == -1:
                continue
            id_classes[v['id']].append(v['class'])
            if v.get('plate_text'):
                id_plates[v['id']].append(v['plate_text'])

    summary = {}
    for tid, classes in id_classes.items():
        cls_vote = Counter(classes).most_common(1)[0][0]
        plates = id_plates.get(tid, [])
        plate_vote = Counter(plates).most_common(1)[0][0] if plates else 'unknown'
        summary[tid] = {
            'class': cls_vote,
            'plate': plate_vote,
            'n_frames': len(classes),
            'n_plate_reads': len(plates),
        }
    return summary

def merge_by_plate(summary, frame_tracked, fps, max_gap_seconds=10.0):
    """
    Unisce tracker_id che condividono la stessa targa votata,
    SOLO se le loro finestre temporali sono separate da meno di max_gap_seconds.
    Se il gap e' maggiore, vengono considerati transiti distinti.

    Ritorna:
      - merged_summary: dict {primary_id: info_aggregata}
      - id_remap: dict {tracker_id_originale: primary_id}
    """
    # Per ogni tid: trova primo e ultimo frame in cui appare
    tid_frames = {}  # tid -> (first_frame, last_frame)
    for idx, tracked in enumerate(frame_tracked):
        for v in tracked:
            tid = v['id']
            if tid == -1:
                continue
            if tid not in tid_frames:
                tid_frames[tid] = [idx, idx]
            else:
                tid_frames[tid][1] = idx

    max_gap_frames = max_gap_seconds * fps

    # Raggruppa tid per targa votata (escludi unknown)
    plate_to_tids = {}  # plate -> [tid, tid, ...]
    for tid, info in summary.items():
        plate = info['plate']
        if plate == 'unknown':
            continue
        plate_to_tids.setdefault(plate, []).append(tid)

    # Costruisci id_remap, default = se stesso
    id_remap = {tid: tid for tid in summary.keys()}

    # Per ogni targa con piu' tid, raggruppa in cluster temporali
    for plate, tids in plate_to_tids.items():
        if len(tids) < 2:
            continue
        # Ordina tid per first_frame
        tids_sorted = sorted(tids, key=lambda t: tid_frames[t][0])

        # Greedy clustering: scorri in ordine cronologico,
        # aggrega al cluster corrente se gap <= max_gap_frames
        current_cluster = [tids_sorted[0]]
        cluster_last_frame = tid_frames[tids_sorted[0]][1]
        clusters = [current_cluster]

        for tid in tids_sorted[1:]:
            this_first = tid_frames[tid][0]
            gap = this_first - cluster_last_frame
            if gap <= max_gap_frames:
                # Merge nel cluster corrente
                current_cluster.append(tid)
                cluster_last_frame = max(cluster_last_frame, tid_frames[tid][1])
            else:
                # Apre nuovo cluster (transito distinto)
                current_cluster = [tid]
                cluster_last_frame = tid_frames[tid][1]
                clusters.append(current_cluster)

        # Per ogni cluster, primary_tid = il primo cronologicamente
        for cluster in clusters:
            primary = cluster[0]
            for tid in cluster:
                id_remap[tid] = primary

    # Costruisci summary mergiato
    merged_summary = {}
    for tid, info in summary.items():
        primary = id_remap[tid]
        if primary not in merged_summary:
            merged_summary[primary] = {
                'class': info['class'],
                'plate': info['plate'],
                'n_frames': 0,
                'n_plate_reads': 0,
                'merged_from': [],
            }
        merged_summary[primary]['n_frames'] += info['n_frames']
        merged_summary[primary]['n_plate_reads'] += info['n_plate_reads']
        if tid != primary:
            merged_summary[primary]['merged_from'].append(tid)

    return merged_summary, id_remap

# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=' * 70)
    print('NGS2026 Pipeline v2')
    print('=' * 70)
    print(f'Video:      {VIDEO_PATH}')
    print(f'Output:     {OUTPUT_DIR}')
    print(f'Mode:       {EXECUTION_MODE}')
    print(f'Small HEF:  {SMALL_HEF}')
    print(f'Plate HEF:  {PLATE_HEF}')
    print(f'OCR model:  {OCR_MODEL_NAME}')
    print()

    # Carica OCR (singolo caricamento riutilizzato)
    print('[OCR] Caricamento fast-plate-ocr...')
    ocr_model = LicensePlateRecognizer(OCR_MODEL_NAME)
    print('  OK')
    print()

    # Apri video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Video info: {total_frames} frames, {fps:.1f} fps, {width}x{height}')

    # Carica tutti i frame in memoria
    print('Lettura frame in memoria...')
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f'  caricati {len(frames)} frame')
    print()

    frame_vehicles = [None] * len(frames)
    plate_input_name_holder = [None]  # box mutabile per condividere tra branch

    t_start = time.time()

    if EXECUTION_MODE == 'sequential':
        # FASE 1: yolo11s su tutti i frame
        print('[Fase 1/4] Vehicle detection (yolo11s)...')
        small_hef = HEF(SMALL_HEF)
        target = VDevice()
        small_ng = target.configure(small_hef)[0]
        small_in = InputVStreamParams.make(small_ng, format_type=FormatType.UINT8)
        small_out = OutputVStreamParams.make(small_ng, format_type=FormatType.FLOAT32)
        with small_ng.activate():
            with InferVStreams(small_ng, small_in, small_out) as pipeline:
                for idx, frame in enumerate(frames):
                    frame_vehicles[idx] = run_small_inference(pipeline, frame)
                    if idx % 30 == 0:
                        print(f'  frame {idx}/{len(frames)}')
        del small_ng, target

        # FASE 2: v1n sui crop veicoli
        print('[Fase 2/4] Plate detection (license_plate_finetune_v1n)...')
        plate_hef = HEF(PLATE_HEF)
        target = VDevice()
        plate_ng = target.configure(plate_hef)[0]
        plate_in = InputVStreamParams.make(plate_ng, format_type=FormatType.UINT8)
        plate_out = OutputVStreamParams.make(plate_ng, format_type=FormatType.FLOAT32)
        plate_input_name = plate_hef.get_input_vstream_infos()[0].name
        plate_input_name_holder[0] = plate_input_name
        with plate_ng.activate():
            with InferVStreams(plate_ng, plate_in, plate_out) as pipeline:
                for idx, vehicles in enumerate(frame_vehicles):
                    run_plate_inference(pipeline, frames[idx], vehicles, plate_input_name)
                    if idx % 30 == 0:
                        print(f'  frame {idx}/{len(frame_vehicles)}')
        del plate_ng, target

    elif EXECUTION_MODE == 'alternating':
        print('[Fase 1+2/4] Vehicle + plate detection (alternating per-frame)...')
        small_hef = HEF(SMALL_HEF)
        plate_hef = HEF(PLATE_HEF)
        plate_input_name = plate_hef.get_input_vstream_infos()[0].name
        plate_input_name_holder[0] = plate_input_name

        for idx, frame in enumerate(frames):
            # Vehicle
            target = VDevice()
            small_ng = target.configure(small_hef)[0]
            small_in = InputVStreamParams.make(small_ng, format_type=FormatType.UINT8)
            small_out = OutputVStreamParams.make(small_ng, format_type=FormatType.FLOAT32)
            with small_ng.activate():
                with InferVStreams(small_ng, small_in, small_out) as pipeline:
                    vehicles = run_small_inference(pipeline, frame)
            del small_ng, target

            # Plate
            target = VDevice()
            plate_ng = target.configure(plate_hef)[0]
            plate_in = InputVStreamParams.make(plate_ng, format_type=FormatType.UINT8)
            plate_out = OutputVStreamParams.make(plate_ng, format_type=FormatType.FLOAT32)
            with plate_ng.activate():
                with InferVStreams(plate_ng, plate_in, plate_out) as pipeline:
                    vehicles = run_plate_inference(pipeline, frame, vehicles, plate_input_name)
            del plate_ng, target

            frame_vehicles[idx] = vehicles
            if idx % 10 == 0:
                print(f'  frame {idx}/{len(frames)}')
    else:
        raise ValueError(f'EXECUTION_MODE sconosciuto: {EXECUTION_MODE}')

    t_detection = time.time() - t_start

    # FASE 3: tracking
    print()
    print('[Fase 3/4] Tracking (ByteTrack)...')
    t0 = time.time()
    frame_tracked = apply_tracking(frame_vehicles)
    t_tracking = time.time() - t0

    # FASE 4: OCR
    print()
    print('[Fase 4/4] OCR (fast-plate-ocr cct-xs-v1)...')
    t0 = time.time()
    frame_tracked = run_ocr_on_plates(frame_tracked, frames, ocr_model)
    t_ocr = time.time() - t0

    # MAJORITY VOTING
    print('Majority voting per tracker_id (a fine video)...')
    summary = majority_voting(frame_tracked)
    print(f'  trovati {len(summary)} ID univoci (pre-merge)')

    print('Merge by plate (unifica ID che leggono la stessa targa)...')
    merged_summary, id_remap = merge_by_plate(summary, frame_tracked, fps,
                                               max_gap_seconds=MERGE_MAX_GAP_SECONDS)
    n_merged = sum(1 for k, v in id_remap.items() if k != v)
    print(f'  ID unificati: {n_merged}')
    print(f'  veicoli univoci dopo merge: {len(merged_summary)}')
    print()

    # Sostituisce summary con merged_summary per il resto del codice
    summary = merged_summary

    # OUTPUT VIDEO
    print('Salvataggio video annotato...')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(OUTPUT_DIR, 'output.mp4')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    json_data = {
        'video': str(VIDEO_PATH),
        'fps': fps,
        'frames_total': len(frames),
        'execution_mode': EXECUTION_MODE,
        'timings_seconds': {
            'detection_total': t_detection,
            'tracking': t_tracking,
            'ocr': t_ocr,
        },
        'vehicle_count': len(summary),
        'vehicles': [],  # popolato dopo
        'frames': [],
    }

    for idx, tracked in enumerate(frame_tracked):
        frame = frames[idx].copy()
        frame_record = {'frame': idx, 'vehicles': []}
        for v in tracked:
            x1, y1, x2, y2 = v['bbox']
            color = VEHICLE_COLORS[v['cls_id']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Usa label finale (classe e targa votate) dal summary
            original_tid = v['id']
            tid = id_remap.get(original_tid, original_tid)  # usa primary ID
            final_class = summary.get(tid, {}).get('class', v['class'])
            final_plate = summary.get(tid, {}).get('plate', 'unknown')

            label = f"ID{tid} {final_class} {v['conf']:.2f}"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            record = {
                'tracker_id': tid,
                'class': v['class'],
                'bbox': v['bbox'],
                'conf': v['conf'],
            }
            if v.get('plate_bbox'):
                px1, py1, px2, py2 = v['plate_bbox']
                cv2.rectangle(frame, (px1, py1), (px2, py2), PLATE_COLOR, 2)
                # Sotto la targa: lettura corrente (frame) e in piccolo
                current_text = v.get('plate_text', '')
                if current_text:
                    cv2.putText(frame, current_text, (px1, py2 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, PLATE_COLOR, 2)
                # Sopra la targa: lettura finale votata
                if final_plate and final_plate != 'unknown':
                    cv2.putText(frame, f'[{final_plate}]', (px1, max(15, py1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                record['plate_bbox'] = v['plate_bbox']
                record['plate_text_frame'] = current_text
            frame_record['vehicles'].append(record)
        json_data['frames'].append(frame_record)
        out.write(frame)

    out.release()

    # Popola riepilogo veicoli nel JSON
    for tid in sorted(summary.keys()):
        info = summary[tid]
        json_data['vehicles'].append({
            'tracker_id': tid,
            'class': info['class'],
            'plate': info['plate'],
            'n_frames_seen': info['n_frames'],
            'n_plate_reads': info['n_plate_reads'],
        })

    json_path = os.path.join(OUTPUT_DIR, 'detections.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    # OUTPUT TESTUALE
    print()
    print('=' * 70)
    print('RIEPILOGO FINALE')
    print('=' * 70)
    print(f'Video processato:    {len(frames)} frame @ {fps:.1f} fps')
    print(f'Modalita\':           {EXECUTION_MODE}')
    print(f'Tempo detection:     {t_detection:.1f}s')
    print(f'Tempo tracking:      {t_tracking:.1f}s')
    print(f'Tempo OCR:           {t_ocr:.1f}s')
    print(f'Tempo totale:        {(t_detection + t_tracking + t_ocr):.1f}s')
    print()
    print(f'VEICOLI TROVATI:     {len(summary)}')
    print('-' * 70)
    print(f'{"ID":<6} {"CLASSE":<12} {"TARGA":<14} {"FRAME":>6} {"OCR_OK":>7}')
    print('-' * 70)
    for tid in sorted(summary.keys()):
        info = summary[tid]
        print(f'{tid:<6} {info["class"]:<12} {info["plate"]:<14} '
              f'{info["n_frames"]:>6} {info["n_plate_reads"]:>7}')
    print('-' * 70)
    print()
    print(f'Output:  {out_path}')
    print(f'JSON:    {json_path}')


if __name__ == '__main__':
    main()
