#!/usr/bin/env python3
"""
NGS2026 Pipeline v3 — STREAMING REAL-TIME-READY

Differenze rispetto a v2:
  - Streaming: i frame vengono letti dal video uno alla volta (no pre-load in RAM)
  - Modalita' B nativa: yolo11s + v1n caricati una volta sola sul Hailo-8
  - Voting incrementale: la targa votata si aggiorna frame per frame
  - Chiusura tracker: quando un tracker_id non si vede da TRACKER_TIMEOUT_FRAMES,
    viene committato (targa finale fissata) e rimosso dal voting attivo
  - Video annotato scritto frame per frame, non a fine video
  - Pronto per sostituire VideoCapture(file) con VideoCapture(0) per camera live

Modalita' supportate:
  - streaming  (default): tutto frame-per-frame, simula real-time
  - sequential          : batch in due passate, piu' veloce su video registrati

Lancio:
  source ~/hailo-apps/setup_env.sh
  PYTHONIOENCODING=utf-8 LANG=C.UTF-8 python3 ngs_pipeline_v3.py
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

VIDEO_PATH = '/home/pi/Downloads/video2.mp4'   # path file o 0 per camera live
OUTPUT_DIR = '/home/pi/Desktop/ngs_pipeline_v3_out'

SMALL_HEF = '/home/pi/yolov11s.hef'
PLATE_HEF = '/home/pi/Downloads/license_plate_finetune_v1n.hef'

OCR_MODEL_NAME = "cct-xs-v1-global-model"

# 'streaming' (default, real-time-ready) | 'sequential' (batch in due passate)
EXECUTION_MODE = 'streaming'

# Numero di frame consecutivi senza apparizioni dopo cui un tracker_id viene
# considerato uscito di scena (commit + rimozione dal voting attivo).
# A 25 fps, 30 frame = 1.2 secondi.
TRACKER_TIMEOUT_FRAMES = 30

# Frame skipping: processa 1 frame ogni FRAME_SKIP del video sorgente.
# 1 = nessun skip (default). 2 = elabora la meta' dei frame. 3 = un terzo, ecc.
# Utile quando la camera produce piu' frame di quanti la pipeline ne possa gestire.
FRAME_SKIP = 1

# Merge by plate: due tracker chiusi con stessa targa entro questo gap = 1 veicolo
MERGE_MAX_GAP_SECONDS = 15.0

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
PLATE_INPUT_H, PLATE_INPUT_W = 640, 640

# Scrivi il video annotato
WRITE_ANNOTATED_VIDEO = False


# ============================================================================
# UTILITY
# ============================================================================

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def letterbox(img, target_h=640, target_w=640, pad_value=114):
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
    x1n, y1n, x2n, y2n = bbox_norm
    x1 = x1n * PLATE_INPUT_W - pad[0]
    y1 = y1n * PLATE_INPUT_H - pad[1]
    x2 = x2n * PLATE_INPUT_W - pad[0]
    y2 = y2n * PLATE_INPUT_H - pad[1]
    x1 /= scale; x2 /= scale
    y1 /= scale; y2 /= scale
    return (max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2))


def parse_v1n_output(nms_output, conf_thres=0.3):
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


# ============================================================================
# INFERENZA
# ============================================================================

def infer_vehicles(pipe, frame, input_name='yolov11s/input_layer1'):
    """Inferenza yolo11s. Pipeline gia' attivata."""
    orig_h, orig_w = frame.shape[:2]
    resized = cv2.resize(frame, (640, 640))
    input_data = {input_name: np.expand_dims(resized.astype(np.uint8), axis=0)}
    output = pipe.infer(input_data)
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
            vehicles.append({
                'cls_id': cls_id,
                'class': VEHICLE_CLASSES[cls_id],
                'bbox': [x1, y1, x2, y2],
                'conf': conf,
            })
    return vehicles


def infer_plates(pipe, frame, vehicles, input_name):
    """Plate detection sui crop dei veicoli. Pipeline gia' attivata."""
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
        output = pipe.infer(input_data)
        out_array = list(output.values())[0]
        nms_out = out_array
        while isinstance(nms_out, list) and len(nms_out) == 1 and isinstance(nms_out[0], list):
            nms_out = nms_out[0]
        detections = parse_v1n_output(nms_out, PLATE_CONF)
        if not detections:
            v['plate_bbox'] = None
            continue
        best = max(detections, key=lambda d: d['score'])
        bbox_crop = unletterbox_bbox(best['bbox_norm'], scale, pad, cw, ch)
        px1, py1, px2, py2 = bbox_crop
        v['plate_bbox'] = [int(x1 + px1), int(y1 + py1), int(x1 + px2), int(y1 + py2)]
        v['plate_conf'] = best['score']
    return vehicles


def run_ocr_on_crop(frame, plate_bbox, ocr_model):
    """OCR su un singolo crop targa. Ritorna stringa pulita."""
    px1, py1, px2, py2 = plate_bbox
    crop = frame[py1:py2, px1:px2]
    if crop.size == 0:
        return ''
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    try:
        result = ocr_model.run(crop_rgb)
        if isinstance(result, list) and len(result) > 0:
            pred_obj = result[0]
            text = pred_obj.plate if hasattr(pred_obj, 'plate') else str(pred_obj)
        else:
            text = str(result)
    except Exception:
        text = ''
    return text.upper().strip()


# ============================================================================
# VOTING DATABASE (incrementale + chiusura tracker)
# ============================================================================

class VotingDB:
    """
    Mantiene per ogni tracker_id attivo:
      - lista delle classi votate
      - lista delle letture OCR
      - first_frame, last_frame visti
    Quando un tracker_id non si vede per TRACKER_TIMEOUT_FRAMES, viene
    "committato": classe e targa finali (majority voting), e spostato in
    closed_tracks. Rimosso dal voting attivo per liberare memoria.
    """

    def __init__(self, timeout_frames=30):
        self.active = {}        # tid -> dict(classes, plates, first_frame, last_frame)
        self.closed = []        # lista di dict committati
        self.timeout = timeout_frames

    def add_observation(self, tid, frame_idx, cls_name, plate_text):
        if tid == -1:
            return
        if tid not in self.active:
            self.active[tid] = {
                'tracker_id': tid,
                'classes': [],
                'plates': [],
                'first_frame': frame_idx,
                'last_frame': frame_idx,
            }
        rec = self.active[tid]
        rec['classes'].append(cls_name)
        if plate_text:
            rec['plates'].append(plate_text)
        rec['last_frame'] = frame_idx

    def current_vote(self, tid):
        """Ritorna (classe, targa) votate al momento per il tid, senza chiudere."""
        if tid not in self.active:
            return None, 'unknown'
        rec = self.active[tid]
        cls_vote = Counter(rec['classes']).most_common(1)[0][0] if rec['classes'] else None
        plate_vote = Counter(rec['plates']).most_common(1)[0][0] if rec['plates'] else 'unknown'
        return cls_vote, plate_vote

    def commit_expired(self, current_frame_idx):
        """
        Chiude i tracker_id non visti da timeout_frames frame.
        Ritorna la lista dei record committati in questa chiamata.
        """
        committed_now = []
        to_close = [tid for tid, rec in self.active.items()
                    if current_frame_idx - rec['last_frame'] > self.timeout]
        for tid in to_close:
            rec = self.active.pop(tid)
            cls_vote = Counter(rec['classes']).most_common(1)[0][0] if rec['classes'] else 'unknown'
            plate_vote = Counter(rec['plates']).most_common(1)[0][0] if rec['plates'] else 'unknown'
            committed = {
                'tracker_id': tid,
                'class': cls_vote,
                'plate': plate_vote,
                'first_frame': rec['first_frame'],
                'last_frame': rec['last_frame'],
                'n_frames': len(rec['classes']),
                'n_plate_reads': len(rec['plates']),
            }
            self.closed.append(committed)
            committed_now.append(committed)
        return committed_now

    def flush_all(self, current_frame_idx):
        """A fine pipeline: chiude tutti i tracker rimasti aperti."""
        for tid in list(self.active.keys()):
            rec = self.active.pop(tid)
            cls_vote = Counter(rec['classes']).most_common(1)[0][0] if rec['classes'] else 'unknown'
            plate_vote = Counter(rec['plates']).most_common(1)[0][0] if rec['plates'] else 'unknown'
            self.closed.append({
                'tracker_id': tid,
                'class': cls_vote,
                'plate': plate_vote,
                'first_frame': rec['first_frame'],
                'last_frame': rec['last_frame'],
                'n_frames': len(rec['classes']),
                'n_plate_reads': len(rec['plates']),
            })


# ============================================================================
# MERGE BY PLATE (post-processing finale sui tracker chiusi)
# ============================================================================

def merge_closed_by_plate(closed_tracks, fps, max_gap_seconds=15.0):
    """
    Unisce tracker chiusi che condividono la stessa targa, se temporalmente vicini.
    Ritorna lista di veicoli univoci.
    """
    max_gap_frames = max_gap_seconds * fps

    # Raggruppa per targa
    plate_to_records = defaultdict(list)
    for rec in closed_tracks:
        plate_to_records[rec['plate']].append(rec)

    merged = []
    for plate, records in plate_to_records.items():
        if plate == 'unknown' or len(records) == 1:
            for r in records:
                merged.append({**r, 'merged_from': []})
            continue
        # Ordina per first_frame
        records_sorted = sorted(records, key=lambda r: r['first_frame'])
        current_cluster = [records_sorted[0]]
        cluster_last = records_sorted[0]['last_frame']
        clusters = [current_cluster]
        for r in records_sorted[1:]:
            gap = r['first_frame'] - cluster_last
            if gap <= max_gap_frames:
                current_cluster.append(r)
                cluster_last = max(cluster_last, r['last_frame'])
            else:
                current_cluster = [r]
                cluster_last = r['last_frame']
                clusters.append(current_cluster)
        for cluster in clusters:
            primary = cluster[0]
            merged_record = {
                'tracker_id': primary['tracker_id'],
                'class': Counter([c['class'] for c in cluster]).most_common(1)[0][0],
                'plate': plate,
                'first_frame': primary['first_frame'],
                'last_frame': max(c['last_frame'] for c in cluster),
                'n_frames': sum(c['n_frames'] for c in cluster),
                'n_plate_reads': sum(c['n_plate_reads'] for c in cluster),
                'merged_from': [c['tracker_id'] for c in cluster[1:]],
            }
            merged.append(merged_record)
    merged.sort(key=lambda r: r['first_frame'])
    return merged


# ============================================================================
# RENDER VIDEO ANNOTATO (per frame)
# ============================================================================

def render_frame(frame, tracked, voting_db):
    """Disegna bbox veicoli, targhe, ID e targa votata corrente."""
    annotated = frame.copy()
    for v in tracked:
        x1, y1, x2, y2 = v['bbox']
        color = VEHICLE_COLORS[v['cls_id']]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Classe e targa votate al momento (live)
        live_cls, live_plate = voting_db.current_vote(v['id'])
        display_cls = live_cls if live_cls else v['class']
        label = f"ID{v['id']} {display_cls} {v['conf']:.2f}"
        cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if v.get('plate_bbox'):
            px1, py1, px2, py2 = v['plate_bbox']
            cv2.rectangle(annotated, (px1, py1), (px2, py2), PLATE_COLOR, 2)
            current_text = v.get('plate_text', '')
            if current_text:
                cv2.putText(annotated, current_text, (px1, py2 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, PLATE_COLOR, 2)
            if live_plate and live_plate != 'unknown':
                cv2.putText(annotated, f'[{live_plate}]', (px1, max(15, py1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return annotated


# ============================================================================
# MAIN STREAMING (default)
# ============================================================================

def main_streaming():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=' * 70)
    print('NGS2026 Pipeline v3 - STREAMING REAL-TIME-READY')
    print('=' * 70)
    print(f'Video:      {VIDEO_PATH}')
    print(f'Output:     {OUTPUT_DIR}')
    print(f'Mode:       streaming')
    print(f'Tracker timeout: {TRACKER_TIMEOUT_FRAMES} frames (pipeline)')
    print(f'Frame skip:      {FRAME_SKIP} (processa 1 frame ogni {FRAME_SKIP})')
    print()

    # Carica OCR (una volta)
    print('[OCR] Caricamento fast-plate-ocr...')
    ocr_model = LicensePlateRecognizer(OCR_MODEL_NAME)
    print('  OK')

    # Apri video sorgente (no preload)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f'ERRORE: impossibile aprire video {VIDEO_PATH}')
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Video info: {total_frames} frames, {fps:.1f} fps, {width}x{height}')
    print()

    # Setup Hailo: VDevice unico + entrambi gli HEF
    print('Apertura VDevice unico (modalita B)...')
    small_hef = HEF(SMALL_HEF)
    plate_hef = HEF(PLATE_HEF)
    target = VDevice()
    small_cp = ConfigureParams.create_from_hef(hef=small_hef, interface=HailoStreamInterface.PCIe)
    small_ng = target.configure(small_hef, small_cp)[0]
    plate_cp = ConfigureParams.create_from_hef(hef=plate_hef, interface=HailoStreamInterface.PCIe)
    plate_ng = target.configure(plate_hef, plate_cp)[0]
    print('  Entrambi i modelli caricati sullo stesso VDevice')

    small_in = InputVStreamParams.make(small_ng, format_type=FormatType.UINT8)
    small_out = OutputVStreamParams.make(small_ng, format_type=FormatType.FLOAT32)
    plate_in = InputVStreamParams.make(plate_ng, format_type=FormatType.UINT8)
    plate_out = OutputVStreamParams.make(plate_ng, format_type=FormatType.FLOAT32)
    plate_input_name = plate_hef.get_input_vstream_infos()[0].name

    # Output video writer
    video_writer = None
    if WRITE_ANNOTATED_VIDEO:
        out_path = os.path.join(OUTPUT_DIR, 'output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Tracker e voting database
    tracker = sv.ByteTrack()
    voting_db = VotingDB(timeout_frames=TRACKER_TIMEOUT_FRAMES)

    # Stats
    timing_log = {
        'infer_small_ms': [],
        'infer_plate_ms': [],
        'ocr_ms': [],
        'tracking_ms': [],
        'render_ms': [],
        'total_frame_ms': [],
    }

    print()
    print('Inizio streaming loop...')
    t_pipeline_start = time.time()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FRAME SKIPPING: salta interi frame in base a FRAME_SKIP
            if FRAME_SKIP > 1 and (frame_idx % FRAME_SKIP) != 0:
                frame_idx += 1
                continue

            t_frame_start = time.time()

            # === Stage 1: vehicle detection (yolo11s) ===
            t0 = time.time()
            with small_ng.activate():
                with InferVStreams(small_ng, small_in, small_out) as pipe:
                    vehicles = infer_vehicles(pipe, frame)
            timing_log['infer_small_ms'].append((time.time() - t0) * 1000)

            # === Stage 2: plate detection (v1n) ===
            t0 = time.time()
            if vehicles:
                with plate_ng.activate():
                    with InferVStreams(plate_ng, plate_in, plate_out) as pipe:
                        vehicles = infer_plates(pipe, frame, vehicles, plate_input_name)
            timing_log['infer_plate_ms'].append((time.time() - t0) * 1000)

            # === Stage 3: tracking (incrementale) ===
            t0 = time.time()
            if vehicles:
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
                    # Ritrova plate_bbox associato (IoU match)
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
            else:
                # Update tracker comunque (mantiene state corretto anche su frame vuoti)
                empty = sv.Detections.empty()
                tracker.update_with_detections(empty)
                tracked = []
            timing_log['tracking_ms'].append((time.time() - t0) * 1000)

            # === Stage 4: OCR sui crop targa + voting incrementale ===
            t0 = time.time()
            for v in tracked:
                plate_text = ''
                if v.get('plate_bbox') is not None:
                    plate_text = run_ocr_on_crop(frame, v['plate_bbox'], ocr_model)
                v['plate_text'] = plate_text
                # Voting incrementale (qui i risultati si accumulano)
                voting_db.add_observation(v['id'], frame_idx, v['class'], plate_text)
            timing_log['ocr_ms'].append((time.time() - t0) * 1000)

            # === Chiusura tracker scaduti ===
            voting_db.commit_expired(frame_idx)

            # === Render + scrittura video ===
            t0 = time.time()
            if video_writer is not None:
                annotated = render_frame(frame, tracked, voting_db)
                video_writer.write(annotated)
            timing_log['render_ms'].append((time.time() - t0) * 1000)

            timing_log['total_frame_ms'].append((time.time() - t_frame_start) * 1000)

            if frame_idx % 30 == 0:
                elapsed = time.time() - t_pipeline_start
                rate = (frame_idx + 1) / elapsed
                processed_so_far = (frame_idx // FRAME_SKIP) + 1 if FRAME_SKIP > 0 else frame_idx + 1
                print(f'  frame {frame_idx}{f"/{total_frames}" if total_frames > 0 else ""}  '
                      f'processed {processed_so_far}  '
                      f'rate {rate:.1f} fps (camera)  '
                      f'active_tracks={len(voting_db.active)}  '
                      f'closed_tracks={len(voting_db.closed)}')

            frame_idx += 1
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        # Flush dei tracker ancora aperti a fine pipeline
        voting_db.flush_all(frame_idx)
        del small_ng, plate_ng, target

    t_pipeline_end = time.time()
    total_time = t_pipeline_end - t_pipeline_start

    # === Post-processing: merge by plate sui tracker chiusi ===
    print()
    print('Merge by plate sui tracker chiusi...')
    final_vehicles = merge_closed_by_plate(voting_db.closed, fps,
                                            max_gap_seconds=MERGE_MAX_GAP_SECONDS)
    n_merged = sum(len(v.get('merged_from', [])) for v in final_vehicles)
    print(f'  closed_tracks: {len(voting_db.closed)}')
    print(f'  ID unificati:  {n_merged}')
    print(f'  veicoli univoci: {len(final_vehicles)}')

    # === Output JSON ===
    json_data = {
        'video': str(VIDEO_PATH),
        'fps': fps,
        'frames_total': frame_idx,
        'execution_mode': 'streaming',
        'tracker_timeout_frames': TRACKER_TIMEOUT_FRAMES,
        'frame_skip': FRAME_SKIP,
        'merge_max_gap_seconds': MERGE_MAX_GAP_SECONDS,
        'timing_summary_ms': {
            'avg_infer_small': float(np.mean(timing_log['infer_small_ms'])),
            'avg_infer_plate': float(np.mean(timing_log['infer_plate_ms'])),
            'avg_ocr': float(np.mean(timing_log['ocr_ms'])),
            'avg_tracking': float(np.mean(timing_log['tracking_ms'])),
            'avg_render': float(np.mean(timing_log['render_ms'])),
            'avg_total_per_frame': float(np.mean(timing_log['total_frame_ms'])),
        },
        'total_time_seconds': total_time,
        'effective_fps': frame_idx / total_time if total_time > 0 else 0,
        'vehicle_count': len(final_vehicles),
        'vehicles': final_vehicles,
    }

    json_path = os.path.join(OUTPUT_DIR, 'detections.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    # === Riepilogo testuale ===
    print()
    print('=' * 70)
    print('RIEPILOGO FINALE (streaming)')
    print('=' * 70)
    print(f'Frame processati:    {frame_idx}')
    print(f'Tempo totale:        {total_time:.1f}s')
    print(f'FPS effettivi:       {frame_idx/total_time:.1f}')
    print()
    print(f'Tempi medi per frame (ms):')
    print(f'  infer yolo11s:     {np.mean(timing_log["infer_small_ms"]):.1f}')
    print(f'  infer v1n:         {np.mean(timing_log["infer_plate_ms"]):.1f}')
    print(f'  OCR:               {np.mean(timing_log["ocr_ms"]):.1f}')
    print(f'  tracking:          {np.mean(timing_log["tracking_ms"]):.1f}')
    print(f'  render:            {np.mean(timing_log["render_ms"]):.1f}')
    print(f'  totale per frame:  {np.mean(timing_log["total_frame_ms"]):.1f}')
    print()
    print(f'VEICOLI TROVATI:     {len(final_vehicles)}')
    print('-' * 70)
    print(f'{"ID":<6} {"CLASSE":<12} {"TARGA":<14} {"FRAME":>6} {"OCR_OK":>7}')
    print('-' * 70)
    for v in final_vehicles:
        print(f'{v["tracker_id"]:<6} {v["class"]:<12} {v["plate"]:<14} '
              f'{v["n_frames"]:>6} {v["n_plate_reads"]:>7}')
    print('-' * 70)
    print()
    out_path = os.path.join(OUTPUT_DIR, 'output.mp4')
    print(f'Output video:  {out_path}')
    print(f'Output JSON:   {json_path}')


# ============================================================================
# MAIN SEQUENTIAL (preservato come opzione benchmark batch)
# ============================================================================

def main_sequential():
    """Modalita' batch in due passate (per benchmark offline). Vecchio comportamento v2."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=' * 70)
    print('NGS2026 Pipeline v3 - SEQUENTIAL (batch)')
    print('=' * 70)
    print(f'Video:      {VIDEO_PATH}')
    print(f'Output:     {OUTPUT_DIR}')
    print(f'Mode:       sequential')
    print()

    print('[OCR] Caricamento fast-plate-ocr...')
    ocr_model = LicensePlateRecognizer(OCR_MODEL_NAME)
    print('  OK')

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Video info: {total_frames} frames, {fps:.1f} fps, {width}x{height}')

    # Pre-load frames (modalita' batch)
    print('Lettura frame in memoria...')
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f'  caricati {len(frames)} frame')

    t_start = time.time()
    frame_vehicles = [None] * len(frames)

    # Fase 1: yolo11s su tutti
    print('[Fase 1/4] Vehicle detection (yolo11s)...')
    small_hef = HEF(SMALL_HEF)
    target = VDevice()
    small_cp = ConfigureParams.create_from_hef(hef=small_hef, interface=HailoStreamInterface.PCIe)
    small_ng = target.configure(small_hef, small_cp)[0]
    small_in = InputVStreamParams.make(small_ng, format_type=FormatType.UINT8)
    small_out = OutputVStreamParams.make(small_ng, format_type=FormatType.FLOAT32)
    with small_ng.activate():
        with InferVStreams(small_ng, small_in, small_out) as pipe:
            for idx, frame in enumerate(frames):
                frame_vehicles[idx] = infer_vehicles(pipe, frame)
                if idx % 30 == 0:
                    print(f'  frame {idx}/{len(frames)}')
    del small_ng, target

    # Fase 2: v1n sui crop
    print('[Fase 2/4] Plate detection (v1n)...')
    plate_hef = HEF(PLATE_HEF)
    target = VDevice()
    plate_cp = ConfigureParams.create_from_hef(hef=plate_hef, interface=HailoStreamInterface.PCIe)
    plate_ng = target.configure(plate_hef, plate_cp)[0]
    plate_in = InputVStreamParams.make(plate_ng, format_type=FormatType.UINT8)
    plate_out = OutputVStreamParams.make(plate_ng, format_type=FormatType.FLOAT32)
    plate_input_name = plate_hef.get_input_vstream_infos()[0].name
    with plate_ng.activate():
        with InferVStreams(plate_ng, plate_in, plate_out) as pipe:
            for idx, vehicles in enumerate(frame_vehicles):
                infer_plates(pipe, frames[idx], vehicles, plate_input_name)
                if idx % 30 == 0:
                    print(f'  frame {idx}/{len(frame_vehicles)}')
    del plate_ng, target

    t_detection = time.time() - t_start

    # Fase 3: tracking sequenziale
    print('[Fase 3/4] Tracking...')
    t0 = time.time()
    tracker = sv.ByteTrack()
    frame_tracked = []
    for vehicles in frame_vehicles:
        if not vehicles:
            tracker.update_with_detections(sv.Detections.empty())
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
    t_tracking = time.time() - t0

    # Fase 4: OCR + voting
    print('[Fase 4/4] OCR...')
    t0 = time.time()
    voting_db = VotingDB(timeout_frames=10**9)  # niente chiusure intermedie in batch
    for idx, tracked in enumerate(frame_tracked):
        frame = frames[idx]
        for v in tracked:
            plate_text = ''
            if v.get('plate_bbox') is not None:
                plate_text = run_ocr_on_crop(frame, v['plate_bbox'], ocr_model)
            v['plate_text'] = plate_text
            voting_db.add_observation(v['id'], idx, v['class'], plate_text)
        if idx % 30 == 0:
            print(f'  OCR frame {idx}/{len(frame_tracked)}')
    voting_db.flush_all(len(frame_tracked))
    t_ocr = time.time() - t0

    # Render video annotato dopo aver tutto
    print('Salvataggio video annotato...')
    out_path = os.path.join(OUTPUT_DIR, 'output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for idx, tracked in enumerate(frame_tracked):
        annotated = render_frame(frames[idx], tracked, voting_db)
        writer.write(annotated)
    writer.release()

    # Merge by plate
    final_vehicles = merge_closed_by_plate(voting_db.closed, fps, MERGE_MAX_GAP_SECONDS)

    json_data = {
        'video': str(VIDEO_PATH),
        'fps': fps,
        'frames_total': len(frames),
        'execution_mode': 'sequential',
        'timings_seconds': {
            'detection_total': t_detection,
            'tracking': t_tracking,
            'ocr': t_ocr,
        },
        'vehicle_count': len(final_vehicles),
        'vehicles': final_vehicles,
    }
    json_path = os.path.join(OUTPUT_DIR, 'detections.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print()
    print('=' * 70)
    print('RIEPILOGO FINALE (sequential)')
    print('=' * 70)
    print(f'Frame processati:    {len(frames)}')
    print(f'Tempo detection:     {t_detection:.1f}s')
    print(f'Tempo tracking:      {t_tracking:.1f}s')
    print(f'Tempo OCR:           {t_ocr:.1f}s')
    print(f'Tempo totale:        {t_detection + t_tracking + t_ocr:.1f}s')
    print()
    print(f'VEICOLI TROVATI:     {len(final_vehicles)}')
    print('-' * 70)
    for v in final_vehicles:
        print(f'{v["tracker_id"]:<6} {v["class"]:<12} {v["plate"]:<14} '
              f'{v["n_frames"]:>6} {v["n_plate_reads"]:>7}')
    print('-' * 70)
    print(f'Output video:  {out_path}')
    print(f'Output JSON:   {json_path}')


# ============================================================================
# DISPATCH
# ============================================================================

if __name__ == '__main__':
    if EXECUTION_MODE == 'streaming':
        main_streaming()
    elif EXECUTION_MODE == 'sequential':
        main_sequential()
    else:
        raise ValueError(f'EXECUTION_MODE non valido: {EXECUTION_MODE}. '
                         f'Usa "streaming" o "sequential".')
