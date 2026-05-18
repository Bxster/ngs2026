#!/usr/bin/env python3
"""
Plate detection (v1n HEF) + OCR (EasyOCR) end-to-end sul sample UniData italiano.
v4: legge TUTTE le targhe rilevate, immagine = match se almeno una uguaglia GT.

Lancio:
  source ~/hailo-apps/setup_env.sh
  PYTHONIOENCODING=utf-8 LANG=C.UTF-8 python3 italian_plate_benchmark.py
"""

import os
import re
import csv
import time
from pathlib import Path

import numpy as np
import cv2
import easyocr

from hailo_platform import (
    HEF, VDevice, HailoStreamInterface,
    ConfigureParams, InputVStreamParams, OutputVStreamParams,
    InferVStreams, FormatType,
)

from plate_postprocessing import postprocess_plate, clean_ocr_text


# ============================================================================
# CONFIG
# ============================================================================

HEF_PATH = Path("/home/pi/Downloads/license_plate_finetune_v1n.hef")
DATASET_DIR = Path("/home/pi/Downloads/sample_unidatapro")
GT_CSV = DATASET_DIR / "Italy.csv"

RESULTS_DIR = Path("/home/pi/italian_plate_results")
ANNOTATED_DIR = RESULTS_DIR / "annotated"
RESULTS_CSV = RESULTS_DIR / "results.csv"

INPUT_H, INPUT_W = 640, 640
CONF_THRES = 0.2


# ============================================================================
# UTILITY
# ============================================================================

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
    x1 = x1n * INPUT_W - pad[0]
    y1 = y1n * INPUT_H - pad[1]
    x2 = x2n * INPUT_W - pad[0]
    y2 = y2n * INPUT_H - pad[1]
    x1 /= scale; x2 /= scale
    y1 /= scale; y2 /= scale
    return (max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2))


def parse_hailo_nms_output(nms_output, conf_thres=0.2):
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


def levenshtein(a, b):
    if len(a) < len(b):
        return levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1,
                            prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def cer(pred, gt):
    if not gt:
        return 1.0 if pred else 0.0
    return levenshtein(pred, gt) / len(gt)


# ============================================================================
# MAIN
# ============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("ITALIAN PLATE BENCHMARK v4 (leggi TUTTE le targhe, match=almeno una)")
    print("=" * 70)
    print(f"HEF:        {HEF_PATH}")
    print(f"Dataset:    {DATASET_DIR}")
    print(f"Results:    {RESULTS_DIR}")
    print()

    print("[1/4] Carico GT...")
    gt_dict = {}
    with open(GT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_dict[row['File'].strip()] = row['Plate text'].strip()
    print(f"  {len(gt_dict)} righe")
    print()

    print("[2/4] Inizializzo EasyOCR...")
    reader_ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
    print()

    print("[3/4] Setup Hailo VDevice...")
    hef = HEF(str(HEF_PATH))
    results = []

    with VDevice() as target:
        cp = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
        ng = target.configure(hef, cp)[0]
        ng_params = ng.create_params()
        isp = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        osp = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        input_name = hef.get_input_vstream_infos()[0].name
        print(f"  Input: {input_name}")
        print()

        print(f"[4/4] Inferenza su {len(gt_dict)} immagini...")
        with InferVStreams(ng, isp, osp) as infer:
            with ng.activate(ng_params):
                for file_name, gt_text in gt_dict.items():
                    img_path = DATASET_DIR / file_name
                    if not img_path.exists():
                        print(f"  WARN: {file_name} non trovato")
                        continue
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    orig_h, orig_w = img.shape[:2]

                    img_lb, scale, pad = letterbox(img)
                    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
                    input_data = {input_name: np.expand_dims(img_rgb, 0).astype(np.uint8)}

                    output = infer.infer(input_data)
                    out_array = list(output.values())[0]
                    nms_out = out_array
                    while isinstance(nms_out, list) and len(nms_out) == 1 and isinstance(nms_out[0], list):
                        nms_out = nms_out[0]

                    detections = parse_hailo_nms_output(nms_out, CONF_THRES)
                    gt_clean = clean_ocr_text(gt_text)

                    if not detections:
                        print(f"  {file_name}: GT='{gt_clean}' NESSUNA TARGA RILEVATA")
                        results.append({
                            'file': file_name, 'gt': gt_clean,
                            'n_detections': 0,
                            'readings': [],
                            'best_raw': '', 'best_post': '',
                            'exact_raw': False, 'exact_post': False,
                            'cer_best_raw': 1.0, 'cer_best_post': 1.0,
                        })
                        annotated = img.copy()
                        cv2.putText(annotated, f"GT: {gt_clean}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(annotated, "NO PLATE DETECTED", (10, 65),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imwrite(str(ANNOTATED_DIR / file_name), annotated)
                        continue

                    # Leggi TUTTE le detection
                    readings = []
                    for d in detections:
                        bbox = unletterbox_bbox(d['bbox_norm'], scale, pad, orig_w, orig_h)
                        x1, y1, x2, y2 = map(int, bbox)
                        crop = img[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        # Preprocessing: grayscale (best variant dal report europeo v2)
                        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        ocr_results = reader_ocr.readtext(
                            crop_gray, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                            detail=0, paragraph=False)
                        raw_text = ''.join(ocr_results).upper()
                        raw_clean = clean_ocr_text(raw_text)
                        pp = postprocess_plate(raw_clean)
                        readings.append({
                            'bbox': bbox,
                            'score': d['score'],
                            'ocr_raw': raw_clean,
                            'ocr_post': pp['text'],
                            'format': pp['format'],
                            'matches_gt_raw': (raw_clean == gt_clean),
                            'matches_gt_post': (pp['text'] == gt_clean),
                            'cer_raw': cer(raw_clean, gt_clean),
                            'cer_post': cer(pp['text'], gt_clean),
                        })

                    # Match se almeno una lettura uguaglia GT
                    exact_raw = any(r['matches_gt_raw'] for r in readings)
                    exact_post = any(r['matches_gt_post'] for r in readings)

                    # Best raw / post = la lettura piu' vicina alla GT
                    best_raw_reading = min(readings, key=lambda r: r['cer_raw'])
                    best_post_reading = min(readings, key=lambda r: r['cer_post'])

                    results.append({
                        'file': file_name, 'gt': gt_clean,
                        'n_detections': len(readings),
                        'readings': readings,
                        'best_raw': best_raw_reading['ocr_raw'],
                        'best_post': best_post_reading['ocr_post'],
                        'exact_raw': exact_raw, 'exact_post': exact_post,
                        'cer_best_raw': best_raw_reading['cer_raw'],
                        'cer_best_post': best_post_reading['cer_post'],
                    })

                    print(f"  {file_name}: GT='{gt_clean}' ({len(readings)} targhe)  "
                          f"raw_match={exact_raw} post_match={exact_post}")
                    for i, r in enumerate(readings):
                        marker = "  >>" if (r['matches_gt_raw'] or r['matches_gt_post']) else "    "
                        print(f"{marker} [{i}] conf={r['score']:.2f} "
                              f"RAW='{r['ocr_raw']}' POST='{r['ocr_post']}' [{r['format']}]")

                    # Annotazione: tutte le bbox, ciascuna con il suo OCR
                    annotated = img.copy()
                    for i, r in enumerate(readings):
                        x1, y1, x2, y2 = map(int, r['bbox'])
                        # Verde se matcha (raw o post), giallo altrimenti
                        if r['matches_gt_raw'] or r['matches_gt_post']:
                            color = (0, 255, 0); thickness = 3
                        else:
                            color = (0, 255, 255); thickness = 2
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                        cv2.putText(annotated, f"#{i} {r['score']:.2f}",
                                    (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        # Testo sotto: RAW e POST
                        ly = y2 + 25
                        cv2.putText(annotated, f"RAW: {r['ocr_raw']}", (x1, ly),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 200, 0) if r['matches_gt_raw'] else (0, 100, 255), 2)
                        cv2.putText(annotated, f"POST: {r['ocr_post']}", (x1, ly + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 200, 0) if r['matches_gt_post'] else (0, 100, 255), 2)

                    # GT in alto a sinistra dell'immagine
                    cv2.putText(annotated, f"GT: {gt_clean}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                    cv2.putText(annotated, f"GT: {gt_clean}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                    cv2.imwrite(str(ANNOTATED_DIR / file_name), annotated)

    print()
    print("Salvataggio CSV...")
    with open(RESULTS_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'gt', 'n_detections',
                    'best_raw', 'best_post',
                    'exact_raw', 'exact_post',
                    'cer_best_raw', 'cer_best_post',
                    'all_readings_raw', 'all_readings_post'])
        for r in results:
            all_raw = "|".join(rd['ocr_raw'] for rd in r['readings'])
            all_post = "|".join(rd['ocr_post'] for rd in r['readings'])
            w.writerow([r['file'], r['gt'], r['n_detections'],
                        r['best_raw'], r['best_post'],
                        r['exact_raw'], r['exact_post'],
                        f"{r['cer_best_raw']:.4f}", f"{r['cer_best_post']:.4f}",
                        all_raw, all_post])

    n = len(results)
    detected = sum(1 for r in results if r['n_detections'] > 0)
    e_raw = sum(1 for r in results if r['exact_raw'])
    e_post = sum(1 for r in results if r['exact_post'])
    avg_cer_raw = sum(r['cer_best_raw'] for r in results) / n if n else 0
    avg_cer_post = sum(r['cer_best_post'] for r in results) / n if n else 0

    print()
    print("=" * 70)
    print("RIEPILOGO")
    print("=" * 70)
    print(f"Immagini totali:                {n}")
    print(f"Immagini con detection:         {detected}/{n}")
    print(f"OCR exact RAW (any reading):    {e_raw}/{n} ({e_raw/n*100:.1f}%)")
    print(f"OCR exact POSTPROC (any read):  {e_post}/{n} ({e_post/n*100:.1f}%)")
    print(f"CER medio (best RAW per img):   {avg_cer_raw:.3f}")
    print(f"CER medio (best POST per img):  {avg_cer_post:.3f}")
    print()
    print(f"Output:  {RESULTS_DIR}")


if __name__ == "__main__":
    main()
