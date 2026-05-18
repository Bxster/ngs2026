#!/usr/bin/env python3
"""
Benchmark ibrido OCR: fast-plate-ocr cct-xs-v1 + Gemini Flash come fallback.

Strategia:
  1. fast-plate-ocr legge tutte le 735 immagini europee
  2. Sui casi falliti (predizione != GT) viene chiamato Gemini Flash via Vertex AI
  3. Si misura quanti casi falliti Gemini recupera correttamente

Setup Gemini: Vertex AI + ADC (uguale al benchmark EasyOCR vs Gemini precedente).

Output:
  RESULTS_DIR/
    fastplate_results.csv      -> filename, gt, fastplate_pred, exact, cer, time_ms
    gemini_fallback_results.csv -> solo i casi falliti, con risposta Gemini
    hybrid_summary.csv          -> metriche ibride (fp+gemini) vs solo fp
    confusion_pairs_hybrid.csv  -> errori carattere residui dopo fallback

Lancio:
  PYTHONIOENCODING=utf-8 LANG=C.UTF-8 python3 /home/pi/hybrid_ocr_benchmark.py
"""

import os
import sys
import csv
import time
import re
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2

from fast_plate_ocr import LicensePlateRecognizer


# ============================================================================
# CONFIG
# ============================================================================

PLATE_CROPS_DIR = Path("/home/pi/targhe_europee")
RESULTS_DIR = Path("/home/pi/hybrid_ocr_benchmark_results")

# Google Cloud Vertex AI - ADC (Application Default Credentials)
GCP_PROJECT_ID = "ngs2026"
GCP_LOCATION = "europe-west1"

# Modello Gemini per il fallback (solo Flash come da richiesta)
GEMINI_MODEL = "gemini-2.5-flash"

# Modello fast-plate-ocr primario
FASTPLATE_MODEL = "cct-xs-v1-global-model"

# Parallelismo chiamate Gemini
N_PARALLEL_GEMINI = 5

# Limita numero immagini (None = tutte)
MAX_IMAGES = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG"}


# ============================================================================
# UTILITY
# ============================================================================

def clean_text(text: str) -> str:
    """Tieni solo A-Z e 0-9 maiuscoli."""
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if len(a) == 0: return len(b)
    if len(b) == 0: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def compute_cer(gt: str, pred: str) -> float:
    if not gt:
        return 1.0 if pred else 0.0
    return levenshtein(gt, pred) / len(gt)


def char_level_confusion(gt: str, pred: str) -> list:
    """Coppie (gt_char, pred_char) di confusione carattere per carattere."""
    pairs = []
    m, n = len(gt), len(pred)
    if m == 0 or n == 0:
        return pairs
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if gt[i-1] == pred[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    i, j = m, n
    while i > 0 and j > 0:
        if gt[i-1] == pred[j-1]:
            i -= 1; j -= 1
        else:
            sub, ins, dele = dp[i-1][j-1], dp[i][j-1], dp[i-1][j]
            best = min(sub, ins, dele)
            if best == sub:
                pairs.append((gt[i-1], pred[j-1]))
                i -= 1; j -= 1
            elif best == ins:
                j -= 1
            else:
                i -= 1
    return pairs


# ============================================================================
# GROUND TRUTH (dal filename)
# ============================================================================

def load_ground_truth(images_dir: Path) -> dict:
    gt = {}
    for img_path in images_dir.iterdir():
        if not (img_path.is_file() and img_path.suffix.lower() in {e.lower() for e in IMAGE_EXTS}):
            continue
        raw = img_path.stem.upper()
        text = clean_text(raw)
        if text:
            gt[img_path.name] = text
    return gt


# ============================================================================
# FAST-PLATE-OCR
# ============================================================================

class FastPlateReader:
    def __init__(self, model_name=FASTPLATE_MODEL):
        print(f"Caricamento fast-plate-ocr {model_name}...")
        self.model = LicensePlateRecognizer(model_name)
        print("  OK")

    def read(self, image_path):
        t0 = time.time()
        try:
            result = self.model.run(str(image_path))
            if isinstance(result, list) and len(result) > 0:
                pred_obj = result[0]
                text = pred_obj.plate if hasattr(pred_obj, 'plate') else str(pred_obj)
            elif isinstance(result, str):
                text = result
            else:
                text = str(result)
        except Exception as e:
            print(f"  [fast-plate-ocr] errore su {Path(image_path).name}: {e}")
            text = ""
        elapsed = time.time() - t0
        return clean_text(text), elapsed * 1000


# ============================================================================
# GEMINI VIA VERTEX AI + ADC
# ============================================================================

class GeminiReader:
    """Gemini Flash via Vertex AI con ADC."""

    OCR_PROMPT = (
        "Read the license plate in this image and respond with ONLY the plate text, "
        "uppercase, no spaces, no dashes, no punctuation, no explanation. "
        "If you cannot read it clearly, respond with the empty string. "
        "Use only letters A-Z and digits 0-9."
    )

    def __init__(self, model_name=GEMINI_MODEL, project_id=GCP_PROJECT_ID, location=GCP_LOCATION):
        from google import genai
        from google.genai import types
        print(f"Inizializzazione Gemini client (Vertex AI, project={project_id}, "
              f"location={location})...")
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        self.types = types
        self.model_name = model_name
        # Disabilita thinking per Flash (evita chain-of-thought leakage)
        self.generation_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        print("  OK")

    def read(self, image_path):
        t0 = time.time()
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            ext = Path(image_path).suffix.lower()
            mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    self.types.Part.from_bytes(data=img_bytes, mime_type=mime),
                    self.OCR_PROMPT,
                ],
                config=self.generation_config,
            )
            text = (response.text or "").strip().upper()
            text = clean_text(text)
            # Safety net: output troppo lungo -> prendi gli ultimi 10 char
            if len(text) > 15:
                text = text[-10:]
        except Exception as e:
            print(f"  [Gemini] errore su {Path(image_path).name}: {e}")
            text = ""
        elapsed = time.time() - t0
        return text, elapsed * 1000


# ============================================================================
# RUNNER
# ============================================================================

def run_fastplate(image_files, gt_map, output_csv):
    print(f"\n--- Fase 1: fast-plate-ocr {FASTPLATE_MODEL} su tutte le immagini ---")
    reader = FastPlateReader()
    results = []
    for idx, img_path in enumerate(image_files):
        gt = gt_map.get(img_path.name, "")
        pred, time_ms = reader.read(img_path)
        exact = (pred == gt and gt != "")
        cer = compute_cer(gt, pred)
        results.append({
            'filename': img_path.name,
            'gt': gt,
            'pred': pred,
            'exact_match': exact,
            'cer': round(cer, 4),
            'time_ms': round(time_ms, 1),
        })
        if (idx + 1) % 100 == 0 or (idx + 1) == len(image_files):
            print(f"  fast-plate-ocr {idx+1}/{len(image_files)}")
    _write_csv(output_csv, results)
    return results


def _gemini_one(reader, img_path, gt, fp_pred):
    pred, time_ms = reader.read(img_path)
    return {
        'filename': img_path.name,
        'gt': gt,
        'fastplate_pred': fp_pred,
        'gemini_pred': pred,
        'gemini_correct': (pred == gt and gt != ""),
        'gemini_cer': round(compute_cer(gt, pred), 4),
        'gemini_time_ms': round(time_ms, 1),
    }


def run_gemini_fallback(failed_cases, image_dir, output_csv):
    """Chiama Gemini solo sui falliti di fast-plate-ocr (in parallelo)."""
    if not failed_cases:
        print("\nNessun fallimento di fast-plate-ocr: salto Gemini.")
        return []
    print(f"\n--- Fase 2: Gemini {GEMINI_MODEL} sui {len(failed_cases)} falliti ---")
    reader = GeminiReader()
    results = []
    with ThreadPoolExecutor(max_workers=N_PARALLEL_GEMINI) as executor:
        futures = {
            executor.submit(_gemini_one, reader,
                            image_dir / fc['filename'], fc['gt'], fc['pred']): fc
            for fc in failed_cases
        }
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 20 == 0 or completed == len(failed_cases):
                print(f"  Gemini {completed}/{len(failed_cases)}")
    results.sort(key=lambda r: r['filename'])
    _write_csv(output_csv, results)
    return results


def _write_csv(path, results):
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV salvato: {path}")


# ============================================================================
# ANALISI
# ============================================================================

def compute_summary(fp_results, gemini_results):
    total = len(fp_results)
    fp_correct = sum(1 for r in fp_results if r['exact_match'])
    fp_failed = total - fp_correct

    gemini_called = len(gemini_results)
    gemini_recovered = sum(1 for r in gemini_results if r['gemini_correct'])
    gemini_failed = gemini_called - gemini_recovered

    # Pipeline ibrida finale: per ogni immagine, vince la lettura giusta tra fp e gemini (se chiamato)
    # In pratica: se fp era corretto -> ok. Se fp falliva ma gemini recupera -> ok.
    hybrid_correct = fp_correct + gemini_recovered
    hybrid_failed = total - hybrid_correct

    summary = {
        'total_images': total,
        'fastplate_correct': fp_correct,
        'fastplate_failed': fp_failed,
        'fastplate_accuracy': round(fp_correct / total * 100, 2),
        'gemini_called_on': gemini_called,
        'gemini_recovered': gemini_recovered,
        'gemini_failed': gemini_failed,
        'gemini_recovery_rate': round(gemini_recovered / gemini_called * 100, 2) if gemini_called > 0 else 0,
        'hybrid_correct': hybrid_correct,
        'hybrid_failed': hybrid_failed,
        'hybrid_accuracy': round(hybrid_correct / total * 100, 2),
        'improvement_pct_points': round((hybrid_correct - fp_correct) / total * 100, 2),
    }
    return summary


def analyze_residual_errors(fp_results, gemini_results, output_path):
    """Confusioni carattere sui casi ancora falliti DOPO il fallback Gemini."""
    # Mappa filename -> gemini result
    gemini_map = {r['filename']: r for r in gemini_results}

    counter = Counter()
    for r in fp_results:
        if r['exact_match']:
            continue
        # Se Gemini ha recuperato, l'errore non e' piu' un errore della pipeline ibrida
        gemini_r = gemini_map.get(r['filename'])
        if gemini_r and gemini_r['gemini_correct']:
            continue
        # Errore residuo della pipeline ibrida: prendi la migliore predizione disponibile
        # (cioe' quella di Gemini se chiamato, altrimenti quella di fast-plate-ocr)
        final_pred = gemini_r['gemini_pred'] if gemini_r else r['pred']
        for p in char_level_confusion(r['gt'], final_pred):
            counter[p] += 1

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gt_char", "pred_char", "occurrences"])
        for (gt_c, pred_c), count in counter.most_common(30):
            writer.writerow([gt_c, pred_c, count])
    print(f"  Confusioni residue salvate: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not PLATE_CROPS_DIR.exists():
        print(f"ERRORE: cartella crop non trovata: {PLATE_CROPS_DIR}")
        sys.exit(1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BENCHMARK IBRIDO fast-plate-ocr + Gemini Flash (fallback)")
    print("=" * 70)
    print(f"Crops dir:      {PLATE_CROPS_DIR}")
    print(f"Results dir:    {RESULTS_DIR}")
    print(f"fast-plate-ocr: {FASTPLATE_MODEL}")
    print(f"Gemini model:   {GEMINI_MODEL}")
    print(f"GCP project:    {GCP_PROJECT_ID} ({GCP_LOCATION})")
    print()

    # Ground truth
    print("Caricamento GT dai filename...")
    gt_map = load_ground_truth(PLATE_CROPS_DIR)
    print(f"  {len(gt_map)} GT caricate")

    # Lista immagini
    image_files = sorted([
        p for p in PLATE_CROPS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {e.lower() for e in IMAGE_EXTS}
    ])
    image_files = [p for p in image_files if p.name in gt_map]
    if MAX_IMAGES:
        image_files = image_files[:MAX_IMAGES]
    print(f"  Immagini da processare: {len(image_files)}")

    # FASE 1: fast-plate-ocr su tutto
    fp_csv = RESULTS_DIR / "fastplate_results.csv"
    fp_results = run_fastplate(image_files, gt_map, fp_csv)

    # Identifica i falliti
    failed = [r for r in fp_results if not r['exact_match']]
    print(f"\nfast-plate-ocr ha sbagliato su {len(failed)}/{len(fp_results)} immagini "
          f"({len(failed)/len(fp_results)*100:.1f}%)")

    # FASE 2: Gemini solo sui falliti
    gemini_csv = RESULTS_DIR / "gemini_fallback_results.csv"
    gemini_results = run_gemini_fallback(failed, PLATE_CROPS_DIR, gemini_csv)

    # SUMMARY
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = compute_summary(fp_results, gemini_results)
    for k, v in summary.items():
        print(f"  {k:<30} {v}")

    summary_csv = RESULTS_DIR / "hybrid_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])
    print(f"\nSummary salvato: {summary_csv}")

    # ANALISI ERRORI RESIDUI
    if gemini_results:
        residual_csv = RESULTS_DIR / "confusion_pairs_hybrid.csv"
        analyze_residual_errors(fp_results, gemini_results, residual_csv)

    print("\nBenchmark completato.")


if __name__ == "__main__":
    main()
