#!/usr/bin/env python3
"""
Benchmark preprocessing pre-OCR su EasyOCR, run completo su 735 immagini.

Solo 4 varianti dello screening precedente, le migliori per accuratezza/velocita':
  1. baseline_raw           - immagine come arriva
  2. grayscale              - conversione a scala di grigi
  3. gray_clahe             - grayscale + CLAHE (contrasto adattivo)
  4. gray_clahe_sharpen     - CLAHE + unsharp masking

3 METRICHE CALCOLATE per ogni variante:
  - exact_match            : pred == gt (rigoroso)
  - substring_match        : gt e' contenuta in pred (cattura targhe lette + rumore intorno)
  - exact_match_postproc   : pred == gt DOPO post-processing automatico

POST-PROCESSING (agnostico, no GT-dependent):
  1. Rimuove codici paese noti all'inizio (IRL, NL, DE, IT, F, GB, ecc.)
  2. Estrae blocchi alfanumerici lunghi 5-9 caratteri
  3. Sceglie il blocco con miglior "punteggio targa": mix di lettere+cifre,
     lunghezza vicina a 7, posizione iniziale

OUTPUT:
  RESULTS_DIR/
    preprocessing_comparison.csv      - metriche aggregate per variante
    per_variant/
      <variant>_results.csv           - dettaglio per immagine, con colonne:
                                        filename, gt, pred, pred_postproc,
                                        exact_match, substring_match,
                                        exact_match_postproc, cer, time_ms
    examples/
      <variant>/                      - 5 esempi del preprocessing applicato

LANCIO:
  PYTHONIOENCODING=utf-8 LANG=C.UTF-8 python3 /home/pi/benchmark_preprocessing_final.py
"""

import os
import sys
import csv
import re
import time
from pathlib import Path

import numpy as np
import cv2


# ============================================================================
# CONFIGURAZIONE
# ============================================================================

PLATE_CROPS_DIR = Path("/home/pi/targhe_europee")
RESULTS_DIR = Path("/home/pi/preprocessing_final_results")

MAX_IMAGES = None         # None = tutte (735)

EASYOCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

N_EXAMPLES_PER_VARIANT = 5

# CLAHE parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Codici paese europei (banda blu sinistra targhe UE)
# Ordinati per lunghezza decrescente per matching greedy
COUNTRY_CODES = sorted([
    "IRL", "GBM", "GB", "NL", "DE", "FR", "IT", "ES", "PT", "BE", "AT", "CH",
    "DK", "SE", "FI", "NO", "PL", "CZ", "SK", "HU", "RO", "BG", "GR", "HR",
    "SI", "EE", "LV", "LT", "LU", "MT", "CY", "IS", "TR",
    "F", "D", "E", "I", "P", "B", "A", "S", "N", "L", "H", "M",
], key=lambda x: -len(x))

# Range lunghezza targhe (dal dataset osservato: 6-9 caratteri)
MIN_PLATE_LEN = 5
MAX_PLATE_LEN = 9
TYPICAL_PLATE_LEN = 7


# ============================================================================
# GROUND TRUTH
# ============================================================================

def load_ground_truth_from_filenames(images_dir: Path) -> dict:
    gt = {}
    for img_path in images_dir.iterdir():
        if not (img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS):
            continue
        raw = img_path.stem.upper()
        text = "".join(c for c in raw if c.isalnum())
        if text:
            gt[img_path.name] = text
    return gt


# ============================================================================
# METRICHE
# ============================================================================

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


# ============================================================================
# POST-PROCESSING (estrazione targa da stringa rumorosa)
# ============================================================================

def remove_country_prefix(text: str) -> str:
    """Rimuove un codice paese in testa, se presente.
    Usa matching greedy (codici lunghi prima)."""
    if not text:
        return text
    for code in COUNTRY_CODES:
        if text.startswith(code) and len(text) > len(code):
            # Verifica che dopo il codice ci siano almeno 4 caratteri (per non
            # cannibalizzare una targa che inizia per caso con quelle lettere)
            remainder = text[len(code):]
            if len(remainder) >= 4:
                return remainder
    return text


def score_plate_candidate(candidate: str, position: int) -> float:
    """
    Punteggio per scegliere il miglior candidato targa.
    Criteri (somma pesata):
      + lunghezza vicina a 7 caratteri
      + mix di lettere e cifre (almeno una di ciascuna)
      + posizione iniziale nella stringa (le targhe sono di solito all'inizio)
    """
    if not candidate:
        return 0.0

    score = 0.0
    has_letter = any(c.isalpha() for c in candidate)
    has_digit = any(c.isdigit() for c in candidate)

    # Criterio 1: mix lettere + cifre (le targhe non sono mai monotipiche)
    if has_letter and has_digit:
        score += 10.0
    elif has_letter or has_digit:
        score += 3.0

    # Criterio 2: lunghezza vicina a 7 (penalita' per distanza da TYPICAL_PLATE_LEN)
    length_penalty = abs(len(candidate) - TYPICAL_PLATE_LEN)
    score += max(0, 5.0 - length_penalty)

    # Criterio 3: posizione iniziale (le targhe sono solitamente prima del rumore)
    score += max(0, 3.0 - position * 0.5)

    return score


def postprocess_plate(raw_pred: str) -> str:
    """
    Estrae la stringa "piu' simile a una targa" dalla predizione OCR rumorosa.
    Approccio agnostico: non usa la GT, non assume struttura specifica.
    """
    if not raw_pred:
        return ""

    # Step 1: rimuovi codice paese in testa
    cleaned = remove_country_prefix(raw_pred)

    # Step 2: estrai tutti i candidati di lunghezza plausibile.
    # Visto che abbiamo gia' filtrato A-Z e 0-9, la stringa e' un unico blocco
    # alfanumerico. Estraiamo tutte le sliding windows di lunghezza 5-9.
    candidates = []
    n = len(cleaned)
    if n == 0:
        return ""

    # Se la stringa e' gia' di lunghezza plausibile, e' lei la candidata principale
    if MIN_PLATE_LEN <= n <= MAX_PLATE_LEN:
        return cleaned

    # Altrimenti, sliding window
    for length in range(MIN_PLATE_LEN, MAX_PLATE_LEN + 1):
        if length > n:
            break
        for start in range(n - length + 1):
            cand = cleaned[start:start + length]
            score = score_plate_candidate(cand, start)
            candidates.append((cand, score))

    if not candidates:
        # Fallback: ritorna i primi 7 caratteri della stringa pulita
        return cleaned[:TYPICAL_PLATE_LEN]

    # Step 3: scegli il candidato con punteggio piu' alto
    candidates.sort(key=lambda c: -c[1])
    return candidates[0][0]


# ============================================================================
# FUNZIONI DI PREPROCESSING (solo le 4 selezionate)
# ============================================================================

def _to_bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def preprocess_baseline_raw(img_bgr):
    return img_bgr.copy()


def preprocess_grayscale(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return _to_bgr(gray)


def preprocess_gray_clahe(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    enhanced = clahe.apply(gray)
    return _to_bgr(enhanced)


def preprocess_gray_clahe_sharpen(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return _to_bgr(sharpened)


VARIANTS = [
    ("baseline_raw",       preprocess_baseline_raw,
     "immagine originale senza preprocessing"),
    ("grayscale",          preprocess_grayscale,
     "conversione a scala di grigi"),
    ("gray_clahe",         preprocess_gray_clahe,
     "grayscale + CLAHE (contrasto adattivo)"),
    ("gray_clahe_sharpen", preprocess_gray_clahe_sharpen,
     "CLAHE + unsharp masking"),
]


# ============================================================================
# OCR ENGINE
# ============================================================================

class EasyOCRReader:
    def __init__(self, allowlist=EASYOCR_ALLOWLIST):
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        self.allowlist = allowlist

    def read_array(self, img_bgr):
        t0 = time.time()
        results = self.reader.readtext(img_bgr, allowlist=self.allowlist,
                                       detail=0, paragraph=False)
        t1 = time.time()
        text = "".join(results).upper().replace(" ", "")
        return text, t1 - t0


# ============================================================================
# RUNNER PER VARIANTE
# ============================================================================

def run_variant(variant_name, preprocess_fn, description, image_files, gt_map,
                ocr_reader, examples_dir, per_variant_dir):
    print(f"\n{'=' * 60}")
    print(f"VARIANTE: {variant_name}")
    print(f"  {description}")
    print(f"{'=' * 60}")

    variant_examples_dir = examples_dir / variant_name
    variant_examples_dir.mkdir(parents=True, exist_ok=True)

    results = []
    saved_examples = 0

    for idx, img_path in enumerate(image_files):
        gt = gt_map.get(img_path.name, "")
        if not gt:
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [WARN] impossibile leggere: {img_path.name}")
            continue

        # Preprocessing
        t_pre0 = time.time()
        try:
            img_processed = preprocess_fn(img_bgr)
        except Exception as e:
            print(f"  [ERROR] preprocessing fallito su {img_path.name}: {e}")
            continue
        t_pre1 = time.time()
        pre_time_ms = (t_pre1 - t_pre0) * 1000

        # OCR
        try:
            pred, ocr_seconds = ocr_reader.read_array(img_processed)
        except Exception as e:
            print(f"  [ERROR] OCR fallito su {img_path.name}: {e}")
            continue
        ocr_time_ms = ocr_seconds * 1000

        # Post-processing
        t_post0 = time.time()
        pred_postproc = postprocess_plate(pred)
        t_post1 = time.time()
        post_time_ms = (t_post1 - t_post0) * 1000

        # Metriche
        cer = compute_cer(gt, pred)
        exact = (gt == pred and gt != "")
        substring = (gt in pred) if (gt and pred) else False
        exact_postproc = (gt == pred_postproc and gt != "")
        cer_postproc = compute_cer(gt, pred_postproc)

        results.append({
            'filename': img_path.name,
            'gt': gt,
            'pred': pred,
            'pred_postproc': pred_postproc,
            'exact_match': exact,
            'substring_match': substring,
            'exact_match_postproc': exact_postproc,
            'cer': round(cer, 4),
            'cer_postproc': round(cer_postproc, 4),
            'preprocess_ms': round(pre_time_ms, 2),
            'ocr_ms': round(ocr_time_ms, 1),
            'postproc_ms': round(post_time_ms, 3),
            'total_ms': round(pre_time_ms + ocr_time_ms + post_time_ms, 1),
        })

        # Esempi visivi
        if saved_examples < N_EXAMPLES_PER_VARIANT:
            safe_pred = pred if pred else "EMPTY"
            example_path = variant_examples_dir / f"{idx:04d}_{img_path.stem}_GT_{gt}_PRED_{safe_pred}.png"
            cv2.imwrite(str(example_path), img_processed)
            saved_examples += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_files):
            done_exact = sum(1 for r in results if r['exact_match'])
            done_substr = sum(1 for r in results if r['substring_match'])
            done_post = sum(1 for r in results if r['exact_match_postproc'])
            print(f"  {idx+1}/{len(image_files)}  -  "
                  f"exact: {done_exact}  substring: {done_substr}  "
                  f"post: {done_post}")

    # Checkpoint CSV per variante
    csv_path = per_variant_dir / f"{variant_name}_results.csv"
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    print(f"  Checkpoint salvato in: {csv_path}")

    return results


# ============================================================================
# AGGREGAZIONE
# ============================================================================

def aggregate_variant(variant_name, description, results):
    if not results:
        return {
            'variant': variant_name,
            'description': description,
            'num_images': 0,
        }
    total = len(results)
    exact_count = sum(1 for r in results if r['exact_match'])
    substring_count = sum(1 for r in results if r['substring_match'])
    exact_postproc_count = sum(1 for r in results if r['exact_match_postproc'])

    avg_cer = float(np.mean([r['cer'] for r in results]))
    median_cer = float(np.median([r['cer'] for r in results]))
    avg_cer_postproc = float(np.mean([r['cer_postproc'] for r in results]))
    median_cer_postproc = float(np.median([r['cer_postproc'] for r in results]))

    avg_pre = float(np.mean([r['preprocess_ms'] for r in results]))
    avg_ocr = float(np.mean([r['ocr_ms'] for r in results]))
    avg_post = float(np.mean([r['postproc_ms'] for r in results]))
    avg_total = float(np.mean([r['total_ms'] for r in results]))

    empty_preds = sum(1 for r in results if not r['pred'])

    return {
        'variant': variant_name,
        'description': description,
        'num_images': total,
        'exact_match': exact_count,
        'exact_match_rate': round(exact_count / total, 4),
        'substring_match': substring_count,
        'substring_match_rate': round(substring_count / total, 4),
        'exact_match_postproc': exact_postproc_count,
        'exact_match_postproc_rate': round(exact_postproc_count / total, 4),
        'avg_cer': round(avg_cer, 4),
        'median_cer': round(median_cer, 4),
        'avg_cer_postproc': round(avg_cer_postproc, 4),
        'median_cer_postproc': round(median_cer_postproc, 4),
        'empty_predictions': empty_preds,
        'avg_preprocess_ms': round(avg_pre, 2),
        'avg_ocr_ms': round(avg_ocr, 1),
        'avg_postproc_ms': round(avg_post, 3),
        'avg_total_ms': round(avg_total, 1),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not PLATE_CROPS_DIR.exists():
        print(f"ERRORE: cartella non trovata: {PLATE_CROPS_DIR}")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    examples_dir = RESULTS_DIR / "examples"
    examples_dir.mkdir(exist_ok=True)
    per_variant_dir = RESULTS_DIR / "per_variant"
    per_variant_dir.mkdir(exist_ok=True)

    print(f"Caricamento ground truth da filename in: {PLATE_CROPS_DIR}")
    gt_map = load_ground_truth_from_filenames(PLATE_CROPS_DIR)
    print(f"  Ground truth caricate: {len(gt_map)} entries")

    image_files = sorted([
        p for p in PLATE_CROPS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])
    image_files = [p for p in image_files if p.name in gt_map]
    print(f"  Immagini con GT: {len(image_files)}")

    if MAX_IMAGES:
        image_files = image_files[:MAX_IMAGES]
        print(f"  Limitato a: {len(image_files)} immagini (MAX_IMAGES={MAX_IMAGES})")

    if not image_files:
        print("ERRORE: nessuna immagine con GT.")
        sys.exit(1)

    print("\nInizializzazione EasyOCR...")
    ocr_reader = EasyOCRReader()
    print("EasyOCR pronto.")

    all_summaries = []
    t_start_total = time.time()

    for variant_name, preprocess_fn, description in VARIANTS:
        t_v0 = time.time()
        results = run_variant(
            variant_name, preprocess_fn, description,
            image_files, gt_map,
            ocr_reader, examples_dir, per_variant_dir
        )
        t_v1 = time.time()
        print(f"  Variante completata in {(t_v1 - t_v0)/60:.1f} minuti")

        summary = aggregate_variant(variant_name, description, results)
        all_summaries.append(summary)

    t_end_total = time.time()
    total_minutes = (t_end_total - t_start_total) / 60
    print(f"\n{'=' * 60}")
    print(f"TUTTI I BENCHMARK COMPLETATI in {total_minutes:.1f} minuti")
    print(f"{'=' * 60}")

    # Ordina per exact_match_postproc_rate (la metrica piu' significativa post-processing)
    all_summaries.sort(key=lambda s: -s.get('exact_match_postproc_rate', 0))

    summary_csv = RESULTS_DIR / "preprocessing_comparison.csv"
    if all_summaries:
        keys = list(all_summaries[0].keys())
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_summaries)
        print(f"\nReport comparativo salvato in: {summary_csv}")

    print("\nCLASSIFICA FINALE (ordinata per Exact Match dopo post-processing):")
    print("=" * 100)
    print(f"{'Variante':<22s} | {'Exact':>8s} | {'Substring':>10s} | {'PostProc':>9s} | "
          f"{'CER':>6s} | {'CER post':>8s} | {'Tempo':>8s}")
    print("-" * 100)
    for s in all_summaries:
        print(f"{s['variant']:<22s} | "
              f"{s.get('exact_match_rate', 0)*100:>7.2f}% | "
              f"{s.get('substring_match_rate', 0)*100:>9.2f}% | "
              f"{s.get('exact_match_postproc_rate', 0)*100:>8.2f}% | "
              f"{s.get('avg_cer', 0):>6.3f} | "
              f"{s.get('avg_cer_postproc', 0):>8.3f} | "
              f"{s.get('avg_total_ms', 0):>6.0f}ms")
    print("=" * 100)
    print("\nLegenda metriche:")
    print("  Exact      : pred == gt (rigoroso, prima del post-processing)")
    print("  Substring  : gt e' contenuta in pred (cattura targhe lette + rumore)")
    print("  PostProc   : pred == gt DOPO post-processing automatico")
    print("  CER        : Character Error Rate medio sulla pred raw")
    print("  CER post   : Character Error Rate medio dopo post-processing")

    print(f"\nEsempi visivi: {examples_dir}")
    print(f"CSV per variante: {per_variant_dir}")


if __name__ == "__main__":
    main()
