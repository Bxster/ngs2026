#!/usr/bin/env python3
"""
Benchmark fast-plate-ocr full su crop di targhe europee.

Setup:
  - 3 modelli, ciascuno con input nativo (libreria gestisce internamente)
  - Niente forzature raw/gray: passiamo il path del file e lasciamo
    alla libreria la conversione color mode corretta per ogni modello
  - Niente post-processing

Output:
  - CSV con tutte le predizioni
  - Riepilogo exact% / CER medio / tempo medio per modello
  - File errori_per_modello.csv con i casi falliti (per analisi)

Lancio:
  PYTHONIOENCODING=utf-8 LANG=C.UTF-8 python3 fast_plate_benchmark_full.py
"""

import csv
import re
import time
from pathlib import Path
from collections import Counter

from fast_plate_ocr import LicensePlateRecognizer


# ============================================================================
# CONFIG
# ============================================================================

DATASET_DIR = Path("/home/pi/targhe_europee")
RESULTS_DIR = Path("/home/pi/fast_plate_results_full")
RESULTS_CSV = RESULTS_DIR / "results_full.csv"
ERRORS_CSV = RESULTS_DIR / "errors_per_model.csv"

MODELS = [
    "cct-xs-v1-global-model",
    "cct-s-v2-global-model",
    "european-plates-mobile-vit-v2-model",
]


# ============================================================================
# UTILITY
# ============================================================================

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())


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


def run_inference(model, img_path):
    """
    Esegue inferenza passando il path del file.
    fast-plate-ocr gestisce internamente la conversione color mode richiesta.
    Ritorna (testo_pulito, tempo_ms, error_str).
    """
    t0 = time.time()
    try:
        result = model.run(str(img_path))
    except Exception as e:
        return ('', (time.time() - t0) * 1000, str(e))
    elapsed_ms = (time.time() - t0) * 1000

    if isinstance(result, list) and len(result) > 0:
        pred_obj = result[0]
        if hasattr(pred_obj, 'plate'):
            text = pred_obj.plate
        else:
            text = str(pred_obj)
    elif isinstance(result, str):
        text = result
    else:
        text = str(result)

    return (clean_text(text), elapsed_ms, '')


# ============================================================================
# MAIN
# ============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FAST-PLATE-OCR BENCHMARK FULL (735 immagini)")
    print("=" * 70)
    print(f"Dataset:  {DATASET_DIR}")
    print(f"Modelli:  {len(MODELS)}")
    for m in MODELS:
        print(f"          - {m}")
    print()

    image_files = sorted([
        p for p in DATASET_DIR.iterdir()
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    ])
    if not image_files:
        print(f"ERRORE: nessuna immagine in {DATASET_DIR}")
        return
    print(f"Trovate {len(image_files)} immagini")
    print()

    # Carica i 3 modelli
    print("[1/2] Caricamento modelli...")
    models = {}
    for name in MODELS:
        print(f"  {name}...")
        try:
            models[name] = LicensePlateRecognizer(name)
            print(f"    OK")
        except Exception as e:
            print(f"    ERRORE: {e}")
    print()

    # Inferenza
    print(f"[2/2] Inferenza su {len(image_files)} immagini con {len(models)} modelli...")
    rows = []
    stats = {m: {'tp': 0, 'cer_sum': 0.0, 'time_sum': 0.0, 'n': 0, 'err': 0}
             for m in models.keys()}
    # Errori carattere-per-carattere per analisi successiva
    error_rows = []

    t_start = time.time()
    for idx, img_path in enumerate(image_files):
        gt = clean_text(img_path.stem)
        row = {'file': img_path.name, 'gt': gt}
        for model_name, model in models.items():
            pred, t_ms, err = run_inference(model, img_path)
            exact = (pred == gt)
            c = cer(pred, gt)
            row[f"{model_name}__pred"] = pred
            row[f"{model_name}__exact"] = exact
            row[f"{model_name}__cer"] = c
            row[f"{model_name}__time_ms"] = t_ms

            stats[model_name]['tp'] += int(exact)
            stats[model_name]['cer_sum'] += c
            stats[model_name]['time_sum'] += t_ms
            stats[model_name]['n'] += 1
            if err:
                stats[model_name]['err'] += 1

            if not exact and not err:
                error_rows.append({
                    'file': img_path.name,
                    'model': model_name,
                    'gt': gt,
                    'pred': pred,
                    'cer': c,
                })

        rows.append(row)
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta = (len(image_files) - idx - 1) / rate if rate > 0 else 0
            print(f"  {idx + 1}/{len(image_files)}  rate {rate:.1f} img/s  eta {eta:.0f}s")

    print()
    print(f"Done in {time.time() - t_start:.1f}s")
    print()

    # Salva CSV principale
    print("Salvataggio CSV...")
    columns = ['file', 'gt']
    for m in MODELS:
        columns += [f"{m}__pred", f"{m}__exact", f"{m}__cer", f"{m}__time_ms"]
    with open(RESULTS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            r_fmt = {}
            for k, v in r.items():
                if isinstance(v, float):
                    r_fmt[k] = f"{v:.4f}"
                else:
                    r_fmt[k] = v
            w.writerow(r_fmt)
    print(f"  {RESULTS_CSV}")

    # Salva errori per modello
    if error_rows:
        with open(ERRORS_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['file', 'model', 'gt', 'pred', 'cer'])
            w.writeheader()
            for er in error_rows:
                er_fmt = dict(er)
                er_fmt['cer'] = f"{er['cer']:.4f}"
                w.writerow(er_fmt)
        print(f"  {ERRORS_CSV}  ({len(error_rows)} errori totali)")
    print()

    # Riepilogo
    print("=" * 70)
    print(f"RIEPILOGO (n={len(rows)} immagini)")
    print("=" * 70)
    print(f"{'Modello':<42} {'Exact':>9} {'CER':>8} {'Time(ms)':>10} {'Err':>5}")
    print("-" * 80)
    for m in MODELS:
        s = stats[m]
        n = s['n']
        if n == 0:
            continue
        exact_pct = s['tp'] / n * 100
        cer_avg = s['cer_sum'] / n
        time_avg = s['time_sum'] / n
        print(f"{m:<42} {exact_pct:>8.1f}% {cer_avg:>8.3f} {time_avg:>10.1f} {s['err']:>5}")

    print()
    print(f"Output: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
