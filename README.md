# NGS2026 — Sistema di Rilevamento Veicoli Edge AI

Progetto sviluppato per **New Generation Sensors** — sistema di rilevamento veicoli e lettura targhe su hardware embedded.

## Hardware

| Componente | Dettaglio |
|------------|-----------|
| Raspberry Pi 5 | 16GB RAM |
| Hailo-8 AI HAT+ | 26 TOPS |
| Sony IMX500 AI Camera | 12MP, inferenza on-chip |

## Architettura pipeline (attuale)

```
Camera / video input
       |
       v
 Hailo-8 HAT+ (VDevice unico, multi-network)
       |
   YOLOv11s ──────────────► bbox veicoli (6 classi)
       |
       v
   YOLOv11n v1n ──────────► bbox targhe sui crop veicolo
       |
       v
 ByteTrack (CPU) ─────────► tracker_id stabili
       |
       v
 fast-plate-ocr (CPU, ONNX) ► testo targa
       |
       v
 Majority voting + merge by plate
       |
       v
 Output: JSON con conteggio veicoli, ID, classe, targa
```

**Output finale per veicolo:** `{ tracker_id, classe, targa, n_frames_seen, n_plate_reads }`

L'integrazione IMX500 (nano YOLO on-chip per detection veicoli) è **sospesa**: la pipeline attuale gira interamente su Hailo-8 + CPU RPi5, in real-time con frame skipping configurabile. L'IMX500 verrà valutata in una seconda fase di sviluppo.

## Modelli

I file dei modelli sono esclusi dal repository via `.gitignore`. La cartella `models/` contiene un `.gitkeep` come placeholder.

| File | Formato | Target | Uso attuale | Descrizione |
|------|---------|--------|-------------|-------------|
| `yolov11s.hef` | HEF | Hailo-8 | ✅ in pipeline | Classificatore veicoli (6 classi) |
| `license_plate_finetune_v1n.hef` | HEF | Hailo-8 | ✅ in pipeline | Plate detection (F1=0.955) |
| `best_nano_v2_merged.rpk` | RPK | IMX500 | ⏸️ sospeso | Detector veicoli on-chip (per fase 2) |
| `yolo11n_uc3m.hef` | HEF | Hailo-8 | ❌ legacy | Plate detection alternativo (F1=0.849) |
| `tiny_yolov4_license_plates.hef` | HEF | Hailo-8 | ❌ legacy | Plate detection v1 (F1=0.638, sostituito) |

### Classi veicoli (YOLOv11s + YOLOv11n)

`bus` · `car` · `pickup` · `truck` · `van` · `motorcycle`

### Dataset di training

Dataset merged: Kaggle (`hammadjavaid/vehicle-object-detection-dataset-5-classes`) + motorcycle da OpenImages V7.

Dataset pubblicato: [NGS2026 Vehicle Dataset](https://www.kaggle.com/datasets/bxster/ngs2026-vehicle-dataset)

| Split | Immagini |
|-------|----------|
| Train | 30.508 |
| Val | 8.331 |
| Test | 4.316 |

### Risultati vehicle detection

| Modello | mAP50 (test) | mAP50-95 (test) |
|---------|-------------|-----------------|
| YOLOv11s small (Hailo-8) | 0.900 | 0.700 |
| YOLOv11n nano (IMX500) | 0.852 | 0.652 |

### Risultati plate detection (su 1474 ground truth)

Benchmark su dataset estraneo al training (Persian Car Plates, 1381 immagini).

| Modello | F1 | Precision | Recall | mAP50 | FPS Hailo-8 |
|---------|----|-----------|--------|-------|-------------|
| `license_plate_finetune_v1n.hef` ✅ | **0.955** | 0.917 | 0.998 | 0.994 | 108.8 |
| `yolo11n_uc3m.hef` | 0.849 | 0.749 | 0.980 | 0.956 | 91.1 |
| `tiny_yolov4_license_plates.hef` ❌ | 0.638 | 0.615 | 0.662 | 0.602 | 447.8 |

### Risultati OCR (su 735 targhe europee miste)

| Tecnologia | Variante | Exact match | Tempo/crop | On-device |
|------------|----------|-------------|------------|-----------|
| **fast-plate-ocr** `cct-xs-v1` ✅ | RGB | **86.0%** | 21 ms | ✅ |
| fast-plate-ocr `cct-s-v2` | RGB | 86.8% | 128 ms | ✅ |
| Gemini 2.5 Pro (pilota 50 img) | API | 86.0% | 6.2 s | ❌ |
| Gemini 2.5 Flash | API | 81.2% | 0.98 s | ❌ |
| EasyOCR ❌ | raw | 16.7% | ~1000 ms | ✅ |

**Su sample italiano UniData (10 immagini): fast-plate-ocr cct-xs-v1 → 10/10 = 100% exact match raw.**

## Struttura repository

```
ngs2026/
├── models/                          # Cartella modelli (file esclusi da .gitignore)
│   └── .gitkeep
├── docs/                            # Reports tecnici (ex "Memoria progetto")
│   ├── Report_OCR_Finale.pdf
│   ├── NGS2026_Report_Tecnico.pdf
│   ├── NGS2026_Report_Preprocessing_OCR_v2.pdf
│   ├── Report_Confronto_PT_vs_HEF.pdf
│   ├── NGS2026_Pipeline_v2_Report.md
│   └── NGS2026_Hybrid_OCR_Test.md
├── test_images/                     # Immagini di test
├── test_nano_small/                 # Test vehicle detection (legacy)
├── test_plate_detection_recognition/  # Test plate detection + OCR (legacy)
├── ngs_pipeline_v2.py               # Pipeline batch (sequential + alternating)
├── ngs_pipeline_v3.py               # Pipeline streaming real-time-ready
├── test_pipeline_full.py            # Pipeline v1 (legacy)
├── test_video_pipeline.py           # Pipeline v1 senza tracking (legacy)
└── README.md
```

## Pipeline principali

### `ngs_pipeline_v3.py` — Streaming real-time-ready (consigliata)

Pipeline frame-per-frame, processa il video in streaming senza pre-caricamento in RAM. Predisposta per camera live.

**Caratteristiche:**
- Modalità B nativa: yolo11s + v1n caricati su singolo VDevice Hailo-8
- Voting incrementale: la targa votata si aggiorna frame per frame
- Chiusura tracker automatica: i tracker_id non visti da N frame vengono "committati" e rimossi dal voting attivo
- Frame skipping configurabile (`FRAME_SKIP`): processa 1 frame ogni N del video sorgente
- Merge by plate con timeout temporale: gestisce occlusioni e transiti distinti

**Performance** (test su video 715 frame, 27 secondi, scenario stress urbano):

| FRAME_SKIP | Tempo totale | FPS effettivi | Real-time? |
|-----------:|-------------:|--------------:|------------|
| 1 | 56 s | 13 | No |
| 3 | 21 s | 34 | ✅ anche camera 30 fps |

### `ngs_pipeline_v2.py` — Batch (riferimento storico)

Pipeline a due passate (sequential) o alternating per-frame. Più veloce in batch, ma non adatta a real-time.

**Performance batch** (sequential, stesso video): 39 secondi totali, 18 fps.

## Setup ambiente

### Prerequisiti Raspberry Pi

```bash
sudo apt install imx500-all hailo-all
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps && sudo ./install.sh
```

### Attivazione environment Hailo

```bash
cd ~/hailo-apps
source setup_env.sh
```

### Installazione dipendenze pipeline

```bash
pip install --break-system-packages "fast-plate-ocr[onnx]" supervision opencv-python
```

## Esecuzione

### Pipeline v3 streaming (consigliata)

```bash
source ~/hailo-apps/setup_env.sh
PYTHONIOENCODING=utf-8 LANG=C.UTF-8 python3 ngs_pipeline_v3.py
```

Parametri principali in cima allo script:
- `VIDEO_PATH`: path al video di input
- `EXECUTION_MODE`: `streaming` (default) o `sequential`
- `FRAME_SKIP`: 1 = nessun skip, 2-3 per camera ad alto framerate
- `TRACKER_TIMEOUT_FRAMES`: gap di chiusura tracker (default 30)
- `MERGE_MAX_GAP_SECONDS`: gap temporale max per merge by plate (default 15)
- `WRITE_ANNOTATED_VIDEO`: `False` in produzione, `True` per debug visivo

### Pipeline v2 batch (legacy, per benchmark)

```bash
PYTHONIOENCODING=utf-8 LANG=C.UTF-8 python3 ngs_pipeline_v2.py
```

## Note tecniche

- **Hailo-8 multi-network**: la pipeline v3 carica yolo11s + v1n contemporaneamente sullo stesso VDevice (modalità "B"). Activate/deactivate sono ~0.7 ms ciascuno, niente overhead di riconfigurazione
- **fast-plate-ocr**: modello ONNX `cct-xs-v1-global-model`, ~2 MB, RGB input. Ha sostituito EasyOCR (5× più accurato, 50× più veloce, no problemi locale)
- **PaddleOCR**: non compatibile con ARM64 (segfault su RPi5)
- **Locale italiano**: il prefisso `PYTHONIOENCODING=utf-8 LANG=C.UTF-8` è necessario per evitare crash su caratteri unicode
- **Hailo coordinate order**: `y1, x1, y2, x2` (non standard, gestito esplicitamente nel parsing NMS)
- **IMX500**: l'RPK è pronto ma l'integrazione è sospesa. La pipeline attuale gira già in real-time senza l'AI on-chip della camera

## Prossimi passi

- Validazione su scenario reale (camera a 2-3m, traffico italiano)
- Benchmark fast-plate-ocr su dataset corposo di targhe italiane
- Eventuale integrazione IMX500 in seconda fase (architettura distribuita)
- Possibili estensioni: detection mezzi di emergenza via pattern OCR (CC/VF/EI), lookup motorizzazione, fuzzy merge by plate
