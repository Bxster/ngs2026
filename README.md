# NGS2026 — Sistema di Rilevamento Veicoli Edge AI

Progetto sviluppato per **New Generation Sensors (Firenze)** — sistema di rilevamento veicoli e lettura targhe su hardware embedded.

## Hardware

| Componente | Dettaglio |
|------------|-----------|
| Raspberry Pi 5 | 16GB RAM |
| Hailo-8 AI HAT+ | 26 TOPS |
| Sony IMX500 AI Camera | 12MP, inferenza on-chip |

## Architettura pipeline

```
IMX500 (on-chip)         Hailo-8 HAT+            RPi5 CPU
       |                       |                      |
  YOLOv11n nano          YOLOv11s small         EasyOCR
  Detection veicoli      Classificazione        Lettura targa
  bbox → RPi5            tipo veicolo           testo → output
                               |
                     tiny_yolov4_license_plates
                         Detection targa
                         bbox → EasyOCR
```

**Output finale per veicolo:** `{ id, classe, targa }`

## Modelli

I file dei modelli sono esclusi dal repository via `.gitignore`. La cartella `models/` contiene un `.gitkeep` come placeholder.

| File | Formato | Target | Descrizione |
|------|---------|--------|-------------|
| `yolov11s.hef` | HEF | Hailo-8 | Classificatore veicoli (6 classi) |
| `best_nano_v2_merged.rpk` | RPK | IMX500 | Detector veicoli on-chip |
| `tiny_yolov4_license_plates.hef` | HEF | Hailo-8 | Detector targhe |

### Classi veicoli (YOLOv11s + YOLOv11n)

`bus` · `car` · `pickup` · `truck` · `van` · `motorcycle`

### Dataset di training

Dataset merged: Kaggle (`hammadjavaid/vehicle-object-detection-dataset-5-classes`) + motorcycle da OpenImages V7.

| Split | Immagini |
|-------|----------|
| Train | 30.508 |
| Val | 8.331 |
| Test | 4.316 |

### Risultati

| Modello | mAP50 (test) | mAP50-95 (test) |
|---------|-------------|-----------------|
| YOLOv11s small (Hailo-8) | 0.900 | 0.700 |
| YOLOv11n nano (IMX500) | 0.852 | 0.652 |

## Struttura repository

```
ngs2026/
├── models/                          # Cartella modelli (file esclusi da .gitignore)
│   └── .gitkeep
├── test_images/                     # Immagini di test (veicoli vari)
├── test_nano_small/                 # Test YOLOv11s su Hailo-8 e YOLOv11n su IMX500
│   ├── test_small.py                # Inferenza small HEF su Hailo-8
│   ├── test_nano.py                 # Test nano RPK su IMX500
│   ├── debug.py                     # Debug formato output HEF
│   ├── imx500_object_detection_demo.py  # Demo ufficiale picamera2 IMX500
│   └── test_output/                 # Immagini con bbox veicoli
└── test_plate_detection_recognition/  # Test detection targhe + OCR
    ├── test_plate.py                # Plate detection con tiny_yolov4 su Hailo-8
    ├── test_lpr.py                  # Pipeline plate detection + LPRNet (solo numeri)
    ├── test_paddle.py               # Pipeline plate detection + EasyOCR
    ├── test_plate_output/           # Immagini con bbox targhe
    ├── test_lpr_output/             # Crop targhe con testo LPRNet
    └── test_paddle_output/          # Crop targhe con testo EasyOCR
```

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

### Installazione dipendenze OCR

```bash
pip install easyocr
```

## Esecuzione script

### Test classificazione veicoli (small su Hailo-8)

```bash
cd ~/hailo-apps && source setup_env.sh
python3 test_nano_small/test_small.py
```

### Test detection targhe (tiny_yolov4 su Hailo-8)

```bash
python3 test_plate_detection_recognition/test_plate.py
```

### Pipeline completa plate detection + OCR

```bash
PYTHONIOENCODING=utf-8 LANG=C.UTF-8 python3 test_plate_detection_recognition/test_paddle.py
```

> Nota: il prefisso `PYTHONIOENCODING=utf-8 LANG=C.UTF-8` è necessario per il locale italiano che non supporta caratteri unicode nella barra di progresso di EasyOCR.

## Note tecniche

- **Hailo-8** non supporta due network attivate contemporaneamente — nella pipeline completa le inferenze vanno schedulata in sequenza sullo stesso `VDevice`
- **LPRNet** (incluso nel Hailo Model Zoo) è addestrato su targhe israeliane con soli numeri — non adatto per targhe europee
- **EasyOCR** è la scelta attuale per l'OCR — gira su CPU RPi5, supporta lettere e numeri
- **PaddleOCR** non è compatibile con ARM64 (segmentation fault su RPi5)
- Il prefisso `PYTHONIOENCODING=utf-8 LANG=C.UTF-8` è necessario per alcuni script a causa del locale italiano `it_IT.ISO-8859-1`
- La nano su IMX500 funziona solo con flusso live della camera — non supporta inferenza su file video

## Prossimi passi

- Integrazione pipeline completa: IMX500 → Hailo-8 (small + tiny_yolov4) → EasyOCR
- Test in condizioni reali (parcheggio, veicoli veri)
- Valutazione fine-tuning tiny_yolov4 su dataset targhe europee
- Implementazione tracking con BotSORT per deduplicazione veicoli
