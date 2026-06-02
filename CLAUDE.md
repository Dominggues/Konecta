# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**KONECTA** is a TCC (Trabalho de Conclusão de Curso) at Faculdade Unimax that builds an AI-powered Brazilian Sign Language (Libras) recognition system. Core scope (Guilherme's responsibility): webcam → hand landmarks → text on terminal. Web interface, voice synthesis, and other expansions are out of scope here.

Contributors: Guilherme Domingues, Vinicius Rosa, Matheus Isídio. Supervisor: Luyz Chiavini.

## Environment Setup

O projeto usa Python 3.12 do runtime interno. O venv da raiz (Scripts/, Lib/) está obsoleto — use `.venv/`:

```powershell
# Ativar venv (já criado)
.venv\Scripts\Activate.ps1

# Ou invocar diretamente sem ativar:
.venv\Scripts\python.exe OCR/coletar_dados.py
```

Dependências já instaladas em `.venv/`. Para reinstalar:
```powershell
.venv\Scripts\python.exe -m pip install -r OCR/requirements.txt
```

## Three-Step Workflow

```powershell
# 1. Collect training samples (press a key per sign, 100 samples each)
.venv\Scripts\python.exe OCR/coletar_dados.py

# 2. Train the classifier
.venv\Scripts\python.exe OCR/treinar_modelo.py

# 3. Real-time recognition (prints detected sign to terminal)
.venv\Scripts\python.exe OCR/reconhecer_libras.py          # default webcam
.venv\Scripts\python.exe OCR/reconhecer_libras.py 1        # secondary camera
.venv\Scripts\python.exe OCR/reconhecer_libras.py video.mp4
```

On first run, `coletar_dados.py` and `reconhecer_libras.py` auto-download `hand_landmarker.task` (~12 MB) into `OCR/dados_libras/`.

## Architecture

```
OCR/
├── utils.py              # shared: normalizar_landmarks() — scale+translation invariant
├── coletar_dados.py      # step 1: webcam → MediaPipe → saves labeled CSV
├── treinar_modelo.py     # step 2: CSV → RandomForestClassifier → modelo.pkl
├── reconhecer_libras.py  # step 3: webcam/file → MediaPipe → model → terminal
├── requirements.txt
└── dados_libras/         # gitignored: dados.csv + modelo.pkl (generated locally)
```

**MediaPipe version:** uses Tasks API (`mediapipe.tasks.python.vision.HandLandmarker`, `RunningMode.VIDEO`). The legacy `mp.solutions.hands` was removed in mediapipe 0.10.35+.

**Feature vector:** 21 MediaPipe hand landmarks × 3 (x, y, z) = **63 features**, normalized so the wrist is at the origin and scale is fixed by the wrist→middle-MCP distance. The same `normalizar_landmarks()` from `utils.py` must be used in both collection and recognition — changing it invalidates existing models.

**Stabilization:** `reconhecer_libras.py` buffers the last 15 frame predictions and only emits a terminal print when 70% of frames agree on the same sign, preventing flicker.

## Gitignore Notes

`OCR/dados_libras/` (CSV + pkl) and `Datasets/` are never committed — large local files generated at runtime.
