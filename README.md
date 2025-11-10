# Embedded Vision Assistant for Color-Blind Drivers (Raspberry Pi 5 + Windows)

Real-time **traffic-light** (Red/Yellow/Green) and **road-sign** detection with clear **on-screen banners** and **voice prompts** (English/Tagalog) to assist drivers with Color Vision Deficiency (CVD).

Built with **YOLOv8**, **OpenCV**, and a **Tkinter** GUI. Runs on **Raspberry Pi 5** and Windows laptops/desktops.

---

## Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Hardware](#hardware)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
  - [Windows](#windows)
  - [Raspberry-Pi-5](#raspberry-pi-5)
- [Model Weights](#model-weights)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Autostart on Raspberry Pi (systemd)](#autostart-on-raspberry-pi-systemd)
- [Troubleshooting](#troubleshooting)
- [Dataset & Training (YOLOv8)](#dataset--training-yolov8)
- [Evaluation & User Studies](#evaluation--user-studies)
- [Roadmap](#roadmap)
- [Ethics, Safety, and Privacy](#ethics-safety-and-privacy)
- [Cite this Project](#cite-this-project)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features
- **Detections**
  - Traffic lights: **red**, **yellow**, **green**
  - Road signs: **no parking**, **no entry**, **no U-turn**, **stop**, **yield**, **pedestrian crossing**
- **Smart feedback logic**
  - Per-class **confidence thresholds**
  - Per-class **stability frames** (consecutive-frame requirement)
  - Per-class **cooldowns** and **max announcements**
  - **Priority rules** (traffic lights before road signs)
- **UI**
  - Live video preview, bounding boxes, banner text
  - Recent Detections panel
  - Language toggle (EN/TL)
- **Voice**
  - **eSpeak-NG** (AI TTS) or **MP3 prompts** (human voice)
- **Sources**
  - USB camera / built-in webcam
  - Video files for demos and testing
- **Cross-platform**
  - Windows (Python 3.10–3.12)
  - Raspberry Pi 5 (64-bit OS)

---

## System Architecture
High-level flow:


