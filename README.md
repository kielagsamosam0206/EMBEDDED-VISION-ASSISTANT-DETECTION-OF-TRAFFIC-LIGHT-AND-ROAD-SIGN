# Embedded Vision Assistant for Color-Blind Drivers
An embedded real-time traffic light and road sign recognition system designed to assist Color Vision Deficient (CVD) drivers. The system provides **visual** and **auditory feedback** to help recognize traffic light states and selected road signs while driving.

This project uses **YOLOv8 object detection**, **Raspberry Pi 5**, and a **Tkinter-based GUI** to display live camera feed overlays and announce detected traffic signals in either **English or Tagalog**.

---

## 🔍 Features
- Real-time detection of:
  - Traffic Light States (Red, Yellow, Green)
  - Road Signs such as:
    - No Parking
    - No Entry
    - Stop
    - Yield
    - Pedestrian Crossing
- On-screen banner and bounding box visual feedback
- Voice feedback (English / Tagalog)
- Adjustable detection thresholds and stability frames
- Supports both camera feed and video file sources
- Prioritized detection (traffic lights > road signs)
- Cooldown and announcement limit logic to prevent repeated prompts

---

## 🛠 Hardware Used
| Component | Description |
|---------|-------------|
| Raspberry Pi 5 | Main processing unit |
| ELP USB Camera | Real-time video capture |
| 7-inch HDMI LCD Display | Visual feedback output |
| Mini USB Speaker (Adafruit) | Audio feedback output |
| UPS HAT E with 21700 Batteries | Stable and portable power supply |

---
