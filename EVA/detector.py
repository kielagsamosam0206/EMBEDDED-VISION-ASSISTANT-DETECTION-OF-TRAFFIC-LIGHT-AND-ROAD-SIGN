from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from config import TILING_ENABLED, TILE_SIZE, TILE_OVERLAP, TILING_MIN_WIDTH, TILING_NMS_IOU
import os
import cv2


#SAHI tiling helpers
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0, (ax2-ax1)) * max(0, (ay2-ay1))
    area_b = max(0, (bx2-bx1)) * max(0, (by2-by1))
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0

def _nms_by_label(dets, iou_thr=0.5):
    by_label = {}
    for d in dets:
        by_label.setdefault(d.get('label',''), []).append(d)
    merged = []
    for lab, arr in by_label.items():
        arr = sorted(arr, key=lambda x: float(x.get('conf',0.0)), reverse=True)
        keep = []
        while arr:
            top = arr.pop(0)
            keep.append(top)
            arr = [d for d in arr if _compute_iou(top['bbox'], d['bbox']) < iou_thr]
        merged.extend(keep)
    return merged
@dataclass
class SourceConfig:
    mode: str
    cam_index: int = 0
    width: int = 1280
    height: int = 720
    video_path: Optional[str] = None
    loop_video: bool = False

class YoloDetector:
    def __init__(self) -> None:
        self.model = None
        self.cap = None  # type: Optional[cv2.VideoCapture]
        self.source = None  # type: Optional[SourceConfig]

    def load(self, model_path: str) -> None:
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def open_source(self, source: SourceConfig) -> bool:
        self.source = source
        if source.mode == "camera":
            cap = cv2.VideoCapture(source.cam_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source.cam_index)
            if not cap.isOpened():
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, source.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, source.height)
            self.cap = cap
            return True
        if not source.video_path or not os.path.exists(source.video_path):
            return False
        cap = cv2.VideoCapture(source.video_path)
        if not cap.isOpened():
            return False
        self.cap = cap
        return True

    def read_frame(self):
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if not ret and self.source and self.source.mode == "video" and self.source.loop_video:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    def predict(self, frame) -> List[Dict]:
        if self.model is None:
            return []
        h, w = frame.shape[:2]
        use_tiling = bool(TILING_ENABLED and w >= TILING_MIN_WIDTH and TILE_SIZE > 0 and 0.0 <= TILE_OVERLAP < 0.5)
        if not use_tiling:
            results = self.model.predict(frame, verbose=False)
            r0 = results[0]
            names = r0.names if hasattr(r0, "names") else {}
            out: List[Dict] = []
            if getattr(r0, "boxes", None) is not None:
                for b in r0.boxes:
                    try:
                        cls_id = int(b.cls[0].item()); conf = float(b.conf[0].item()); xyxy = b.xyxy[0].tolist()
                    except Exception:
                        cls_id = int(b.cls.item()); conf = float(b.conf.item()); xyxy = list(b.xyxy.view(-1).tolist())
                    label = names.get(cls_id, str(cls_id))
                    out.append({"label": label, "conf": conf, "bbox": xyxy})
            return out

        step = int(TILE_SIZE * (1.0 - float(TILE_OVERLAP)))
        if step <= 0: step = TILE_SIZE
        dets_all: List[Dict] = []
        results0 = self.model.predict(frame, imgsz=TILE_SIZE, verbose=False)
        names = results0[0].names if hasattr(results0[0], "names") else {}
        y = 0
        while y < h:
            x = 0
            y2 = min(h, y + TILE_SIZE)
            while x < w:
                x2 = min(w, x + TILE_SIZE)
                tile = frame[y:y2, x:x2]
                results = self.model.predict(tile, imgsz=TILE_SIZE, verbose=False)
                r = results[0]
                if getattr(r, "boxes", None) is not None:
                    for b in r.boxes:
                        try:
                            cls_id = int(b.cls[0].item()); conf = float(b.conf[0].item()); xyxy = b.xyxy[0].tolist()
                        except Exception:
                            cls_id = int(b.cls.item()); conf = float(b.conf.item()); xyxy = list(b.xyxy.view(-1).tolist())
                        x1, y1, x2b, y2b = xyxy
                        ox1 = float(x1 + x); oy1 = float(y1 + y); ox2 = float(x2b + x); oy2 = float(y2b + y)
                        label = names.get(cls_id, str(cls_id))
                        dets_all.append({"label": label, "conf": conf, "bbox": [ox1, oy1, ox2, oy2]})
                x += step
            y += step
        merged = _nms_by_label(dets_all, iou_thr=float(TILING_NMS_IOU))
        return merged

        return out

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
