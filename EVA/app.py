from __future__ import annotations
import os, threading, queue, time
from typing import Tuple, List, Dict, Optional

import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from detector import YoloDetector, SourceConfig
from voice_manager import SpeechManager
from mp3_manager import Mp3Manager
from utils import now_ms, normalize_label, choose_priority_label
from config import (
    APP_NAME, APP_VERSION, DEFAULT_MODEL_PATH, DETECTION_MODEL_CANDIDATES,
    FRAME_WIDTH, FRAME_HEIGHT, CONF_THRESHOLD, CLASS_THRESHOLDS, BASE_CONF_FOR_MODEL,
    CLASS_COOLDOWNS_MS, CLASS_PRIORITY, LABELS_EN, LABELS_TL, VISUAL_BANNER,
    VOICE_MODE_DEFAULT, ESPEAKNG_RATE_WPM, ESPEAKNG_AMPLITUDE,
    MP3_PATHS, MP3_REPEAT_GAP_MS, RECENT_LIMIT, RECENT_LOG_THROTTLE_MS, RECENT_HEADER,
    CLASS_STABLE_FRAMES, STABLE_FRAMES, BG, CARD_BG, ACCENT,
    DRAW_BOXES, DEBUG_LOG_DETECTIONS, BOX_THICKNESS, BOX_FONT_SCALE, BOX_FONT_TH,
    LIVE_MAX_WIDTH, LIVE_MAX_HEIGHT, BANNER_TEXTS, MP3_REPEAT_COUNT,
    MAX_VOICE_EVENTS_PER_CLASS, ALWAYS_UPDATE_BANNER_ON_DETECTION, FEEDBACK_STRICT_STABILITY
)

TL_SET = {"red", "yellow", "green"}

def _iou(a, b):
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
    return inter / union if union > 0 else 0.0

def nms_same_class(dets, iou_thr=0.5):
    by_label = {}
    for d in dets:
        lab = d.get("label","")
        by_label.setdefault(lab, []).append(d)
    out = []
    for lab, arr in by_label.items():
        arr = sorted(arr, key=lambda x: float(x.get("conf", 0.0)), reverse=True)
        kept = []
        for d in arr:
            bb = d.get("bbox",[0,0,0,0])
            if not any(_iou(bb, k.get("bbox",[0,0,0,0])) > iou_thr for k in kept):
                kept.append(d)
        out.extend(kept)
    return out

def resolve_tl_conflicts(dets, iou_thr=0.5, conf_margin=0.08):
    tls = [d for d in dets if d.get("label","") in TL_SET]
    others = [d for d in dets if d.get("label","") not in TL_SET]
    used = set()
    clusters = []
    for i, d in enumerate(tls):
        if i in used: continue
        cluster = [i]; used.add(i)
        for j, e in enumerate(tls):
            if j in used: continue
            if _iou(d.get("bbox",[0,0,0,0]), e.get("bbox",[0,0,0,0])) > iou_thr:
                cluster.append(j); used.add(j)
        clusters.append(cluster)
    resolved = []
    for idxs in clusters:
        group = [tls[k] for k in idxs]
        group.sort(key=lambda x: float(x.get("conf",0.0)), reverse=True)
        top = group[0]
        resolved.append(top)
    return others + resolved

TEXT_OK = "#00ff9c"; TEXT_WARN = "#ffd166"; TEXT_STOP = "#ff4d4d"; TEXT_NORMAL = "#e6e6e6"

class VisionAssistantApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} {APP_VERSION}")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Backends & state
        self.detector = YoloDetector()
        self.video_thread = None  # type: Optional[threading.Thread]
        self.video_running = False
        self.info_q = queue.Queue()  # type: queue.Queue[Tuple[str, object]]

        # UI state
        self.banner_text = tk.StringVar(value="")
        self.recent_items: List[str] = []
        self.last_log_ms = 0

        # Pacing state
        self.per_class_last_ms: Dict[str, int] = {}
        self.per_class_stable: Dict[str, int] = {}
        self.voice_events_count: Dict[str, int] = {}

        # Settings variables (Frame 1)
        self.var_lang = tk.StringVar(value="tl")
        self.var_detect = tk.StringVar(value="both")
        self.var_voice_mode = tk.StringVar(value=VOICE_MODE_DEFAULT)
        self.var_source_mode = tk.StringVar(value="camera")
        self.var_cam_index = tk.IntVar(value=0)
        self.var_cam_width = tk.IntVar(value=FRAME_WIDTH)
        self.var_cam_height = tk.IntVar(value=FRAME_HEIGHT)
        self.var_video_path = tk.StringVar(value="")
        self.var_loop_video = tk.BooleanVar(value=False)
        self.var_muted = tk.BooleanVar(value=False)

        # Voice backends
        self.speech = SpeechManager(rate_wpm=ESPEAKNG_RATE_WPM, amplitude=ESPEAKNG_AMPLITUDE)
        self.mp3 = Mp3Manager(MP3_PATHS, repeat_gap_ms=MP3_REPEAT_GAP_MS)
        # Hysteresis for traffic-light color switching
        self._tl_last_label = None
        self._tl_last_change_ms = 0
        self._tl_hysteresis_ms = 1200

        # Build UI
        self._build_frame1()
        self._build_frame2()
        self._show(self.frame1)
        self._schedule_drain()
        self._sync_language_to_backends()

    
    def _noop(self, *args, **kwargs): 
        pass

    def _card(self, parent: tk.Misc, *, pad: int = 24) -> tk.Frame:
        f = tk.Frame(parent, bg=CARD_BG, bd=0, highlightthickness=1, highlightbackground="#1f2937")
        f.pack(fill="x", padx=pad, pady=(pad, 0))
        return f

    def _build_header(self, parent: tk.Misc, title: str) -> None:
        h = tk.Frame(parent, bg=BG)
        h.pack(fill="x", padx=24, pady=(18, 8))
        tk.Label(h, text=APP_NAME, bg=BG, fg="#e5e7eb", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(h, text=title, bg=BG, fg="#e5e7eb", font=("Segoe UI", 18, "bold")).pack(anchor="w", pady=(6, 0))
        tk.Label(h, text="Customize your experience to your needs.", bg=BG, fg="#6b7280", font=("Segoe UI", 10)).pack(anchor="w")

    
    def _segmented(self, parent: tk.Misc, pairs: List[Tuple[str, str]], var: tk.Variable) -> None:
        w = tk.Frame(parent, bg=CARD_BG, bd=0, highlightthickness=0)
        w.pack()
        btns: List[Tuple[str, ttk.Button]] = []
        def refresh():
            cur = var.get()
            for (val, btn) in btns:
                btn.configure(style=("Segment.Selected.TButton" if val == cur else "Segment.TButton"))
        for i, (text, val) in enumerate(pairs):
            b = ttk.Button(w, text=text, command=lambda v=val: (var.set(v), refresh()), style="Segment.TButton")
            b.grid(row=0, column=i, padx=(0 if i == 0 else 8, 0))
            btns.append((val, b))
        refresh()


    def _hline(self, parent: tk.Misc) -> None:
        tk.Frame(parent, bg="#1f2937", height=1).pack(fill="x", padx=18, pady=12)

    # frame 1: settings 
    def _build_frame1(self) -> None:
        self.frame1 = tk.Frame(self.root, bg=BG)
        self._build_header(self.frame1, "Language and Detection Settings")

        card = self._card(self.frame1)

        # Language
        sec1 = tk.Frame(card, bg=CARD_BG); sec1.pack(fill="x", padx=18, pady=(18, 12))
        tk.Label(sec1, text="Language", bg=CARD_BG, fg="#e5e7eb", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(sec1, text="Select the primary language.", bg=CARD_BG, fg="#6b7280").grid(row=1, column=0, sticky="w")
        opt1 = tk.Frame(sec1, bg=CARD_BG); opt1.grid(row=0, column=1, rowspan=2, sticky="e", padx=8)
        self._segmented(opt1, [("Tagalog", "tl"), ("English", "en")], self.var_lang)

        # Detection
        self._hline(card)
        sec2 = tk.Frame(card, bg=CARD_BG); sec2.pack(fill="x", padx=18, pady=12)
        tk.Label(sec2, text="Detection", bg=CARD_BG, fg="#e5e7eb", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(sec2, text="Choose what to detect.", bg=CARD_BG, fg="#6b7280").grid(row=1, column=0, sticky="w")
        opt2 = tk.Frame(sec2, bg=CARD_BG); opt2.grid(row=0, column=1, rowspan=2, sticky="e", padx=8)
        self._segmented(opt2, [("Traffic Lights", "traffic_lights"),
                               ("Road Signs", "road_signs"),
                               ("Both", "both")], self.var_detect)

        # Voice
        self._hline(card)
        sec3 = tk.Frame(card, bg=CARD_BG); sec3.pack(fill="x", padx=18, pady=12)
        tk.Label(sec3, text="Voice", bg=CARD_BG, fg="#e5e7eb", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(sec3, text="Select voice feedback type.", bg=CARD_BG, fg="#6b7280").grid(row=1, column=0, sticky="w")
        opt3 = tk.Frame(sec3, bg=CARD_BG); opt3.grid(row=0, column=1, rowspan=2, sticky="e", padx=8)
        self._segmented(opt3, [("Human", "mp3"), ("AI", "ai")], self.var_voice_mode)

        # Source
        self._hline(card)
        sec4 = tk.Frame(card, bg=CARD_BG); sec4.pack(fill="x", padx=18, pady=12)
        tk.Label(sec4, text="Source", bg=CARD_BG, fg="#e5e7eb", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(sec4, text="Select camera or video file.", bg=CARD_BG, fg="#6b7280").grid(row=1, column=0, sticky="w")

        left = tk.Frame(sec4, bg=CARD_BG); left.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=8)
        rb = tk.Frame(left, bg=CARD_BG); rb.pack(anchor="e")
        ttk.Radiobutton(rb, text="Camera", variable=self.var_source_mode, value="camera", command=self._update_source_ui).pack(side="left", padx=6)
        ttk.Radiobutton(rb, text="Video", variable=self.var_source_mode, value="video", command=self._update_source_ui).pack(side="left", padx=6)

        self.cam_panel = tk.Frame(left, bg=CARD_BG); self.vid_panel = tk.Frame(left, bg=CARD_BG)
        ttk.Label(self.cam_panel, text="Index").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        ttk.Spinbox(self.cam_panel, from_=0, to=9, width=5, textvariable=self.var_cam_index).grid(row=0, column=1, padx=4, pady=4, sticky="w")
        ttk.Label(self.cam_panel, text="Width").grid(row=0, column=2, padx=4, pady=4, sticky="e")
        ttk.Entry(self.cam_panel, width=6, textvariable=self.var_cam_width).grid(row=0, column=3, padx=4, pady=4, sticky="w")
        ttk.Label(self.cam_panel, text="Height").grid(row=0, column=4, padx=4, pady=4, sticky="e")
        ttk.Entry(self.cam_panel, width=6, textvariable=self.var_cam_height).grid(row=0, column=5, padx=4, pady=4, sticky="w")

        ttk.Entry(self.vid_panel, width=48, textvariable=self.var_video_path).grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(self.vid_panel, text="Browse...", command=self._browse_video).grid(row=0, column=1, padx=4, pady=4)
        ttk.Checkbutton(self.vid_panel, text="Loop video", variable=self.var_loop_video).grid(row=0, column=2, padx=8, pady=4)

        action = tk.Frame(card, bg=CARD_BG); action.pack(fill="x", padx=18, pady=(12, 18))
        ttk.Button(action, text="Start", command=self._on_start_clicked, style="Accent.TButton").pack(side="right")

        #Dark UI styling
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Accent.TButton", background=ACCENT, foreground="white")
        style.map("Accent.TButton", background=[("active", "#059669")])
        style.configure("TButton", padding=(16, 10), font=("Segoe UI", 11, "bold"), relief="flat", foreground="#1f2937", background=BG)
        style.map("TButton", background=[("active", "#0b7f67")], foreground=[("disabled", "#6b7280")])
        style.configure("TRadiobutton", font=("Segoe UI", 11), foreground="#1f2937", background=CARD_BG)
        style.configure("TCheckbutton", font=("Segoe UI", 11), foreground="#1f2937", background=CARD_BG)
        style.configure("TLabel", foreground="#1f2937", background=BG)
        style.configure("Card.TLabel", foreground="#1f2937", background=CARD_BG)
        style.configure("Segment.TButton", padding=(14,8), font=("Segoe UI", 10, "bold"), background="#0f172a", foreground="#1f2937")
        style.configure("Segment.Selected.TButton", padding=(14,8), font=("Segoe UI", 10, "bold"), background=ACCENT, foreground="#0b0f14")
        style.map("Segment.TButton", background=[("active", "#152238")])

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Accent.TButton", background=ACCENT, foreground="white")
        style.map("Accent.TButton", background=[("active", "#4338ca")])

        self._update_source_ui()

    #Frame 2
    def _build_frame2(self) -> None:
        self.frame2 = tk.Frame(self.root, bg=BG)

        # Header
        header = tk.Frame(self.frame2, bg=BG)
        header.pack(fill="x", padx=24, pady=(18, 10))
        tk.Label(header, text=APP_NAME, bg=BG, fg="#e5e7eb", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        # Card container
        main = self._card(self.frame2, pad=24)
        body = tk.Frame(main, bg=CARD_BG); body.pack(fill="both", expand=True, padx=18, pady=18)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # Left: Video + Banner + Buttons
        left = tk.Frame(body, bg=CARD_BG); left.grid(row=0, column=0, sticky="nsew")
        self.video_area = tk.Frame(left, bg="#0b0c10", height=LIVE_MAX_HEIGHT, width=LIVE_MAX_WIDTH)
        self.video_area.pack(fill="both", expand=True)
        self.video_area.update_idletasks()
        self.canvas = tk.Label(self.video_area, bg="#0b0c10")
        self.canvas.place(relx=0.5, rely=0.5, anchor="center")

        self.banner = tk.Label(left, textvariable=self.banner_text, bg="#0b1220", fg="white",
                               font=("Segoe UI", 12, "bold"), height=1, anchor="center")
        self.banner.pack(fill="x", pady=(8, 0))

        bottom = tk.Frame(left, bg=CARD_BG); bottom.pack(fill="x", pady=(12, 0))
        ttk.Button(bottom, text="Back", command=self._go_back).pack(side="left")
        self.btn_stop = ttk.Button(bottom, text="Stop", command=self.stop); self.btn_stop.pack(side="left", padx=12)
        self.btn_continue = ttk.Button(bottom, text="Continue", command=self._on_continue_clicked)
        self.btn_continue.pack(side="left")

        # Right: Recent panel
        right = tk.Frame(body, bg=CARD_BG); right.grid(row=0, column=1, sticky="nsew", padx=(18, 0))
        panel = tk.Frame(right, bg="#0f172a", highlightbackground="#1f2937", highlightthickness=1, bd=0)
        panel.pack(fill="both", expand=True)
        hdr = tk.Frame(panel, bg="#0f172a"); hdr.pack(fill="x", padx=10, pady=(10, 6))
        tk.Label(hdr, text=RECENT_HEADER, bg="#0f172a", fg="#e5e7eb", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        listwrap = tk.Frame(panel, bg="#0f172a"); listwrap.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.recent_scroll = tk.Scrollbar(listwrap, orient="vertical")
        self.recent_list = tk.Listbox(listwrap, yscrollcommand=self.recent_scroll.set, activestyle="none", highlightthickness=0, relief="flat", bg="#0b0f14", fg="#e5e7eb", selectbackground="#10b981", selectforeground="#0b0f14", font=("Segoe UI", 10))
        self.recent_scroll.config(command=self.recent_list.yview)
        self.recent_list.pack(side="left", fill="both", expand=True)
        self.recent_scroll.pack(side="right", fill="y")

    # Handlers
    def _on_start_clicked(self) -> None:
        self._sync_language_to_backends()
        self.start()

    def _on_continue_clicked(self) -> None:
        if not self.video_running:
            self._sync_language_to_backends()
            self.start()

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(title="Select video",
                                          filetypes=[("Videos","*.mp4 *.avi *.mov *.mkv"), ("All files","*.*")])
        if path:
            self.var_video_path.set(path)

    def _update_source_ui(self) -> None:
        if self.var_source_mode.get() == "camera":
            self.vid_panel.forget()
            self.cam_panel.pack(anchor="e", pady=(6, 0))
        else:
            self.cam_panel.forget()
            self.vid_panel.pack(anchor="e", pady=(6, 0))

    def _go_back(self) -> None:
        if self.video_running:
            self.stop()
        self._show(self.frame1)

    def _on_mute_changed(self) -> None:
        flag = bool(self.var_muted.get())
        self.speech.mute(flag); self.mp3.mute(flag)

    def _show(self, frame: tk.Misc) -> None:
        for f in (getattr(self, "frame1", None), getattr(self, "frame2", None)):
            if f is not None:
                f.pack_forget()
        frame.pack(fill="both", expand=True)

    def _pick_model_for_scope(self) -> str:
        cand = DETECTION_MODEL_CANDIDATES.get(self.var_detect.get(), []) + [DEFAULT_MODEL_PATH]
        for p in cand:
            if os.path.exists(p):
                return p
        return DEFAULT_MODEL_PATH

    def _build_source_config(self) -> SourceConfig:
        if self.var_source_mode.get() == "camera":
            return SourceConfig(mode="camera", cam_index=int(self.var_cam_index.get()),
                                width=int(self.var_cam_width.get()), height=int(self.var_cam_height.get()))
        return SourceConfig(mode="video", video_path=self.var_video_path.get(),
                            loop_video=bool(self.var_loop_video.get()))

    def _sync_language_to_backends(self) -> None:
        lang = self.var_lang.get()
        self.speech.set_language(lang, voice_override=None)
        self.mp3.set_language(lang)

    # Start/Stop nang video
    def start(self) -> None:
        if self.video_running:
            return
        try:
            self.detector.load(self._pick_model_for_scope())
        except Exception as e:
            messagebox.showerror(APP_NAME, "Failed to load model:\n{}".format(e))
            return
        if not self.detector.open_source(self._build_source_config()):
            messagebox.showerror(APP_NAME, "Failed to open source (camera/video). Check your settings.")
            return

        self.video_running = True
        self.banner_text.set("")
        self.recent_items.clear()
        try:
            self.recent_list.delete(0, tk.END)
        except Exception:
            pass
        self.per_class_last_ms.clear(); self.per_class_stable.clear(); self.voice_events_count.clear()
        self._show(self.frame2); self.btn_stop.configure(state="normal")
        try:
            self.btn_continue.configure(state="disabled")
        except Exception:
            pass

        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def stop(self) -> None:
        if not self.video_running:
            return
        self.video_running = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.5)
        try:
            self.detector.close()
        except Exception:
            pass
        self.video_thread = None
        self.banner_text.set("Stopped.")
        try:
            self.btn_stop.configure(state="disabled")
            self.btn_continue.configure(state="normal")
        except Exception:
            pass

    def on_close(self) -> None:
        try: self.stop()
        except Exception: pass
        try: self.speech.stop(); self.mp3.stop()
        except Exception: pass
        self.root.destroy()

    # UI queue
    def _schedule_drain(self) -> None:
        self.root.after(60, self._drain_info_queue)

    def _drain_info_queue(self) -> None:
        try:
            while True:
                typ, payload = self.info_q.get_nowait()
                if typ == "image":
                    self._set_canvas_image(payload)
                elif typ == "banner":
                    if isinstance(payload, tuple):
                        txt, fg = payload
                        self.banner_text.set(txt)
                        try: self.banner.configure(fg=fg)
                        except Exception: pass
                    else:
                        self.banner_text.set(payload)
                elif typ == "recent":
                    self._add_recent(payload)
        except queue.Empty:
            pass
        self._schedule_drain()

    def _set_canvas_image(self, frame_bgr) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ih, iw = rgb.shape[:2]
        img_ratio = iw / ih if ih else 1.0
        try:
            cw = max(1, int(self.video_area.winfo_width()))
            ch = max(1, int(self.video_area.winfo_height()))
        except Exception:
            cw, ch = 960, 540
        cont_ratio = cw / ch if ch else img_ratio
        if img_ratio > cont_ratio:
            tw = cw; th = int(cw / img_ratio)
        else:
            th = ch; tw = int(ch * img_ratio)
        if tw < 1: tw = 1
        if th < 1: th = 1
        pil = Image.fromarray(rgb).resize((tw, th), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(image=pil)
        self.canvas.configure(image=self._photo)
        self.canvas.place(relx=0.5, rely=0.5, anchor='center')

    # Feedback sa voice & detections 
    def _add_recent(self, text: str) -> None:
        now = now_ms()
        if now - self.last_log_ms < RECENT_LOG_THROTTLE_MS:
            return
        self.last_log_ms = now
        stamp = time.strftime("%H:%M:%S")
        entry = f"[{stamp}] {text}"
        self.recent_items.append(entry)
        if len(self.recent_items) > RECENT_LIMIT:
            self.recent_items = self.recent_items[-RECENT_LIMIT:]
        try:
            self.recent_list.delete(0, tk.END)
            for it in self.recent_items:
                self.recent_list.insert(tk.END, it)
            self.recent_list.see(tk.END)
        except Exception:
            pass

    def _update_banner(self, label: str) -> None:
        txt = BANNER_TEXTS.get(label, label.upper())
        if label == 'red': fg = TEXT_STOP
        elif label == 'yellow': fg = TEXT_WARN
        elif label == 'green': fg = TEXT_OK
        else: fg = TEXT_NORMAL
        self.info_q.put(('banner', (txt, fg)))

    def _draw_boxes(self, frame, dets: List[Dict]) -> None:
        try:
            import cv2
        except Exception:
            return
        thickness = int(BOX_THICKNESS); fscale = float(BOX_FONT_SCALE); fth = int(BOX_FONT_TH)
        h, w = frame.shape[:2]
        for d in dets:
            try:
                x1, y1, x2, y2 = [int(v) for v in d.get('bbox', [0,0,0,0])]
                x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
                y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
                if x2 <= x1 or y2 <= y1: continue
                label = str(d.get('label', '')); conf = float(d.get('conf', 0.0))
                txt = f'{label} {conf:.2f}'
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), thickness)
                cv2.putText(frame, txt, (x1, max(y1-6, 10)), cv2.FONT_HERSHEY_SIMPLEX, fscale, (0,255,0), fth, cv2.LINE_AA)
            except Exception:
                continue

    def _video_loop(self) -> None:
        labels_map = LABELS_TL if self.var_lang.get() == "tl" else LABELS_EN
        is_mp3 = (self.var_voice_mode.get() == "mp3")
        while self.video_running:
            ok, frame = self.detector.read_frame()
            if not ok:
                time.sleep(0.01)
                continue

            try:
                dets = self.detector.predict(frame)
            except Exception:
                dets = []

            if DRAW_BOXES:
                self._draw_boxes(frame, dets)
            if DEBUG_LOG_DETECTIONS and dets:
                self.info_q.put(("recent", f"raw: {len(dets)} detections"))

            present: List[str] = []
            for d in dets:
                label = normalize_label(d.get("label", ""))
                conf = float(d.get("conf", 0.0))
                thr = CLASS_THRESHOLDS.get(label, CONF_THRESHOLD)
                if conf >= max(BASE_CONF_FOR_MODEL, thr):
                    present.append(label)

            winner = choose_priority_label(present)
            if not winner:
                s = set(present)
                for p in CLASS_PRIORITY:
                    if p in s:
                        winner = p
                        break
          
            if winner in TL_SET:
                nowt_h = now_ms()
                if self._tl_last_label and winner != self._tl_last_label:
                    if (nowt_h - self._tl_last_change_ms) < self._tl_hysteresis_ms:
                      
                        winner = self._tl_last_label
                    else:
                        self._tl_last_label = winner
                        self._tl_last_change_ms = nowt_h
                elif self._tl_last_label is None:
                    self._tl_last_label = winner
                    self._tl_last_change_ms = nowt_h


            if ALWAYS_UPDATE_BANNER_ON_DETECTION and winner:
                self._update_banner(winner)

            to_speak = None
            if winner:
                req = CLASS_STABLE_FRAMES.get(winner, STABLE_FRAMES)
                if not FEEDBACK_STRICT_STABILITY:
                    req = 1
                self.per_class_stable[winner] = self.per_class_stable.get(winner, 0) + 1
                for k in list(self.per_class_stable.keys()):
                    if k != winner:
                        self.per_class_stable[k] = 0
                if self.per_class_stable[winner] >= req:
                    nowt = now_ms(); last = self.per_class_last_ms.get(winner, 0); cd = CLASS_COOLDOWNS_MS.get(winner, 6000)
                    if (nowt - last) >= cd:
                        cnt = self.voice_events_count.get(winner, 0)
                        if MAX_VOICE_EVENTS_PER_CLASS < 0 or cnt < MAX_VOICE_EVENTS_PER_CLASS:
                            to_speak = winner
                            self.per_class_last_ms[winner] = nowt
                            self.voice_events_count[winner] = cnt + 1

            if to_speak:
                phrase = (LABELS_TL.get(to_speak, to_speak) if self.var_lang.get()=='tl' else LABELS_EN.get(to_speak, to_speak))
                self._update_banner(to_speak)
                if is_mp3 and self.mp3.available:
                    ok = self.mp3.play_label(to_speak, repeat=MP3_REPEAT_COUNT)
                    if not ok:
                        self.speech.speak(phrase, times=2)
                else:
                    try:
                        self.speech.speak(phrase, times=2)
                    except Exception:
                        self.speech.say(phrase)
                self.info_q.put(("recent", f"Detected: {to_speak.upper()}"))

            self.info_q.put(("image", frame))

        try:
            self.detector.close()
        except Exception:
            pass

def main() -> None:
    app = VisionAssistantApp()
    app.root.geometry("1200x720+120+60")
    app.root.mainloop()

if __name__ == "__main__":
    main()
