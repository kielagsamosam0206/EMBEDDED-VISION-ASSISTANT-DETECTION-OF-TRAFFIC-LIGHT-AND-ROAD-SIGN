from __future__ import annotations
import os, threading, queue, time
from typing import Dict, Optional

try:
    import pygame
    _HAVE_PYGAME = True
except Exception:
    _HAVE_PYGAME = False

from config import MP3_VOLUME, MP3_REPEAT_GAP_MS

class Mp3Manager:
    available_reason: str = ''
    def __init__(self, lang_to_label_to_path: Dict[str, Dict[str, str]], repeat_gap_ms: int = 800) -> None:
        self.available: bool = False
        self._muted: bool = False
        self.lang: str = "en"
        self.paths = lang_to_label_to_path
        self.repeat_gap_ms = repeat_gap_ms
        self._q = queue.Queue()  # type: queue.Queue[str]
        self._stop = threading.Event()

        if _HAVE_PYGAME:
            try:
                pygame.mixer.init()
                pygame.mixer.music.set_volume(float(MP3_VOLUME))
                self.available = True
                self.available_reason = 'pygame OK'
            except Exception as e:
                self.available = False
                self.available_reason = f'pygame init failed: {e}'

        if self.available:
            threading.Thread(target=self._run, daemon=True).start()

    def set_language(self, lang: str) -> None:
        self.lang = lang

    def play_label(self, label: str, repeat: int = 1) -> bool:
        if not self.available or self._muted:
            return False
        p = self._resolve_path(label)
        if p and os.path.exists(p):
            for _ in range(max(1, int(repeat))):
                self._q.put(p)
            return True
        return False

    def mute(self, flag: bool) -> None:
        self._muted = flag

    def stop(self) -> None:
        self._stop.set()
        if _HAVE_PYGAME and self.available:
            try:
                pygame.mixer.stop()
                pygame.mixer.quit()
            except Exception:
                pass

    def _resolve_path(self, label: str) -> Optional[str]:
        d = self.paths.get(self.lang) or {}
        return os.path.abspath(d.get(label, "")) if d.get(label) else None

    def _run(self) -> None:
        import pygame
        while not self._stop.is_set():
            try:
                path = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            # Load and start playback
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
            except Exception:
                continue
            # Wait until current finishes before playing next
            while not self._stop.is_set() and pygame.mixer.music.get_busy():
                time.sleep(0.02)
            time.sleep(0.02)
