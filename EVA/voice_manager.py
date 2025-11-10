from __future__ import annotations
import threading, queue, subprocess
from typing import Optional
from config import ESPEAKNG_BIN, ESPEAKNG_VOICE_EN, ESPEAKNG_VOICE_TL, ESPEAKNG_RATE_WPM, ESPEAKNG_AMPLITUDE

class SpeechManager:
    def __init__(self, rate_wpm: int = 165, amplitude: int = 175) -> None:
        self._q = queue.Queue()  # type: queue.Queue[tuple[str, str]]
        self._stop = threading.Event()
        self._muted = False
        self.rate_wpm = int(rate_wpm or ESPEAKNG_RATE_WPM)
        self.amplitude = int(amplitude or ESPEAKNG_AMPLITUDE)
        self.lang = "en"
        self.voice_override = {"en": ESPEAKNG_VOICE_EN, "tl": ESPEAKNG_VOICE_TL}
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def set_language(self, lang: str, voice_override: Optional[str] = None) -> None:
        self.lang = lang or "en"
        if voice_override:
            self.voice_override[self.lang] = voice_override

    def mute(self, flag: bool) -> None:
        self._muted = bool(flag)

    def say(self, text: str) -> None:
        self.speak(text, times=1)

    def speak(self, text: str, times: int = 1) -> None:
        if not text or self._muted:
            return
        for _ in range(max(1, int(times))):
            self._q.put((self.lang, text))

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                lang, text = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            self._synth(lang, text)

    def _synth(self, lang: str, text: str) -> None:
        voice = self.voice_override.get(lang) or ("tl" if lang == "tl" else "en")
        cmd = [ESPEAKNG_BIN]
        if voice: cmd += ["-v", str(voice)]
        if self.rate_wpm: cmd += ["-s", str(int(self.rate_wpm))]
        if self.amplitude: cmd += ["-a", str(int(self.amplitude))]
        cmd += [text]
        try:
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
