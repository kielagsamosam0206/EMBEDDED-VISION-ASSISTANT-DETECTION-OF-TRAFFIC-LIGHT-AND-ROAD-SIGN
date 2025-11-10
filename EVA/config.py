from __future__ import annotations
from pathlib import Path
from typing import Dict, List

APP_NAME: str = "EMBEDDED VISION ASSISTANT"
APP_VERSION: str = "by alitaptap"

# Paths
BASE_DIR: Path = Path(__file__).resolve().parent
MODELS_DIR: Path = BASE_DIR / "models"

DEFAULT_MODEL_FILENAME: str = "best.pt"
DEFAULT_MODEL_PATH: str = str(MODELS_DIR / DEFAULT_MODEL_FILENAME)


DETECTION_MODEL_CANDIDATES: Dict[str, List[str]] = {
    "traffic_lights": [
        #str(MODELS_DIR / "traffic_light.pt"),
        str(MODELS_DIR / "tf4.onnx"),
        #str(MODELS_DIR / "TL.pt"),
        #str(MODELS_DIR / "TRAFFIC_WITHNULL2.pt"),
        DEFAULT_MODEL_PATH,            # fallback
    ],
    "road_signs": [
        #str(MODELS_DIR / "road_signs.pt"),
        str(MODELS_DIR / "ROADSIGN_FINAL.pt"),
        #str(MODELS_DIR / "signs_best.pt"),
        DEFAULT_MODEL_PATH,            # fallback
    ],
    "both": [
        str(MODELS_DIR / "both.pt"),
        str(MODELS_DIR / "best.pt"),
        DEFAULT_MODEL_PATH,            # fallback
    ],
}


FRAME_WIDTH: int  = 1280
FRAME_HEIGHT: int = 720

CONF_THRESHOLD: float      = 0.50
BASE_CONF_FOR_MODEL: float = 0.50

CLASS_THRESHOLDS: Dict[str, float] = {
    # Traffic lights
    "red": 0.65,
    "yellow": 0.60,
    "green": 0.50,
    
    # Signs
    "no parking": 0.55,
    "no u turn": 0.60,
    "pedestrian crossing": 0.50,
    "stop": 0.60,
    "yield": 0.60,
}

STABLE_FRAMES: int = 3
CLASS_STABLE_FRAMES: Dict[str, int] = {
    "red": 5,
    "yellow": 4,
    "green": 4,
    
    "no parking": 4,
    "no u turn": 2,
    "pedestrian crossing": 2,
    "stop": 3,
    "yield": 2,
}

# cooldowns
CLASS_COOLDOWNS_MS: Dict[str, int] = {
    "red": 30000,
    "yellow": 23000,
    "green": 23000,
    
    "no parking": 23000,
    "no u turn": 23000,
    "pedestrian crossing": 23000,
    "stop": 23000,
    "yield": 23000,
}
CLASS_PRIORITY: List[str] = [
    "red",
    "green",
    "yellow",
    "no u turn",
    "no parking",
    "yield",
    "stop",
    "pedestrian crossing",
]

# Spoken phrases per label
LABELS_EN: Dict[str, str] = {
    "red": "red light ahead, Stop",
    "yellow": "yellow light ahead, Ready to Stop",
    "green": "green light, Go",
    
    "no parking": "No parking zone. Do not park here.",
    "no u turn": "No U-turn zone. Do not make a U-turn.",
    "pedestrian crossing": "Pedestrian crossing ahead. Slow down.",
    "stop": "Stop sign ahead. Prepare to stop.",
    "yield": "Yield ahead. Give way.",
}
LABELS_TL: Dict[str, str] = {
    "red": "Hintu, may red layt",
    "yellow": "Humanda sa Pagpreno, may yelow layt",
    "green": "Puwede na tumakbo, may grin layt",
    
    "no parking": "Bawal pumarada. Huwag pumarada dito.",
    "no u turn": "Bawal lumiko pabalik. Huwag liliko pabalik.",
    "pedestrian crossing": "May malapit na tawiran. Bagalan ang pag takbo",
    "stop": "May istap sayn sa unahan. Humanda sa paghinto.",
    "yield": "Yild sayn sa unahan. Magbigay-daan.",
}

# Banner
VISUAL_BANNER = {
    "red": "RED LIGHTS : STOP",
    "yellow": "YELLOW LIGHTS : READY TO STOP",
    "green": "GREEN LIGHTS : GO",
    "no u turn": "NO U TURN",
    "no entry": "NO ENTRY",
    "pedestrian crossing": "PEDESTRIAN CROSSING",
    "stop": "STOP",
    "yield": "YIELD",
}
BANNER_TEXTS = VISUAL_BANNER

# Recent list
RECENT_LIMIT: int           = 30
RECENT_LOG_THROTTLE_MS: int = 600
RECENT_HEADER: str          = "Recent Detections"


VOICE_MODE_DEFAULT: str = "ai"  # "ai" (espeak-ng) or "mp3"


MAX_VOICE_EVENTS_PER_CLASS: int   = -1   # -1 = unlimited
ALWAYS_UPDATE_BANNER_ON_DETECTION: bool = True  
FEEDBACK_STRICT_STABILITY: bool   = True 

# eSpeak-NG
ESPEAKNG_BIN: str       = "espeak-ng"
ESPEAKNG_VOICE_EN: str  = "en"
ESPEAKNG_VOICE_TL: str  = "id"
ESPEAKNG_RATE_WPM: int  = 185
ESPEAKNG_AMPLITUDE: int = 140

# MP3
MP3_PATHS = {
    "en": {
        "red": "audio/en/red.mp3",
        "yellow": "audio/en/yellow.mp3",
        "green": "audio/en/green.mp3",
        "no u turn": "audio/en/no_u_turn.mp3",
        "no parking": "audio/en/no_parking.mp3",
        "pedestrian crossing": "audio/en/pedestrian_crossing.mp3",
        "stop": "audio/en/stop.mp3",
        "yield": "audio/en/yield.mp3",
    },
    "tl": {
        "red": "audio/tl/red.mp3",
        "yellow": "audio/tl/yellow.mp3",
        "green": "audio/tl/green.mp3",
        "no u turn": "audio/tl/no_u_turn.mp3",
        "no parking": "audio/tl/no_parking.mp3",
        "pedestrian crossing": "audio/tl/pedestrian_crossing.mp3",
        "stop": "audio/tl/stop.mp3",
        "yield": "audio/tl/yield.mp3",
    },
}
MP3_VOLUME: float         = 1.0
MP3_REPEAT_COUNT: int     = 2
MP3_REPEAT_GAP_MS: int    = 800 


BG: str       = "#0b0f14"
CARD_BG: str  = "#111827"
ACCENT: str   = "#10b981"

DRAW_BOXES: bool     = True
DEBUG_LOG_DETECTIONS = False
BOX_THICKNESS: int   = 3
BOX_FONT_SCALE: float= 0.6
BOX_FONT_TH: int     = 1

LIVE_MAX_WIDTH: int  = 920
LIVE_MAX_HEIGHT: int = 650


# SAHI tiling
TILING_ENABLED: bool   = False
TILE_SIZE: int         = 640
TILE_OVERLAP: float    = 0.30
TILING_MIN_WIDTH: int  = 960
TILING_NMS_IOU: float  = 0.50
