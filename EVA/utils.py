from __future__ import annotations
import time
from collections import Counter
from typing import List, Optional

def now_ms() -> int:
    return int(time.time() * 1000)

def normalize_label(label: str) -> str:
    if not label:
        return label
    L = label.strip().lower().replace('-', ' ').replace('_', ' ')
    aliases = {
        "no-u-turn": "no u turn", "no_parking": "no parking", "no-parking": "no parking",
        "pedestrian_crossing": "pedestrian crossing", "t-intersection": "t intersection",
        "slippery_when_wet": "slippery when wet",
    }
    return aliases.get(L, L)

def choose_priority_label(labels: List[str]) -> Optional[str]:
    if not labels:
        return None
    tl = [l for l in labels if l in {"red", "yellow", "green"}]
    if not tl:
        return None
    c = Counter(tl)
    order = {"red": 0, "yellow": 1, "green": 2}
    return sorted(c.items(), key=lambda kv: (-kv[1], order[kv[0]]))[0][0]
