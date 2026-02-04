from __future__ import annotations
import math

EPS = 1e-12

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def sign_nz(x: float) -> int:
    if x > 0: return 1
    if x < 0: return -1
    return 0

def safe_std(n: int, m2: float) -> float:
    # Welford std (sample)
    if n > 1 and m2 >= 0:
        return math.sqrt(m2 / (n - 1))
    return 0.0

def minutes_to_sec(m: float) -> float:
    return m * 60.0

def sec_to_min(s: float) -> float:
    return s / 60.0
