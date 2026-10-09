from __future__ import annotations


def ensure_odd(value: int, minimum: int = 1) -> int:
    value = max(int(value), minimum)
    if value % 2 == 0:
        value += 1
    return value


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def valid_canny_thresholds(low: int, high: int) -> tuple[int, int]:
    low = int(clamp(low, 0, 255))
    high = int(clamp(high, 0, 255))
    if low > high:
        low = high
    return low, high
