"""环境噪声底估计。

门限如果写死，换个房间就失效：安静会议室里 0.008 是合理的，同时开着空调
的教室里 0.008 会让 VAD 一直盯着底噪当语音。所以门限要跟着环境走。

做法是滑动窗口取低分位。低分位而不是均值，是因为底噪的定义就是「持续存在
的那个最低电平」——一次关门声不该把门限抬高五秒钟。
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

__all__ = ["NoiseFloor", "percentile", "rms"]


def rms(audio):
    """均方根。用 float64 累加，和 C++ 版同一个公式。"""
    a = np.asarray(audio, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a * a)))


def percentile(values, p):
    """线性插值分位数，公式写死，和 C++ 版逐位一致。

    pos = (n - 1) * p / 100
    lo = floor(pos)，hi = ceil(pos)
    结果 = v[lo] + (v[hi] - v[lo]) * (pos - lo)
    """
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(values[0])
    q = min(100.0, max(0.0, float(p)))
    s = sorted(float(v) for v in values)
    pos = (n - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


class NoiseFloor:
    """滑动窗口噪声底。

    典型用法是每收到一块音频就 update 一次，拿到当前底噪估计；
    threshold() 直接给出应该用的门限。
    """

    __slots__ = ("_hist", "_size", "_percentile", "_factor", "_min_rms")

    def __init__(self, size=180, pct=20.0, factor=2.4, min_rms=0.008):
        self._size = max(1, int(size))
        self._hist = deque(maxlen=self._size)
        self._percentile = float(pct)
        self._factor = float(factor)
        self._min_rms = float(min_rms)

    def update(self, rms):
        """喂一个 RMS 值，返回更新后的底噪估计。"""
        try:
            v = float(rms)
        except (TypeError, ValueError):
            v = 0.0
        if not math.isfinite(v) or v < 0.0:
            v = 0.0
        self._hist.append(v)
        return self.floor()

    def floor(self):
        """当前底噪估计。窗口为空时返回 0。"""
        if not self._hist:
            return 0.0
        return percentile(self._hist, self._percentile)

    def threshold(self, scale=1.0):
        """当前门限 = max(min_rms, 底噪 × factor) × scale。

        scale 用来叠场景系数（远场 0.65、强制放大 0.45）。
        """
        base = max(self._min_rms, self.floor() * self._factor)
        return base * float(scale)

    def reset(self):
        """清空历史。换设备、重启会话时调用。"""
        self._hist.clear()

    def __len__(self):
        return len(self._hist)

    def __repr__(self):
        return "NoiseFloor(n=%d, floor=%.5f, threshold=%.5f)" % (
            len(self._hist), self.floor(), self.threshold())
