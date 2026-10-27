"""最小 WAV 读写，只用标准库。

不引 soundfile / scipy：这个库的卖点是「拿到就能跑」，为读写文件拉一个
几十 MB 的依赖不值。只支持 PCM 16/24/32 位与 float32，覆盖了实际会遇到的全部情况。
"""

from __future__ import annotations

import os
import struct
import wave

import numpy as np

__all__ = ["read_wav", "write_wav", "WavInfo"]


def _prepare(path):
    """确保父目录存在。目录不存在时 wave.open 的报错很难看懂，这里提前处理。"""
    d = os.path.dirname(os.path.abspath(str(path)))
    if d:
        os.makedirs(d, exist_ok=True)
    return str(path)


class WavInfo:
    """读进来的音频描述。"""

    __slots__ = ("rate", "channels", "samples")

    def __init__(self, rate, channels, samples):
        self.rate = rate
        self.channels = channels
        self.samples = samples

    def __repr__(self):
        dur = self.samples / float(self.rate) if self.rate else 0.0
        return "WavInfo(rate=%d, channels=%d, samples=%d, %.2fs)" % (
            self.rate, self.channels, self.samples, dur)


def read_wav(path):
    """读 WAV，返回 (float32 单声道数组, 采样率)。

    多声道会混成单声道（取平均），因为语音前处理只需要一条信号。
    """
    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)

    if width == 1:              # 8 位是无符号
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif width == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 3:            # 24 位：手工拼
        b = np.frombuffer(raw, dtype=np.uint8)
        b = b[: (b.size // 3) * 3].reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        a = v.astype(np.float64) / float(1 << 23)
    elif width == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError("不支持的位宽：%d 字节" % width)

    if ch > 1:
        a = a[: (a.size // ch) * ch].reshape(-1, ch).mean(axis=1)
    return a.astype(np.float32), rate


def read_info(path):
    """只读文件头，拿到时长等信息，不加载样本。"""
    with wave.open(str(path), "rb") as w:
        return WavInfo(w.getframerate(), w.getnchannels(), w.getnframes())


def write_wav(path, audio, rate=16000, bits=16):
    """写 16 位 PCM WAV（默认）。写之前在 -1..1 之间夹一次，防溢出。"""
    a = np.asarray(audio, dtype=np.float64).ravel()
    a = np.clip(a, -1.0, 1.0)
    if bits == 16:
        pcm = (a * 32767.0).round().astype("<i2").tobytes()
        width = 2
    elif bits == 32:
        pcm = (a * 2147483647.0).round().astype("<i4").tobytes()
        width = 4
    else:
        raise ValueError("只支持 bits=16 或 32")
    with wave.open(_prepare(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(int(rate))
        w.writeframes(pcm)
    return path


def write_wav_stereo_compare(path, original, processed, rate=16000):
    """把原声与处理后并排写成双声道，方便耳朵直接 A/B。"""
    a = np.clip(np.asarray(original, dtype=np.float64).ravel(), -1.0, 1.0)
    b = np.clip(np.asarray(processed, dtype=np.float64).ravel(), -1.0, 1.0)
    n = min(a.size, b.size)
    inter = np.empty(n * 2, dtype="<i2")
    inter[0::2] = (a[:n] * 32767.0).round()
    inter[1::2] = (b[:n] * 32767.0).round()
    with wave.open(_prepare(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(inter.tobytes())
    return path
