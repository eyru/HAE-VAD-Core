"""HAE-VAD Core —— 流式语音活动检测与切句。

    Hearizon Audiology Engine — Voice Activity Detection

一句话：把连续的麦克风音频流切成「一句一句」，并且不把词首词尾切掉。

    from hae_vad import Segmenter, FAR

    seg = Segmenter(rate=16000, config=FAR)
    for block in mic_blocks:
        for sentence in seg.push(block):
            asr(sentence)

设计取舍见 docs/ALGORITHM.md。
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "AGGRESSIVE",
    "DEFAULT",
    "FAR",
    "NEAR",
    "NOISY",
    "NoiseFloor",
    "Segmenter",
    "State",
    "VadConfig",
    "config_preset",
    "config_names",
    "energy_envelope",
    "percentile",
    "rms",
    "read_info",
    "read_wav",
    "trim_tail_valley",
    "trim_to_valley",
    "write_wav",
    "zero_crossing_rate",
]

from .config import AGGRESSIVE, DEFAULT, FAR, NEAR, NOISY, VadConfig
from .config import get as config_preset
from .config import names as config_names
from .noise import NoiseFloor, percentile, rms
from .segmenter import Segmenter, State, zero_crossing_rate
from .trim import energy_envelope, trim_tail_valley, trim_to_valley
from .wav import read_wav, write_wav
from .wav import read_info
