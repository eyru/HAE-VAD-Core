"""切句状态机的单元测试。

所有测试都用合成信号：正弦波模拟语音（过零率低），白噪声模拟非人声
（过零率高），静音就是零。这样测试结果不依赖具体录音，可重复。
"""

import unittest

import numpy as np

import hae_vad
from hae_vad import Segmenter, VadConfig


RATE = 16000
BLOCK_MS = 30
BLOCK = int(RATE * BLOCK_MS / 1000)


def voice(ms, level=0.1, freq=180.0):
    """低频正弦，过零率远低于阈值，会被判成人声。"""
    n = int(RATE * ms / 1000.0)
    t = np.arange(n) / float(RATE)
    return (np.sin(2 * np.pi * freq * t) * level).astype(np.float32)


def silence(ms):
    n = int(RATE * ms / 1000.0)
    return np.zeros(n, dtype=np.float32)


def noise(ms, level=0.05, seed=3):
    """白噪声，过零率约 0.5，会被过零率判据挡掉。"""
    rng = np.random.default_rng(seed)
    n = int(RATE * ms / 1000.0)
    return (rng.standard_normal(n) * level).astype(np.float32)


def feed(seg, audio):
    """把整段音频按 BLOCK 切块喂进去，返回切出的所有句子。"""
    out = []
    for i in range(0, audio.size, BLOCK):
        out.extend(seg.push(audio[i:i + BLOCK]))
    return out


class TestBasics(unittest.TestCase):
    def test_silence_produces_nothing(self):
        seg = Segmenter(rate=RATE)
        got = feed(seg, silence(3000))
        self.assertEqual(got, [])

    def test_single_utterance(self):
        seg = Segmenter(rate=RATE)
        audio = np.concatenate([silence(300), voice(1200), silence(1200)])
        got = feed(seg, audio)
        self.assertEqual(len(got), 1)
        self.assertGreater(got[0].size, 0)

    def test_two_utterances(self):
        seg = Segmenter(rate=RATE)
        audio = np.concatenate([
            silence(300), voice(800), silence(1200),
            voice(800), silence(1200),
        ])
        got = feed(seg, audio)
        self.assertEqual(len(got), 2)

    def test_flush_returns_pending(self):
        seg = Segmenter(rate=RATE)
        seg.push(voice(800))          # 没有尾部静音，还没切
        self.assertEqual(seg.state().active, True)
        got = seg.flush()
        self.assertEqual(len(got), 1)

    def test_flush_on_idle_returns_nothing(self):
        seg = Segmenter(rate=RATE)
        self.assertEqual(seg.flush(), [])


class TestThresholds(unittest.TestCase):
    def test_short_blip_dropped(self):
        cfg = VadConfig(min_speech_ms=400.0)
        seg = Segmenter(rate=RATE, config=cfg)
        audio = np.concatenate([silence(300), voice(150), silence(1200)])
        self.assertEqual(feed(seg, audio), [])

    def test_max_speech_forces_cut(self):
        cfg = VadConfig(max_speech_ms=1000.0, hang_ms=700.0)
        seg = Segmenter(rate=RATE, config=cfg)
        got = feed(seg, voice(3000))     # 一直不停说
        self.assertGreaterEqual(len(got), 2)

    def test_noise_rejected_by_zcr(self):
        # 白噪声的过零率理论上就是 0.5，默认 0.45 的上限要能挡住它
        seg = Segmenter(rate=RATE)
        got = feed(seg, np.concatenate([silence(200), noise(2000), silence(1000)]))
        self.assertEqual(got, [])

    def test_low_freq_noise_is_not_rejected_by_zcr(self):
        """低频稳态噪声过零率低，过零率判据挡不住它——这是已知局限。

        真实的兜底靠噪声底自适应：只要有一段静音，底噪就建起来，
        门限随之抬高。这里验证「确实挡不住」，防止有人误以为 zcr 万能的。
        """
        rng = np.random.default_rng(11)
        n = int(RATE * 1.0)
        hum = (rng.standard_normal(n) * 0.02).astype(np.float32)
        # 一阶低通，做出低频为主的噪声
        for _ in range(8):
            hum = np.convolve(hum, np.ones(9, dtype=np.float32) / 9.0, mode="same").astype(np.float32)
        hum = (hum / (np.max(np.abs(hum)) + 1e-12) * 0.05).astype(np.float32)
        z = hae_vad.zero_crossing_rate(hum)
        self.assertLess(z, hae_vad.DEFAULT.zcr_max)

    def test_noise_floor_adapts(self):
        """环境变吵之后，只要出现静音块，底噪就会跟上。

        注意底噪只在「非人声」块更新，所以这里必须喂静音才能看到变化。
        一直喂响亮噪声是看不到抬升的，因为噪声块还在被误判成人声。
        """
        seg = Segmenter(rate=RATE)
        for _ in range(200):
            seg.push(silence(BLOCK_MS))
        quiet_thr = seg.threshold

        # 环境变吵：喂一段带底噪的静音（电平低于门限，会被判非人声）
        for _ in range(200):
            seg.push(silence(BLOCK_MS) + noise(BLOCK_MS, level=0.004))
        self.assertGreater(seg.threshold, quiet_thr)


class TestPreroll(unittest.TestCase):
    def test_preroll_keeps_onset(self):
        """起句时要把开口前那一段接上，否则词首会被切掉。"""
        cfg = VadConfig(preroll_ms=300.0)
        seg = Segmenter(rate=RATE, config=cfg)

        # 先喂一段极轻的音（低于门限），这会进预卷缓冲
        faint = voice(600, level=0.002)
        for i in range(0, faint.size, BLOCK):
            seg.push(faint[i:i + BLOCK])

        # 再喂正常音量的语音
        loud = voice(600, level=0.15)
        got = []
        for i in range(0, loud.size, BLOCK):
            got.extend(seg.push(loud[i:i + BLOCK]))
        got.extend(seg.flush())

        self.assertEqual(len(got), 1)
        # 片段应该比「只有响的部分」更长，说明预卷接上了
        self.assertGreater(got[0].size, loud.size)

    def test_preroll_capped(self):
        cfg = VadConfig(preroll_ms=120.0)
        seg = Segmenter(rate=RATE, config=cfg)
        for _ in range(100):
            seg.push(silence(BLOCK_MS))
        limit = int(RATE * 120 / 1000.0)
        self.assertLessEqual(seg._pre_samples, limit)


class TestTrim(unittest.TestCase):
    def test_trim_removes_tail_silence(self):
        seg_a = voice(600, level=0.15)
        tail = silence(800)
        audio = np.concatenate([seg_a, tail])
        out = hae_vad.trim_tail_valley(audio, rate=RATE, search_ms=320)
        self.assertLess(out.size, audio.size)
        self.assertGreater(out.size, seg_a.size * 0.8)

    def test_trim_tail_longer_than_search(self):
        """尾部静音比搜索窗口还长时也要能裁——峰值必须取整段。"""
        seg_a = voice(600, level=0.15)
        audio = np.concatenate([seg_a, silence(1500)])
        out = hae_vad.trim_tail_valley(audio, rate=RATE, search_ms=320)
        self.assertLess(out.size, audio.size)
        self.assertGreater(out.size, seg_a.size)

    def test_trim_keeps_all_voice(self):
        audio = voice(600, level=0.15)      # 全是有声，没有静音可裁
        out = hae_vad.trim_tail_valley(audio, rate=RATE)
        self.assertGreaterEqual(out.size, audio.size * 0.9)

    def test_trim_short_input_untouched(self):
        audio = voice(10, level=0.15)
        out = hae_vad.trim_tail_valley(audio, rate=RATE)
        self.assertEqual(out.size, audio.size)

    def test_trim_silent_input_untouched(self):
        audio = silence(2000)
        out = hae_vad.trim_tail_valley(audio, rate=RATE)
        self.assertEqual(out.size, audio.size)


class TestRobustness(unittest.TestCase):
    def test_empty_block(self):
        seg = Segmenter(rate=RATE)
        self.assertEqual(seg.push(None), [])
        self.assertEqual(seg.push(np.zeros(0, dtype=np.float32)), [])
        self.assertEqual(seg.push([]), [])

    def test_nan_block(self):
        seg = Segmenter(rate=RATE)
        bad = np.array([np.nan, np.inf, -np.inf, 0.1], dtype=np.float32)
        out = seg.push(bad)
        self.assertIsInstance(out, list)

    def test_stereo_input_takes_first_channel(self):
        seg = Segmenter(rate=RATE)
        stereo = np.stack([voice(BLOCK_MS, 0.15), voice(BLOCK_MS, 0.15)], axis=1)
        out = seg.push(stereo)
        self.assertIsInstance(out, list)

    def test_reset_clears_state(self):
        seg = Segmenter(rate=RATE)
        seg.push(voice(600))
        self.assertTrue(seg.state().active)
        seg.reset()
        st = seg.state()
        self.assertFalse(st.active)
        self.assertEqual(st.speech_ms, 0.0)
        self.assertEqual(st.buffered_samples, 0)

    def test_config_type_checked(self):
        with self.assertRaises(TypeError):
            Segmenter(rate=RATE, config="far")


class TestConfig(unittest.TestCase):
    def test_presets_listed(self):
        names = hae_vad.config_names()
        self.assertIn("near", names)
        self.assertIn("far", names)
        self.assertIn("noisy", names)
        self.assertIn("aggressive", names)

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            hae_vad.config_preset("nope")

    def test_with_rejects_unknown_field(self):
        with self.assertRaises(TypeError):
            VadConfig().with_(nope=1)

    def test_with_returns_new_object(self):
        a = VadConfig()
        b = a.with_(hang_ms=1000.0)
        self.assertNotEqual(a.hang_ms, b.hang_ms)
        self.assertEqual(a.hang_ms, 700.0)

    def test_preset_ordering(self):
        # 越激进的预设门限系数应该越小
        self.assertLess(hae_vad.AGGRESSIVE.far_field_scale, hae_vad.FAR.far_field_scale)


class TestPartial(unittest.TestCase):
    def test_partial_none_when_idle(self):
        seg = Segmenter(rate=RATE)
        self.assertIsNone(seg.partial())

    def test_partial_after_speech(self):
        seg = Segmenter(rate=RATE, config=VadConfig(stream_min_s=0.1,
                                                    stream_interval_s=0.0))
        seg.push(voice(400))
        got = seg.partial()
        self.assertIsNotNone(got)
        self.assertGreater(got.size, 0)

    def test_partial_respects_interval(self):
        seg = Segmenter(rate=RATE, config=VadConfig(stream_min_s=0.1,
                                                    stream_interval_s=100.0))
        seg.push(voice(400))
        self.assertIsNotNone(seg.partial())
        self.assertIsNone(seg.partial())     # 间隔没到


class TestZeroCrossing(unittest.TestCase):
    def test_sine_low_zcr(self):
        z = hae_vad.zero_crossing_rate(voice(100, 0.1, freq=180.0))
        self.assertLess(z, 0.1)

    def test_noise_high_zcr(self):
        z = hae_vad.zero_crossing_rate(noise(100, 0.1))
        self.assertGreater(z, 0.4)

    def test_empty_zcr(self):
        self.assertEqual(hae_vad.zero_crossing_rate(np.zeros(0)), 0.0)


class TestPercentile(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(hae_vad.percentile([1, 2, 3, 4, 5], 50), 3.0, places=9)

    def test_bounds(self):
        self.assertAlmostEqual(hae_vad.percentile([1, 2, 3, 4], 0), 1.0, places=9)
        self.assertAlmostEqual(hae_vad.percentile([1, 2, 3, 4], 100), 4.0, places=9)

    def test_single(self):
        self.assertAlmostEqual(hae_vad.percentile([7.5], 50), 7.5, places=9)

    def test_empty(self):
        self.assertEqual(hae_vad.percentile([], 50), 0.0)


if __name__ == "__main__":
    unittest.main()
