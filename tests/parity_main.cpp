// HAE-VAD 的 Python / C++ 对拍程序。
//
// 生成一段确定性的伪随机音频（与 tests/run_parity.py 里的生成器逐位一致），
// 用四种预设各跑一遍流式切句，把结果打印成固定格式。
// run_parity.py 会同时跑 Python 版，逐行比对。
//
// 只依赖 iostream 与头文件本身，不需要第三方库。

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "hae_vad.h"

namespace {

// 与 Python 版 run_parity.py 的 gen_audio() 完全一致：
// 线性同余发生器，返回 [-1, 1) 的浮点。
struct Lcg {
    std::uint32_t s;
    explicit Lcg(std::uint32_t seed) : s(seed) {}
    float next() {
        s = s * 1664525u + 1013904223u;
        const std::uint32_t bits = (s >> 8) & 0x00FFFFFFu;
        return static_cast<float>(bits) / 16777216.0f * 2.0f - 1.0f;
    }
};

// 合成一段「说话—停顿—说话—停顿—说话」的音频。
std::vector<float> genAudio(int rate) {
    const auto msToN = [rate](double ms) {
        return static_cast<std::size_t>(static_cast<double>(rate) * ms / 1000.0);
    };
    std::vector<float> out;
    out.reserve(msToN(8000));

    const auto pushSilence = [&](double ms) {
        const std::size_t n = msToN(ms);
        for (std::size_t i = 0; i < n; ++i) out.push_back(0.0f);
    };
    const auto pushTone = [&](double ms, float amp, double freq) {
        const std::size_t n = msToN(ms);
        for (std::size_t i = 0; i < n; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(rate);
            out.push_back(static_cast<float>(amp * std::sin(2.0 * 3.14159265358979323846 * freq * t)));
        }
    };

    pushSilence(500.0);
    pushTone(900.0, 0.08f, 220.0);
    pushSilence(1200.0);
    pushTone(700.0, 0.06f, 180.0);
    pushSilence(1100.0);
    pushTone(1300.0, 0.10f, 260.0);
    pushSilence(1200.0);

    // 叠一点底噪，避免全零让门限判断退化
    Lcg rng(20260101u);
    for (auto& v : out) v += rng.next() * 0.0015f;
    return out;
}

std::string joinLens(const std::vector<std::vector<float>>& segs) {
    std::string s;
    for (std::size_t i = 0; i < segs.size(); ++i) {
        if (i != 0) s += ",";
        s += std::to_string(segs[i].size());
    }
    return s;
}

// 把浮点结果压成一个可比较的整数字节串（保留 6 位小数），避免格式差异。
std::string fmt(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6f", v);
    return std::string(buf);
}

}  // namespace

int main() {
    std::printf("#BEGIN#\n");
    const int rate = 16000;
    const std::vector<float> audio = genAudio(rate);
    const std::size_t blockSize = 480;  // 30 ms

    std::printf("rate %d\n", rate);
    std::printf("samples %zu\n", audio.size());
    std::printf("rms %s\n", fmt(hae::vad::rms(audio.data(), audio.size())).c_str());
    std::printf("zcr %s\n", fmt(hae::vad::zeroCrossingRate(audio.data(), audio.size())).c_str());

    {
        std::vector<double> vals;
        for (std::size_t i = 0; i < 64; ++i) {
            vals.push_back(static_cast<double>(i) * 0.5);
        }
        std::printf("pct20 %s\n", fmt(hae::vad::percentile(vals, 20.0)).c_str());
        std::printf("pct50 %s\n", fmt(hae::vad::percentile(vals, 50.0)).c_str());
        std::printf("pct95 %s\n", fmt(hae::vad::percentile(vals, 95.0)).c_str());
    }

    const struct {
        const char* name;
        hae::vad::Config cfg;
    } presets[] = {
        {"near", hae::vad::presetNear()},
        {"far", hae::vad::presetFar()},
        {"noisy", hae::vad::presetNoisy()},
        {"aggressive", hae::vad::presetAggressive()},
    };

    for (const auto& p : presets) {
        double fakeNow = 0.0;
        hae::vad::Segmenter seg(rate, p.cfg, 1.0, [&fakeNow] { return fakeNow; });
        std::vector<std::vector<float>> got;
        std::size_t partialCalls = 0;
        std::size_t partialNonEmpty = 0;
        std::size_t lastPartial = 0;

        for (std::size_t i = 0; i < audio.size(); i += blockSize) {
            const std::size_t n = std::min(blockSize, audio.size() - i);
            auto batch = seg.push(audio.data() + i, n);
            for (auto& s : batch) got.push_back(std::move(s));

            fakeNow += 0.03;
            const auto part = seg.partial();
            ++partialCalls;
            if (!part.empty()) {
                ++partialNonEmpty;
                lastPartial = part.size();
            }
        }
        auto tail = seg.flush();
        for (auto& s : tail) got.push_back(std::move(s));

        std::printf("preset %s count %zu lens %s\n", p.name, got.size(),
                    joinLens(got).c_str());
        std::printf("preset %s partial calls %zu nonempty %zu last %zu\n", p.name,
                    partialCalls, partialNonEmpty, lastPartial);
        std::printf("preset %s floor %s\n", p.name, fmt(seg.noiseFloor()).c_str());
    }

    // 尾部裁切：用整段音频当一句，看裁完剩多少
    {
        std::vector<float> seg(audio.begin(), audio.end());
        const auto cut = hae::vad::trimTailValley(seg, rate);
        std::printf("trim_tail %zu\n", cut.size());
        const auto fast = hae::vad::trimToValley(seg, rate);
        std::printf("trim_fast %zu\n", fast.size());
    }

    return 0;
}
