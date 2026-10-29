// HAE-VAD Core —— 流式语音活动检测与切句（C++ 版）
//
//     Hearizon Audiology Engine — Voice Activity Detection
//
// 单头文件，C++17，只依赖标准库。与 Python 版 hae_vad 逐样本一致，
// tests/run_parity.py 会自动验证。
//
// 用法：
//
//     hae::vad::Segmenter seg(16000, hae::vad::presetByName("far"));
//     for (每个音频块) {
//         auto sentences = seg.push(block.data(), block.size());
//         for (auto& s : sentences) asr(s);
//     }
//     for (auto& s : seg.flush()) asr(s);
//
// 设计说明见 docs/ALGORITHM.md。
//
// 依赖：只有标准库。C++17。
//
// SPDX-License-Identifier: MIT

#ifndef HAE_VAD_H
#define HAE_VAD_H

#include <algorithm>
#include <chrono>
#include <functional>
#include <cmath>
#include <cstddef>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace hae {
namespace vad {

constexpr double kEps = 1e-12;

// ---------------------------------------------------------------- 配置
struct Config {
    // 门限
    double min_rms = 0.008;
    double zcr_max = 0.45;
    int floor_size = 180;
    double floor_percentile = 20.0;
    double floor_factor = 2.4;
    double far_field_scale = 1.0;   // 1.0 = 不降门限，远场类预设才小于 1
    double force_loud_scale = 0.45;

    // 节奏
    double block_ms = 30.0;
    double hang_ms = 700.0;
    double min_speech_ms = 400.0;
    double max_speech_ms = 12000.0;
    double preroll_ms = 300.0;
    bool hang_blocks_keep = true;

    // 收尾裁切
    double valley_ms = 320.0;
    double valley_ratio = 0.02;
    double valley_pad_ms = 40.0;
    double valley_min_ms = 40.0;

    // 流式
    double stream_window_s = 6.0;
    double stream_interval_s = 0.32;
    double stream_min_s = 0.30;
};

//: 近讲：贴着电脑说话。
inline Config presetNear() { return Config(); }

//: 远场：四五米以上。门限放宽，尾音留长，避免把慢速说话的停顿切断。
inline Config presetFar() {
    Config c;
    c.far_field_scale = 0.65;
    c.hang_ms = 900.0;
    c.max_speech_ms = 15000.0;
    return c;
}

//: 嘈杂环境：空调、风扇、多人。门限抬高，过零率收紧。
inline Config presetNoisy() {
    Config c;
    c.floor_factor = 3.2;
    c.min_rms = 0.012;
    c.zcr_max = 0.50;
    return c;
}

//: 强制模式：极小声也要收进来，代价是误触发变多。
inline Config presetAggressive() {
    Config c;
    c.far_field_scale = 0.45;
    c.min_rms = 0.005;
    c.hang_ms = 900.0;
    c.max_speech_ms = 15000.0;
    return c;
}

//: 按名字取预设。名字不认识时返回 near，并把 ok 置 false。
inline Config presetByName(const char* name, bool* ok = nullptr) {
    if (ok != nullptr) *ok = true;
    if (name == nullptr) {
        if (ok != nullptr) *ok = false;
        return Config();
    }
    std::string n(name);
    for (auto& ch : n) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (n == "near" || n == "default") return presetNear();
    if (n == "far") return presetFar();
    if (n == "noisy") return presetNoisy();
    if (n == "aggressive") return presetAggressive();
    if (ok != nullptr) *ok = false;
    return Config();
}

// ---------------------------------------------------------------- 基础度量
//: 线性插值分位数。公式和 Python 版逐位一致：pos = (n-1)*p/100。
inline double percentile(std::vector<double> values, double p) {
    const std::size_t n = values.size();
    if (n == 0) return 0.0;
    if (n == 1) return values[0];
    const double q = std::min(100.0, std::max(0.0, p));
    // 稳定排序：重复值不改变顺序，和 numpy 的 sort 行为一致
    std::stable_sort(values.begin(), values.end());
    const double pos = static_cast<double>(n - 1) * q / 100.0;
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    if (lo == hi) return values[lo];
    const double lo_v = values[lo];
    const double hi_v = values[hi];
    return lo_v + (hi_v - lo_v) * (pos - static_cast<double>(lo));
}

//: 过零率：相邻样本符号变化的比例。
inline double zeroCrossingRate(const float* p, std::size_t n) {
    if (p == nullptr || n < 2) return 0.0;
    std::size_t changes = 0;
    bool prev = std::signbit(static_cast<double>(p[0]));
    for (std::size_t i = 1; i < n; ++i) {
        const bool cur = std::signbit(static_cast<double>(p[i]));
        if (cur != prev) ++changes;
        prev = cur;
    }
    return static_cast<double>(changes) / static_cast<double>(n - 1);
}

//: 均方根，double 累加。
inline double rms(const float* p, std::size_t n) {
    if (p == nullptr || n == 0) return 0.0;
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = static_cast<double>(p[i]);
        acc += v * v;
    }
    return std::sqrt(acc / static_cast<double>(n));
}

// ---------------------------------------------------------------- 噪声底
class NoiseFloor {
public:
    NoiseFloor(int size = 180, double pct = 20.0, double factor = 2.4,
               double min_rms = 0.008)
        : size_(size > 0 ? size : 1),
          pct_(pct),
          factor_(factor),
          min_rms_(min_rms) {
        hist_.reserve(static_cast<std::size_t>(size_));
    }

    //: 喂一个 RMS 值，返回更新后的底噪估计。
    double update(double v) {
        if (!std::isfinite(v) || v < 0.0) v = 0.0;
        hist_.push_back(v);
        if (static_cast<int>(hist_.size()) > size_) {
            hist_.erase(hist_.begin());
        }
        return floor();
    }

    //: 当前底噪估计。
    double floor() const {
        if (hist_.empty()) return 0.0;
        return percentile(hist_, pct_);
    }

    //: 当前门限 = max(min_rms, 底噪 * factor) * scale。
    double threshold(double scale = 1.0) const {
        const double base = std::max(min_rms_, floor() * factor_);
        return base * scale;
    }

    void reset() { hist_.clear(); }
    std::size_t count() const { return hist_.size(); }

private:
    int size_;
    double pct_;
    double factor_;
    double min_rms_;
    std::vector<double> hist_;
};

// ---------------------------------------------------------------- 尾部裁切
//: 按 win_ms 一窗算能量包络。
inline std::vector<double> energyEnvelope(const std::vector<float>& seg, int rate,
                                          double win_ms = 10.0) {
    std::vector<double> out;
    const std::size_t step = static_cast<std::size_t>(
        std::max(1, static_cast<int>(static_cast<double>(rate) * win_ms / 1000.0)));
    if (seg.size() < step) return out;
    const std::size_t count = (seg.size() - step) / step + 1;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        double acc = 0.0;
        const std::size_t base = i * step;
        for (std::size_t k = 0; k < step; ++k) {
            const double v = static_cast<double>(seg[base + k]);
            acc += v * v;
        }
        out.push_back(acc / static_cast<double>(step));
    }
    return out;
}

//: 把片段尾部多余的静音切到能量谷底。规则见 Python 版同名函数。
inline std::vector<float> trimTailValley(const std::vector<float>& seg, int rate,
                                         double search_ms = 320.0,
                                         double ratio = 0.02,
                                         double pad_ms = 40.0,
                                         double min_ms = 40.0,
                                         double win_ms = 10.0) {
    const std::size_t n = seg.size();
    if (n == 0) return seg;
    const std::size_t step = static_cast<std::size_t>(
        std::max(1, static_cast<int>(static_cast<double>(rate) * win_ms / 1000.0)));
    if (n <= step * 4) return seg;
    if (static_cast<double>(n) < static_cast<double>(rate) * min_ms / 1000.0) return seg;

    const std::vector<double> env = energyEnvelope(seg, rate, win_ms);
    if (env.empty()) return seg;

    double peak = 0.0;
    for (double e : env) peak = std::max(peak, e);
    if (peak <= kEps) return seg;

    std::size_t cut = n;
    for (std::size_t i = env.size(); i-- > 0;) {
        if (env[i] >= peak * ratio) {
            cut = (i + 1) * step;
            break;
        }
    }

    const std::size_t pad = static_cast<std::size_t>(
        static_cast<double>(rate) * pad_ms / 1000.0);
    cut = std::min(n, cut + pad);
    if (cut <= step * 4) return seg;
    return std::vector<float>(seg.begin(), seg.begin() + static_cast<std::ptrdiff_t>(cut));
}

//: 流式临时结果用的轻量裁切：从尾部剥掉连续安静窗口。
inline std::vector<float> trimToValley(const std::vector<float>& seg, int rate,
                                       double search_ms = 320.0) {
    const std::size_t n = seg.size();
    if (n == 0) return seg;
    const std::size_t step = static_cast<std::size_t>(
        std::max(1, static_cast<int>(static_cast<double>(rate) * 0.01)));
    const std::size_t search = static_cast<std::size_t>(
        std::max(1, static_cast<int>(static_cast<double>(rate) * search_ms / 1000.0)));
    const std::size_t start = n > search ? n - search : 0;
    if (start >= n) return seg;

    double acc = 0.0;
    for (std::size_t i = start; i < n; ++i) {
        const double v = static_cast<double>(seg[i]);
        acc += v * v;
    }
    const double mean = acc / static_cast<double>(n - start);
    if (mean <= kEps) return seg;

    std::size_t cut = n;
    for (std::size_t i = n - step; i > start; i -= step) {
        double a = 0.0;
        for (std::size_t k = 0; k < step; ++k) {
            const double v = static_cast<double>(seg[i + k]);
            a += v * v;
        }
        if (a / static_cast<double>(step) >= mean) break;
        cut = i;
    }
    if (cut <= step * 4) return seg;
    return std::vector<float>(seg.begin(), seg.begin() + static_cast<std::ptrdiff_t>(cut));
}

// ---------------------------------------------------------------- 状态机
class Segmenter {
public:
    explicit Segmenter(int rate = 16000, const Config& cfg = Config(), double scale = 1.0,
                       std::function<double()> clock = {})
        : rate_(rate > 0 ? rate : 16000),
          cfg_(cfg),
          scale_(scale),
          clock_(clock ? std::move(clock) : [] {
              return std::chrono::duration<double>(
                         std::chrono::steady_clock::now().time_since_epoch())
                  .count();
          }),
          floor_(cfg.floor_size, cfg.floor_percentile, cfg.floor_factor, cfg.min_rms) {}

    //: 喂一块音频，返回这次切出来的句子（通常 0 或 1 个）。
    std::vector<std::vector<float>> push(const float* p, std::size_t n) {
        std::vector<std::vector<float>> out;
        if (p == nullptr || n == 0) return out;

        block_.assign(p, p + n);
        const double ms = static_cast<double>(n) * 1000.0 / static_cast<double>(rate_);

        level_ = rms(block_.data(), block_.size());
        zcr_ = zeroCrossingRate(block_.data(), block_.size());

        // 先按「上一块为止」的底噪算门限，再用它判断这一块。
        // 顺序反了的话，一块响亮的语音会把门限抬到自己之上。
        threshold_ = floor_.threshold(scale_);
        voiced_ = (level_ > threshold_) && (zcr_ < cfg_.zcr_max);

        // 底噪只在「不是人声」时更新，否则连续说几句就会把门限顶到语音电平。
        if (!voiced_) floor_.update(level_);

        if (voiced_) {
            if (!active_) {
                active_ = true;
                buf_ = pre_;
                pre_.clear();
                pre_samples_ = 0;
            }
            silence_ms_ = 0.0;
            buf_.push_back(block_);
            speech_ms_ += ms;
            appendTail(block_);
        } else if (active_) {
            if (cfg_.hang_blocks_keep) buf_.push_back(block_);
            appendTail(block_);
            silence_ms_ += ms;
        } else {
            appendPre(block_);
        }

        if (shouldCut()) out = cut();
        return out;
    }

    //: 会话结束时把还没切出去的音频吐出来。
    std::vector<std::vector<float>> flush() {
        std::vector<std::vector<float>> out;
        if (!active_) {
            reset();
            return out;
        }
        silence_ms_ = std::max(silence_ms_, cfg_.hang_ms);
        return cut();
    }

    //: 流式临时结果：最近 stream_window_s 秒，尾部裁到能量谷底。
    //: 与 Python 版一致：不到 stream_interval_s 就返回空，调用方跳过刷新。
    std::vector<float> partial() {
        if (!active_) return {};
        if (speech_ms_ / 1000.0 < cfg_.stream_min_s) return {};
        const double now = clock_();
        if (now - lastPartial_ < cfg_.stream_interval_s) return {};
        lastPartial_ = now;
        if (tail_.empty()) return {};
        std::vector<float> seg = concat(tail_);
        return trimToValley(seg, rate_, cfg_.valley_ms);
    }

    void reset() {
        floor_.reset();
        pre_.clear();
        pre_samples_ = 0;
        tail_.clear();
        tail_samples_ = 0;
        buf_.clear();
        active_ = false;
        speech_ms_ = 0.0;
        silence_ms_ = 0.0;
        recorded_ = 0;
        level_ = 0.0;
        zcr_ = 0.0;
        voiced_ = false;
        lastPartial_ = -1e9;
    }

    void setScale(double v) { scale_ = v; }
    void setConfig(const Config& c) {
        cfg_ = c;
        floor_ = NoiseFloor(c.floor_size, c.floor_percentile, c.floor_factor, c.min_rms);
    }

    bool active() const { return active_; }
    double level() const { return level_; }
    double threshold() const { return threshold_; }
    double noiseFloor() const { return floor_.floor(); }
    double speechMs() const { return speech_ms_; }
    double silenceMs() const { return silence_ms_; }
    double zcr() const { return zcr_; }
    bool voiced() const { return voiced_; }
    std::size_t segments() const { return recorded_; }
    const Config& config() const { return cfg_; }

private:
    static std::vector<float> concat(const std::vector<std::vector<float>>& parts) {
        std::size_t total = 0;
        for (const auto& b : parts) total += b.size();
        std::vector<float> out;
        out.reserve(total);
        for (const auto& b : parts) out.insert(out.end(), b.begin(), b.end());
        return out;
    }

    void appendPre(const std::vector<float>& a) {
        const std::size_t limit = static_cast<std::size_t>(
            static_cast<double>(rate_) * cfg_.preroll_ms / 1000.0);
        pre_.push_back(a);
        pre_samples_ += a.size();
        while (!pre_.empty() && pre_samples_ > limit) {
            pre_samples_ -= pre_.front().size();
            pre_.erase(pre_.begin());
        }
        const std::size_t hard = static_cast<std::size_t>(
            static_cast<double>(rate_) * (cfg_.max_speech_ms + cfg_.hang_ms) / 1000.0);
        if (pre_samples_ > hard) {
            pre_.clear();
            pre_samples_ = 0;
        }
    }

    void appendTail(const std::vector<float>& a) {
        const std::size_t limit = static_cast<std::size_t>(
            static_cast<double>(rate_) * cfg_.stream_window_s);
        tail_.push_back(a);
        tail_samples_ += a.size();
        while (!tail_.empty() && tail_samples_ > limit) {
            tail_samples_ -= tail_.front().size();
            tail_.erase(tail_.begin());
        }
    }

    bool shouldCut() const {
        if (!active_) return false;
        if (silence_ms_ >= cfg_.hang_ms) return true;
        if (speech_ms_ >= cfg_.max_speech_ms) return true;
        return false;
    }

    std::vector<std::vector<float>> cut() {
        std::vector<std::vector<float>> out;
        if (buf_.empty()) {
            active_ = false;
            resetSentence();
            return out;
        }
        std::vector<float> seg = concat(buf_);
        const double speech = speech_ms_;
        active_ = false;
        resetSentence();

        if (seg.empty() || speech < cfg_.min_speech_ms) return out;

        seg = trimTailValley(seg, rate_, cfg_.valley_ms, cfg_.valley_ratio,
                             cfg_.valley_pad_ms, cfg_.valley_min_ms);
        ++recorded_;
        out.push_back(std::move(seg));
        return out;
    }

    void resetSentence() {
        buf_.clear();
        tail_.clear();
        tail_samples_ = 0;
        speech_ms_ = 0.0;
        silence_ms_ = 0.0;
    }

    int rate_;
    Config cfg_;
    double scale_;
    NoiseFloor floor_;

    std::vector<float> block_;
    std::vector<std::vector<float>> pre_;
    std::size_t pre_samples_ = 0;
    std::vector<std::vector<float>> tail_;
    std::size_t tail_samples_ = 0;
    std::vector<std::vector<float>> buf_;

    bool active_ = false;
    double speech_ms_ = 0.0;
    double silence_ms_ = 0.0;
    double level_ = 0.0;
    double threshold_ = 0.008;
    double zcr_ = 0.0;
    bool voiced_ = false;
    std::size_t recorded_ = 0;
    std::function<double()> clock_;
    double lastPartial_ = -1e9;
};

}  // namespace vad
}  // namespace hae

#endif  // HAE_VAD_H
