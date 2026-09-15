// ============================================================================
// bench 子命令：性能基准测试
//   扫描 分辨率 x 通道数 x 参数组合 x kernel 版本，输出 CSV 性能日志。
//   每配置 warmup 预热 + repeat 次测量，kernel_ms 取最小值，e2e_ms 取均值。
//   CSV 列与架构设计 §10.2 一致：
//   size,channels,pr,sr,h,sigma,kernel_ver,kernel_ms,e2e_ms,mpx_s,cpu_ms,speedup
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "core/nlm_cpu_ref.h"
#include "nlm/pipeline.h"
#include "tester/utils.h"

namespace {

struct SizeSpec { int w, h; };
struct ParamSet { const char* name; NlmParams p; };

// 解析 "1920x1080,3840x2160" 形式
std::vector<SizeSpec> ParseSizes(const std::string& spec) {
    std::vector<SizeSpec> out;
    size_t pos = 0;
    while (pos < spec.size()) {
        size_t comma = spec.find(',', pos);
        std::string tok = spec.substr(pos, comma == std::string::npos ? comma : comma - pos);
        int w = 0, h = 0;
        if (std::sscanf(tok.c_str(), "%dx%d", &w, &h) == 2 && w > 0 && h > 0)
            out.push_back(SizeSpec{w, h});
        pos = (comma == std::string::npos) ? spec.size() : comma + 1;
    }
    return out;
}

std::vector<int> ParseChannels(const std::string& spec) {
    std::vector<int> out;
    size_t pos = 0;
    while (pos < spec.size()) {
        size_t comma = spec.find(',', pos);
        int c = std::atoi(spec.substr(pos, comma == std::string::npos ? comma : comma - pos).c_str());
        if (c == 1 || c == 3) out.push_back(c);
        pos = (comma == std::string::npos) ? spec.size() : comma + 1;
    }
    return out;
}

} // namespace

int RunBenchmark(const std::string& sizes_spec, const std::string& channels_spec,
                 const std::string& csv_path, int warmup, int repeat, bool with_cpu) {
    std::vector<SizeSpec> sizes = ParseSizes(sizes_spec);
    std::vector<int> channels = ParseChannels(channels_spec);
    if (sizes.empty() || channels.empty()) {
        std::fprintf(stderr, "[bench] 无效的 --sizes 或 --channels 规格\n");
        return 2;
    }
    if (warmup < 0) warmup = 0;
    if (repeat < 1) repeat = 1;

    // 参数组合：任务基线 + 变体（覆盖 pr/sr 上下扫描）
    NlmParams p_base;                                   // pr=3, sr=10, h=10, sigma=25（任务基线）
    NlmParams p_strong; p_strong.h = 15.0f;             // 强滤波
    NlmParams p_small;  p_small.patch_radius = 2; p_small.search_radius = 7;
    p_small.h = 10.0f; p_small.sigma = 15.0f;           // 轻量
    NlmParams p_large;  p_large.patch_radius = 4; p_large.search_radius = 14;
    p_large.h = 12.0f; p_large.sigma = 35.0f;           // 压力
    const ParamSet sets[] = {
        {"base", p_base}, {"strong-h", p_strong}, {"small", p_small}, {"large", p_large},
    };
    const int kernel_versions[] = {0, 1, 2};

    std::ofstream csv(csv_path.c_str());
    if (!csv) {
        std::fprintf(stderr, "[bench] 无法写入 CSV: %s\n", csv_path.c_str());
        return 2;
    }
    csv << "size,channels,pr,sr,h,sigma,kernel_ver,kernel_ms,e2e_ms,mpx_s,cpu_ms,speedup\n";

    std::printf("[bench] warmup=%d repeat=%d，输出 %s\n", warmup, repeat, csv_path.c_str());

    for (size_t si = 0; si < sizes.size(); ++si) {
        const int w = sizes[si].w, h = sizes[si].h;
        for (size_t ci = 0; ci < channels.size(); ++ci) {
            const int c = channels[ci];

            // CPU 参考耗时：仅对 base 参数且规模 <= 1080p 灰度量级时测量（单线程极慢）
            double cpu_ms = -1.0;
            if (with_cpu && (size_t)w * h * c <= 1920u * 1080u) {
                ImageU8 src_cpu = MakeSyntheticImage(w, h, c, p_base.sigma, 12345u + c);
                ImageU8 dst_cpu;
                HostTimer t;
                t.Start();
                NlmDenoiseCpuRef(src_cpu, &dst_cpu, p_base);
                cpu_ms = t.Ms();
                std::printf("[bench] CPU 参考 %dx%dx%d: %.1f ms\n", w, h, c, cpu_ms);
            }

            for (size_t pi = 0; pi < sizeof(sets) / sizeof(sets[0]); ++pi) {
                const NlmParams& prm = sets[pi].p;
                ImageU8 src = MakeSyntheticImage(w, h, c, prm.sigma, 777u + (unsigned)c);
                for (size_t ki = 0; ki < sizeof(kernel_versions) / sizeof(kernel_versions[0]); ++ki) {
                    const int ver = kernel_versions[ki];
                    double best_kernel = 1e30, sum_e2e = 0.0;
                    std::string err;
                    bool ok = true;
                    for (int it = 0; it < warmup + repeat; ++it) {
                        ImageU8 dst;
                        NlmPerfReport perf;
                        if (!NlmDenoiseGpu(src, &dst, prm, ver, &perf, &err)) {
                            std::fprintf(stderr, "[bench] v%d 失败: %s\n", ver, err.c_str());
                            ok = false;
                            break;
                        }
                        if (it >= warmup) {
                            if (perf.kernel_ms < best_kernel) best_kernel = perf.kernel_ms;
                            sum_e2e += perf.e2e_ms;
                        }
                    }
                    if (!ok) return 2;
                    const double mean_e2e = sum_e2e / repeat;
                    const double mpx_s = best_kernel > 0.0
                        ? (double)w * h / (best_kernel * 1e3) : 0.0;
                    const double speedup = (cpu_ms > 0.0) ? cpu_ms / mean_e2e : -1.0;
                    csv << w << "x" << h << ',' << c << ','
                        << prm.patch_radius << ',' << prm.search_radius << ','
                        << prm.h << ',' << prm.sigma << ','
                        << ver << ',' << best_kernel << ',' << mean_e2e << ','
                        << mpx_s << ','
                        << ((cpu_ms > 0.0) ? cpu_ms : -1.0) << ',' << speedup << '\n';
                    std::printf("[bench] %dx%dx%d %-8s v%d: kernel=%.3f ms  e2e=%.3f ms  %.1f Mpx/s%s\n",
                                w, h, c, sets[pi].name, ver, best_kernel, mean_e2e, mpx_s,
                                speedup > 0.0 ?
                                    (std::string("  speedup=") + std::to_string(speedup)).c_str() : "");
                }
            }
        }
    }
    std::printf("[bench] 完成，CSV 已写入 %s\n", csv_path.c_str());
    return 0;
}
