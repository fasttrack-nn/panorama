// Standalone dot-product kernel benchmark for Panorama flat levels.
// Compile: ./bench.sh
// Usage:   ./kernel_bench

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include <functional>

// ---------------------------------------------------------------------------
// GCC imprecise FP pragmas (same as Faiss platform_macros.h)
// ---------------------------------------------------------------------------
#define IMPRECISE_FUNCTION_BEGIN \
    _Pragma("GCC push_options") \
    _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define IMPRECISE_FUNCTION_END _Pragma("GCC pop_options")

static inline uint64_t rdtsc() {
    uint32_t lo, hi;
    asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static void do_not_optimize(void* p) {
    asm volatile("" : : "g"(p) : "memory");
}

static constexpr size_t DIM = 128;
static constexpr size_t N_LEVELS = 4;
static constexpr size_t LEVEL_WIDTH = DIM / N_LEVELS;  // 32
static constexpr size_t BATCH_SIZE = 1024;
static constexpr int WARMUP_ITERS = 500;
static constexpr int BENCH_ITERS = 5000;

// ===================================================================
// ALL VARIANTS — __attribute__((noinline)) to prevent cross-function
// constant propagation. This simulates the real call from Panorama.h
// where the width comes from a struct member.
// ===================================================================

// V0: Baseline — current Panorama.h code, runtime width, batch-of-4
IMPRECISE_FUNCTION_BEGIN
__attribute__((noinline))
static void v0_baseline(
        const float* query_level,
        const float* level_storage,
        size_t num_active,
        size_t level_width_dims,
        float* dot_products) {
    size_t i = 0;
    for (; i + 4 <= num_active; i += 4) {
        const float* y0 = level_storage + i * level_width_dims;
        const float* y1 = level_storage + (i + 1) * level_width_dims;
        const float* y2 = level_storage + (i + 2) * level_width_dims;
        const float* y3 = level_storage + (i + 3) * level_width_dims;

        float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
        for (size_t j = 0; j < level_width_dims; j++) {
            float q = query_level[j];
            dp0 += q * y0[j];
            dp1 += q * y1[j];
            dp2 += q * y2[j];
            dp3 += q * y3[j];
        }

        dot_products[i] = dp0;
        dot_products[i + 1] = dp1;
        dot_products[i + 2] = dp2;
        dot_products[i + 3] = dp3;
    }
    for (; i < num_active; i++) {
        const float* yj = level_storage + i * level_width_dims;
        float dp = 0;
        for (size_t j = 0; j < level_width_dims; j++) {
            dp += query_level[j] * yj[j];
        }
        dot_products[i] = dp;
    }
}
IMPRECISE_FUNCTION_END

// V1: FixedWidth=32 constprop — batch-of-4
IMPRECISE_FUNCTION_BEGIN
template <size_t W>
__attribute__((noinline))
static void v1_constprop(
        const float* query_level,
        const float* level_storage,
        size_t num_active,
        float* dot_products) {
    size_t i = 0;
    for (; i + 4 <= num_active; i += 4) {
        const float* y0 = level_storage + i * W;
        const float* y1 = level_storage + (i + 1) * W;
        const float* y2 = level_storage + (i + 2) * W;
        const float* y3 = level_storage + (i + 3) * W;

        float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
        for (size_t j = 0; j < W; j++) {
            float q = query_level[j];
            dp0 += q * y0[j];
            dp1 += q * y1[j];
            dp2 += q * y2[j];
            dp3 += q * y3[j];
        }

        dot_products[i] = dp0;
        dot_products[i + 1] = dp1;
        dot_products[i + 2] = dp2;
        dot_products[i + 3] = dp3;
    }
    for (; i < num_active; i++) {
        const float* yj = level_storage + i * W;
        float dp = 0;
        for (size_t j = 0; j < W; j++)
            dp += query_level[j] * yj[j];
        dot_products[i] = dp;
    }
}
IMPRECISE_FUNCTION_END

// V2: Batch-of-8, runtime width
IMPRECISE_FUNCTION_BEGIN
__attribute__((noinline))
static void v2_batch8_runtime(
        const float* query_level,
        const float* level_storage,
        size_t num_active,
        size_t level_width_dims,
        float* dot_products) {
    size_t i = 0;
    for (; i + 8 <= num_active; i += 8) {
        const float* y[8];
        for (size_t k = 0; k < 8; k++)
            y[k] = level_storage + (i + k) * level_width_dims;

        float dp[8] = {};
        for (size_t j = 0; j < level_width_dims; j++) {
            float q = query_level[j];
            for (size_t k = 0; k < 8; k++)
                dp[k] += q * y[k][j];
        }
        for (size_t k = 0; k < 8; k++)
            dot_products[i + k] = dp[k];
    }
    for (; i < num_active; i++) {
        const float* yj = level_storage + i * level_width_dims;
        float dp = 0;
        for (size_t j = 0; j < level_width_dims; j++)
            dp += query_level[j] * yj[j];
        dot_products[i] = dp;
    }
}
IMPRECISE_FUNCTION_END

// V3: Batch-of-8 + FixedWidth=32
IMPRECISE_FUNCTION_BEGIN
template <size_t W>
__attribute__((noinline))
static void v3_batch8_constprop(
        const float* query_level,
        const float* level_storage,
        size_t num_active,
        float* dot_products) {
    size_t i = 0;
    for (; i + 8 <= num_active; i += 8) {
        const float* y[8];
        for (size_t k = 0; k < 8; k++)
            y[k] = level_storage + (i + k) * W;

        float dp[8] = {};
        for (size_t j = 0; j < W; j++) {
            float q = query_level[j];
            for (size_t k = 0; k < 8; k++)
                dp[k] += q * y[k][j];
        }
        for (size_t k = 0; k < 8; k++)
            dot_products[i + k] = dp[k];
    }
    for (; i < num_active; i++) {
        const float* yj = level_storage + i * W;
        float dp = 0;
        for (size_t j = 0; j < W; j++)
            dp += query_level[j] * yj[j];
        dot_products[i] = dp;
    }
}
IMPRECISE_FUNCTION_END

// V4: Batch-of-8 + FixedWidth=32 + __restrict__
IMPRECISE_FUNCTION_BEGIN
template <size_t W>
__attribute__((noinline))
static void v4_batch8_restrict(
        const float* __restrict__ query_level,
        const float* __restrict__ level_storage,
        size_t num_active,
        float* __restrict__ dot_products) {
    size_t i = 0;
    for (; i + 8 <= num_active; i += 8) {
        const float* y[8];
        for (size_t k = 0; k < 8; k++)
            y[k] = level_storage + (i + k) * W;

        float dp[8] = {};
        for (size_t j = 0; j < W; j++) {
            float q = query_level[j];
            for (size_t k = 0; k < 8; k++)
                dp[k] += q * y[k][j];
        }
        for (size_t k = 0; k < 8; k++)
            dot_products[i + k] = dp[k];
    }
    for (; i < num_active; i++) {
        const float* yj = level_storage + i * W;
        float dp = 0;
        for (size_t j = 0; j < W; j++)
            dp += query_level[j] * yj[j];
        dot_products[i] = dp;
    }
}
IMPRECISE_FUNCTION_END

// V5: Batch-of-4 + FixedWidth=32 + __restrict__
IMPRECISE_FUNCTION_BEGIN
template <size_t W>
__attribute__((noinline))
static void v5_batch4_restrict(
        const float* __restrict__ query_level,
        const float* __restrict__ level_storage,
        size_t num_active,
        float* __restrict__ dot_products) {
    size_t i = 0;
    for (; i + 4 <= num_active; i += 4) {
        const float* y0 = level_storage + (i + 0) * W;
        const float* y1 = level_storage + (i + 1) * W;
        const float* y2 = level_storage + (i + 2) * W;
        const float* y3 = level_storage + (i + 3) * W;

        float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
        for (size_t j = 0; j < W; j++) {
            float q = query_level[j];
            dp0 += q * y0[j];
            dp1 += q * y1[j];
            dp2 += q * y2[j];
            dp3 += q * y3[j];
        }

        dot_products[i + 0] = dp0;
        dot_products[i + 1] = dp1;
        dot_products[i + 2] = dp2;
        dot_products[i + 3] = dp3;
    }
    for (; i < num_active; i++) {
        const float* yj = level_storage + i * W;
        float dp = 0;
        for (size_t j = 0; j < W; j++)
            dp += query_level[j] * yj[j];
        dot_products[i] = dp;
    }
}
IMPRECISE_FUNCTION_END

// V6: Batch-of-12 + FixedWidth=32 + __restrict__
IMPRECISE_FUNCTION_BEGIN
template <size_t W>
__attribute__((noinline))
static void v6_batch12_restrict(
        const float* __restrict__ query_level,
        const float* __restrict__ level_storage,
        size_t num_active,
        float* __restrict__ dot_products) {
    size_t i = 0;
    for (; i + 12 <= num_active; i += 12) {
        float dp[12] = {};
        for (size_t j = 0; j < W; j++) {
            float q = query_level[j];
            for (size_t k = 0; k < 12; k++)
                dp[k] += q * level_storage[(i + k) * W + j];
        }
        for (size_t k = 0; k < 12; k++)
            dot_products[i + k] = dp[k];
    }
    for (; i < num_active; i++) {
        const float* yj = level_storage + i * W;
        float dp = 0;
        for (size_t j = 0; j < W; j++)
            dp += query_level[j] * yj[j];
        dot_products[i] = dp;
    }
}
IMPRECISE_FUNCTION_END

// V7: Batch-of-16 + FixedWidth=32 + __restrict__
IMPRECISE_FUNCTION_BEGIN
template <size_t W>
__attribute__((noinline))
static void v7_batch16_restrict(
        const float* __restrict__ query_level,
        const float* __restrict__ level_storage,
        size_t num_active,
        float* __restrict__ dot_products) {
    size_t i = 0;
    for (; i + 16 <= num_active; i += 16) {
        float dp[16] = {};
        for (size_t j = 0; j < W; j++) {
            float q = query_level[j];
            for (size_t k = 0; k < 16; k++)
                dp[k] += q * level_storage[(i + k) * W + j];
        }
        for (size_t k = 0; k < 16; k++)
            dot_products[i + k] = dp[k];
    }
    for (; i < num_active; i++) {
        const float* yj = level_storage + i * W;
        float dp = 0;
        for (size_t j = 0; j < W; j++)
            dp += query_level[j] * yj[j];
        dot_products[i] = dp;
    }
}
IMPRECISE_FUNCTION_END

// V8: Batch-of-4 + FixedWidth=32 + explicit pointer style (named vars, not array)
IMPRECISE_FUNCTION_BEGIN
template <size_t W>
__attribute__((noinline))
static void v8_batch4_named(
        const float* __restrict__ query_level,
        const float* __restrict__ level_storage,
        size_t num_active,
        float* __restrict__ dot_products) {
    size_t i = 0;
    for (; i + 4 <= num_active; i += 4) {
        const float* __restrict__ y0 = level_storage + (i + 0) * W;
        const float* __restrict__ y1 = level_storage + (i + 1) * W;
        const float* __restrict__ y2 = level_storage + (i + 2) * W;
        const float* __restrict__ y3 = level_storage + (i + 3) * W;

        float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
        for (size_t j = 0; j < W; j++) {
            float q = query_level[j];
            dp0 += q * y0[j];
            dp1 += q * y1[j];
            dp2 += q * y2[j];
            dp3 += q * y3[j];
        }

        dot_products[i + 0] = dp0;
        dot_products[i + 1] = dp1;
        dot_products[i + 2] = dp2;
        dot_products[i + 3] = dp3;
    }
    for (; i < num_active; i++) {
        const float* yj = level_storage + i * W;
        float dp = 0;
        for (size_t j = 0; j < W; j++)
            dp += query_level[j] * yj[j];
        dot_products[i] = dp;
    }
}
IMPRECISE_FUNCTION_END

// V9: Batch-of-6 + FixedWidth=32 (1024 is divisible by 2, not 6, tests remainder)
IMPRECISE_FUNCTION_BEGIN
template <size_t W>
__attribute__((noinline))
static void v9_batch6_restrict(
        const float* __restrict__ query_level,
        const float* __restrict__ level_storage,
        size_t num_active,
        float* __restrict__ dot_products) {
    size_t i = 0;
    for (; i + 6 <= num_active; i += 6) {
        float dp[6] = {};
        for (size_t j = 0; j < W; j++) {
            float q = query_level[j];
            for (size_t k = 0; k < 6; k++)
                dp[k] += q * level_storage[(i + k) * W + j];
        }
        for (size_t k = 0; k < 6; k++)
            dot_products[i + k] = dp[k];
    }
    for (; i < num_active; i++) {
        const float* yj = level_storage + i * W;
        float dp = 0;
        for (size_t j = 0; j < W; j++)
            dp += query_level[j] * yj[j];
        dot_products[i] = dp;
    }
}
IMPRECISE_FUNCTION_END

// V10: Batch-of-2 + FixedWidth=32 (minimal ILP)
IMPRECISE_FUNCTION_BEGIN
template <size_t W>
__attribute__((noinline))
static void v10_batch2_restrict(
        const float* __restrict__ query_level,
        const float* __restrict__ level_storage,
        size_t num_active,
        float* __restrict__ dot_products) {
    size_t i = 0;
    for (; i + 2 <= num_active; i += 2) {
        const float* y0 = level_storage + (i + 0) * W;
        const float* y1 = level_storage + (i + 1) * W;
        float dp0 = 0, dp1 = 0;
        for (size_t j = 0; j < W; j++) {
            float q = query_level[j];
            dp0 += q * y0[j];
            dp1 += q * y1[j];
        }
        dot_products[i + 0] = dp0;
        dot_products[i + 1] = dp1;
    }
    for (; i < num_active; i++) {
        const float* yj = level_storage + i * W;
        float dp = 0;
        for (size_t j = 0; j < W; j++)
            dp += query_level[j] * yj[j];
        dot_products[i] = dp;
    }
}
IMPRECISE_FUNCTION_END

// V11: Batch-of-3 + FixedWidth=32 + __restrict__
IMPRECISE_FUNCTION_BEGIN
template <size_t W>
__attribute__((noinline))
static void v11_batch3_restrict(
        const float* __restrict__ query_level,
        const float* __restrict__ level_storage,
        size_t num_active,
        float* __restrict__ dot_products) {
    size_t i = 0;
    for (; i + 3 <= num_active; i += 3) {
        const float* y0 = level_storage + (i + 0) * W;
        const float* y1 = level_storage + (i + 1) * W;
        const float* y2 = level_storage + (i + 2) * W;
        float dp0 = 0, dp1 = 0, dp2 = 0;
        for (size_t j = 0; j < W; j++) {
            float q = query_level[j];
            dp0 += q * y0[j];
            dp1 += q * y1[j];
            dp2 += q * y2[j];
        }
        dot_products[i + 0] = dp0;
        dot_products[i + 1] = dp1;
        dot_products[i + 2] = dp2;
    }
    for (; i < num_active; i++) {
        const float* yj = level_storage + i * W;
        float dp = 0;
        for (size_t j = 0; j < W; j++)
            dp += query_level[j] * yj[j];
        dot_products[i] = dp;
    }
}
IMPRECISE_FUNCTION_END

// V12: Direct=false indirect path, batch-of-4, constprop+restrict
IMPRECISE_FUNCTION_BEGIN
template <size_t W>
__attribute__((noinline))
static void v12_indirect_batch4(
        const float* __restrict__ query_level,
        const float* __restrict__ level_storage,
        const uint32_t* active_indices,
        size_t num_active,
        float* __restrict__ dot_products) {
    size_t i = 0;
    for (; i + 4 <= num_active; i += 4) {
        const float* y0 = level_storage + active_indices[i + 0] * W;
        const float* y1 = level_storage + active_indices[i + 1] * W;
        const float* y2 = level_storage + active_indices[i + 2] * W;
        const float* y3 = level_storage + active_indices[i + 3] * W;

        float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
        for (size_t j = 0; j < W; j++) {
            float q = query_level[j];
            dp0 += q * y0[j];
            dp1 += q * y1[j];
            dp2 += q * y2[j];
            dp3 += q * y3[j];
        }

        dot_products[i + 0] = dp0;
        dot_products[i + 1] = dp1;
        dot_products[i + 2] = dp2;
        dot_products[i + 3] = dp3;
    }
    for (; i < num_active; i++) {
        const float* yj = level_storage + active_indices[i] * W;
        float dp = 0;
        for (size_t j = 0; j < W; j++)
            dp += query_level[j] * yj[j];
        dot_products[i] = dp;
    }
}
IMPRECISE_FUNCTION_END

// V13: Direct=false indirect path, batch-of-2, constprop+restrict
IMPRECISE_FUNCTION_BEGIN
template <size_t W>
__attribute__((noinline))
static void v13_indirect_batch2(
        const float* __restrict__ query_level,
        const float* __restrict__ level_storage,
        const uint32_t* active_indices,
        size_t num_active,
        float* __restrict__ dot_products) {
    size_t i = 0;
    for (; i + 2 <= num_active; i += 2) {
        const float* y0 = level_storage + active_indices[i + 0] * W;
        const float* y1 = level_storage + active_indices[i + 1] * W;
        float dp0 = 0, dp1 = 0;
        for (size_t j = 0; j < W; j++) {
            float q = query_level[j];
            dp0 += q * y0[j];
            dp1 += q * y1[j];
        }
        dot_products[i + 0] = dp0;
        dot_products[i + 1] = dp1;
    }
    for (; i < num_active; i++) {
        const float* yj = level_storage + active_indices[i] * W;
        float dp = 0;
        for (size_t j = 0; j < W; j++)
            dp += query_level[j] * yj[j];
        dot_products[i] = dp;
    }
}
IMPRECISE_FUNCTION_END

// ===================================================================
// Benchmark harness
// ===================================================================

struct BenchResult {
    const char* name;
    double min_us;
    double med_us;
    double cycles_per_vec;
    size_t num_active;
};

static std::vector<BenchResult> results;

using KernelFn = std::function<void()>;

static void bench(const char* name, size_t num_active, KernelFn fn) {
    for (int i = 0; i < WARMUP_ITERS; i++) fn();

    std::vector<double> times_us;
    std::vector<uint64_t> cycles;
    times_us.reserve(BENCH_ITERS);
    cycles.reserve(BENCH_ITERS);

    for (int i = 0; i < BENCH_ITERS; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t c0 = rdtsc();
        fn();
        uint64_t c1 = rdtsc();
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        times_us.push_back(us);
        cycles.push_back(c1 - c0);
    }

    std::sort(times_us.begin(), times_us.end());
    std::sort(cycles.begin(), cycles.end());

    // Use p5 (5th percentile) to get stable minimum without outlier sensitivity
    size_t p5 = BENCH_ITERS / 20;
    double min_us = times_us[p5];
    double med_us = times_us[BENCH_ITERS / 2];
    double min_cyc = (double)cycles[p5];
    double cyc_per_vec = min_cyc / (double)num_active;

    printf("%-45s  n=%4zu  p5=%7.2f us  med=%7.2f us  %.1f cyc/vec\n",
           name, num_active, min_us, med_us, cyc_per_vec);

    results.push_back({name, min_us, med_us, cyc_per_vec, num_active});
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> query(DIM);
    for (auto& v : query) v = dist(rng);

    std::vector<float> level_storage(BATCH_SIZE * LEVEL_WIDTH);
    for (auto& v : level_storage) v = dist(rng);

    std::vector<float> dot_products(BATCH_SIZE);

    // Prevent the compiler from seeing LEVEL_WIDTH as constant for runtime variants
    volatile size_t runtime_width = LEVEL_WIDTH;
    size_t rw = runtime_width;

    printf("=== Dot Product Kernel Bench ===\n");
    printf("DIM=%zu  N_LEVELS=%zu  LEVEL_WIDTH=%zu  BATCH=%zu\n",
           DIM, N_LEVELS, LEVEL_WIDTH, BATCH_SIZE);
    printf("WARMUP=%d  BENCH=%d  (p5 timing)\n\n", WARMUP_ITERS, BENCH_ITERS);

    const size_t N = BATCH_SIZE;
    const float* Q = query.data();
    const float* S = level_storage.data();
    float* D = dot_products.data();

    bench("v0  baseline(batch4,runtime)", N, [&]() {
        v0_baseline(Q, S, N, rw, D); do_not_optimize(D);
    });
    bench("v1  constprop<32>(batch4)", N, [&]() {
        v1_constprop<32>(Q, S, N, D); do_not_optimize(D);
    });
    bench("v2  batch8(runtime)", N, [&]() {
        v2_batch8_runtime(Q, S, N, rw, D); do_not_optimize(D);
    });
    bench("v3  batch8+constprop<32>", N, [&]() {
        v3_batch8_constprop<32>(Q, S, N, D); do_not_optimize(D);
    });
    bench("v4  batch8+constprop<32>+restrict", N, [&]() {
        v4_batch8_restrict<32>(Q, S, N, D); do_not_optimize(D);
    });
    bench("v5  batch4+constprop<32>+restrict", N, [&]() {
        v5_batch4_restrict<32>(Q, S, N, D); do_not_optimize(D);
    });
    bench("v6  batch12+constprop<32>+restrict", N, [&]() {
        v6_batch12_restrict<32>(Q, S, N, D); do_not_optimize(D);
    });
    bench("v7  batch16+constprop<32>+restrict", N, [&]() {
        v7_batch16_restrict<32>(Q, S, N, D); do_not_optimize(D);
    });
    bench("v8  batch4+constprop<32>+named+restrict", N, [&]() {
        v8_batch4_named<32>(Q, S, N, D); do_not_optimize(D);
    });
    bench("v9  batch6+constprop<32>+restrict", N, [&]() {
        v9_batch6_restrict<32>(Q, S, N, D); do_not_optimize(D);
    });
    bench("v10 batch2+constprop<32>+restrict", N, [&]() {
        v10_batch2_restrict<32>(Q, S, N, D); do_not_optimize(D);
    });
    bench("v11 batch3+constprop<32>+restrict", N, [&]() {
        v11_batch3_restrict<32>(Q, S, N, D); do_not_optimize(D);
    });

    // Indirect path (Direct=false): simulate pruned levels with ~50% active
    {
        size_t half = N / 2;
        std::vector<uint32_t> indices(half);
        for (size_t i = 0; i < half; i++) indices[i] = i * 2;  // every other vector
        const uint32_t* idx = indices.data();

        bench("v12 indirect_batch4<32> (n=512)", half, [&]() {
            v12_indirect_batch4<32>(Q, S, idx, half, D); do_not_optimize(D);
        });
        bench("v13 indirect_batch2<32> (n=512)", half, [&]() {
            v13_indirect_batch2<32>(Q, S, idx, half, D); do_not_optimize(D);
        });
    }

    // Also test smaller active sets (simulating later levels after pruning)
    for (size_t active : {256uz, 128uz, 64uz}) {
        char name[80];
        snprintf(name, sizeof(name), "v5  batch4+cp<32>+restr (n=%zu)", active);
        bench(name, active, [&, active]() {
            v5_batch4_restrict<32>(Q, S, active, D); do_not_optimize(D);
        });
    }

    printf("\n=== Summary (sorted by p5_us) ===\n");
    std::sort(results.begin(), results.end(),
              [](const BenchResult& a, const BenchResult& b) {
                  return a.min_us < b.min_us;
              });
    for (auto& r : results) {
        printf("  %-45s  %7.2f us  %.1f cyc/vec\n",
               r.name, r.min_us, r.cycles_per_vec);
    }

    return 0;
}
