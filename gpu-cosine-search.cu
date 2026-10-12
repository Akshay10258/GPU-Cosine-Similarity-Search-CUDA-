%%writefile gpu_cosine_search.cu
// GPU cosine-similarity search: naive -> coalesced -> tiled batched, vs CPU and cuBLAS.
// Vectors are L2-normalized up front, so cosine similarity == dot product.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <functional>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

constexpr int D = 128;          // embedding dimension
constexpr int Q = 64;           // number of queries in the batched version
constexpr int TILE = 32;        // tile size for the shared-memory kernel

__constant__ float c_query[D];  // single query lives in constant memory (version 2)

// ---------------------------------------------------------------------------
// Version 1: naive. One thread per doc, docs stored row-major [N][D].
// Adjacent threads read addresses D*4 = 512 bytes apart -> uncoalesced.
// ---------------------------------------------------------------------------
__global__ void sim_naive(const float* docs, const float* q, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.f;
    for (int d = 0; d < D; ++d) s += docs[(size_t)i * D + d] * q[d];
    out[i] = s;
}

// ---------------------------------------------------------------------------
// Version 2: coalesced. Docs stored transposed [D][N], so for a fixed d,
// adjacent threads read adjacent addresses. Query is broadcast from constant memory.
// ---------------------------------------------------------------------------
__global__ void sim_coalesced(const float* docsT, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.f;
#pragma unroll
    for (int d = 0; d < D; ++d) s += docsT[(size_t)d * n + i] * c_query[d];
    out[i] = s;
}

// ---------------------------------------------------------------------------
// Version 3: batched queries as a shared-memory tiled matmul.
// C[Q][N] = Qm[Q][D] x DT[D][N]
// ---------------------------------------------------------------------------
__global__ void sim_tiled(const float* Qm, const float* DT, float* C, int nq, int n, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;   // query index
    int col = blockIdx.x * TILE + threadIdx.x;   // doc index
    float acc = 0.f;
    for (int t = 0; t < K; t += TILE) {
        As[threadIdx.y][threadIdx.x] =
            (row < nq && t + threadIdx.x < K) ? Qm[(size_t)row * K + t + threadIdx.x] : 0.f;
        Bs[threadIdx.y][threadIdx.x] =
            (t + threadIdx.y < K && col < n) ? DT[(size_t)(t + threadIdx.y) * n + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < nq && col < n) C[(size_t)row * n + col] = acc;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static float time_gpu(const std::function<void()>& fn, int iters = 20) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    fn();                                   // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / iters;
}

static void normalize_rows(std::vector<float>& m, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        double norm = 0;
        for (int c = 0; c < cols; ++c) norm += (double)m[(size_t)r * cols + c] * m[(size_t)r * cols + c];
        float inv = 1.0f / (float)std::sqrt(norm);
        for (int c = 0; c < cols; ++c) m[(size_t)r * cols + c] *= inv;
    }
}

static float max_abs_diff(const std::vector<float>& a, const float* b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

int main() {
    const int N = 1 << 20;  // ~1M docs

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int memClockKHz = 0, busWidth = 0;
    cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, 0);
    double peakBW = 2.0 * memClockKHz * 1e3 * (busWidth / 8.0) / 1e9;  // GB/s
    printf("GPU: %s | theoretical peak mem BW: %.0f GB/s\n", prop.name, peakBW);
    printf("N=%d docs, D=%d, batched queries Q=%d\n\n", N, D, Q);

    // ---- Generate data on host ----
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> hDocs((size_t)N * D), hDocsT((size_t)N * D), hQ((size_t)Q * D);
    for (auto& x : hDocs) x = dist(rng);
    for (auto& x : hQ) x = dist(rng);
    normalize_rows(hDocs, N, D);
    normalize_rows(hQ, Q, D);
    for (int i = 0; i < N; ++i)
        for (int d = 0; d < D; ++d) hDocsT[(size_t)d * N + i] = hDocs[(size_t)i * D + d];

    // ---- CPU baseline (single thread, query 0) ----
    std::vector<float> ref(N);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        float s = 0.f;
        for (int d = 0; d < D; ++d) s += hDocs[(size_t)i * D + d] * hQ[d];
        ref[i] = s;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- Device buffers ----
    float *dDocs, *dDocsT, *dQ, *dOut, *dC;
    CUDA_CHECK(cudaMalloc(&dDocs, (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDocsT, (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQ, (size_t)Q * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)Q * N * sizeof(float)));

    // Host -> device transfer time (docs only, ~512 MB)
    cudaEvent_t ta, tb;
    cudaEventCreate(&ta);
    cudaEventCreate(&tb);
    cudaEventRecord(ta);
    CUDA_CHECK(cudaMemcpy(dDocs, hDocs.data(), (size_t)N * D * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(tb);
    cudaEventSynchronize(tb);
    float h2d_ms = 0.f;
    cudaEventElapsedTime(&h2d_ms, ta, tb);
    CUDA_CHECK(cudaMemcpy(dDocsT, hDocsT.data(), (size_t)N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), (size_t)Q * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_query, hQ.data(), D * sizeof(float)));  // query 0

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    std::vector<float> out(N);

    // ---- Version 1: naive ----
    float ms_naive = time_gpu([&] { sim_naive<<<blocks, threads>>>(dDocs, dQ, dOut, N); });
    CUDA_CHECK(cudaMemcpy(out.data(), dOut, N * sizeof(float), cudaMemcpyDeviceToHost));
    float err_naive = max_abs_diff(ref, out.data(), N);

    // ---- Version 2: coalesced + constant memory ----
    float ms_coal = time_gpu([&] { sim_coalesced<<<blocks, threads>>>(dDocsT, dOut, N); });
    CUDA_CHECK(cudaMemcpy(out.data(), dOut, N * sizeof(float), cudaMemcpyDeviceToHost));
    float err_coal = max_abs_diff(ref, out.data(), N);

    // ---- Version 3: tiled, Q queries at once ----
    dim3 tblock(TILE, TILE);
    dim3 tgrid((N + TILE - 1) / TILE, (Q + TILE - 1) / TILE);
    float ms_tiled = time_gpu([&] { sim_tiled<<<tgrid, tblock>>>(dQ, dDocsT, dC, Q, N, D); });
    CUDA_CHECK(cudaMemcpy(out.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost));  // row 0
    float err_tiled = max_abs_diff(ref, out.data(), N);

    // ---- cuBLAS reference (row-major C = Qm * DT via column-major trick) ----
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.f, beta = 0.f;
    float ms_blas = time_gpu([&] {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, Q, D, &alpha, dDocsT, N, dQ, D, &beta, dC, N);
    });
    CUDA_CHECK(cudaMemcpy(out.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost));
    float err_blas = max_abs_diff(ref, out.data(), N);

    // ---- Report ----
    double gb_single = ((double)N * D * 4 + (double)N * 4) / 1e9;                    // docs read + scores written
    double gb_batch = ((double)N * D * 4 + (double)Q * N * 4 + (double)Q * D * 4) / 1e9;
    double gflop_batch = 2.0 * Q * (double)N * D / 1e9;

    printf("Host->device copy of docs: %.1f ms (%.1f GB/s)\n", h2d_ms, (N * D * 4.0 / 1e9) / (h2d_ms / 1e3));
    printf("CPU single-thread (1 query): %.1f ms\n\n", cpu_ms);

    printf("%-26s %10s %14s %12s %10s\n", "version", "time(ms)", "eff BW(GB/s)", "ms/query", "max err");
    printf("%-26s %10.3f %14.1f %12.3f %10.2e\n", "1 naive (row-major)", ms_naive, gb_single / (ms_naive / 1e3), ms_naive, err_naive);
    printf("%-26s %10.3f %14.1f %12.3f %10.2e\n", "2 coalesced + const mem", ms_coal, gb_single / (ms_coal / 1e3), ms_coal, err_coal);
    printf("%-26s %10.3f %14.1f %12.3f %10.2e\n", "3 tiled, 64 queries", ms_tiled, gb_batch / (ms_tiled / 1e3), ms_tiled / Q, err_tiled);
    printf("%-26s %10.3f %14.1f %12.3f %10.2e\n", "cuBLAS sgemm, 64 queries", ms_blas, gb_batch / (ms_blas / 1e3), ms_blas / Q, err_blas);
    printf("\nBatched GFLOP/s: tiled %.0f | cuBLAS %.0f\n", gflop_batch / (ms_tiled / 1e3), gflop_batch / (ms_blas / 1e3));
    printf("Speedup vs CPU (query 0): v1 %.1fx | v2 %.1fx\n", cpu_ms / ms_naive, cpu_ms / ms_coal);

    // ---- Top-5 for query 0 using Thrust ----
    sim_coalesced<<<blocks, threads>>>(dDocsT, dOut, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::device_ptr<float> p(dOut);
    thrust::device_vector<float> keys(p, p + N);
    thrust::device_vector<int> idx(N);
    thrust::sequence(idx.begin(), idx.end());
    thrust::sort_by_key(keys.begin(), keys.end(), idx.begin(), thrust::greater<float>());
    printf("\nTop-5 docs for query 0:\n");
    for (int i = 0; i < 5; ++i) {
        float s = keys[i];
        int id = idx[i];
        printf("  doc %d  score %.4f\n", id, s);
    }

    cublasDestroy(handle);
    cudaFree(dDocs); cudaFree(dDocsT); cudaFree(dQ); cudaFree(dOut); cudaFree(dC);
    return 0;
}

