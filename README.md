# GPU Cosine Similarity Search (CUDA C++)

A small project to study GPU memory behavior. It implements brute-force cosine-similarity search over 1M × 128-dim vectors in three CUDA versions and benchmarks them against a single-thread CPU baseline and cuBLAS.

Vectors are L2-normalized up front, so cosine similarity equals a dot product. Data is random (seeded), so the goal is performance analysis, not retrieval quality.

## Versions

| # | Kernel | Idea |
|---|--------|------|
| 1 | `sim_naive` | One thread per doc, docs stored row-major `[N][D]`. Adjacent threads read addresses 512 B apart, so warp reads are uncoalesced. |
| 2 | `sim_coalesced` | Docs stored transposed `[D][N]` so adjacent threads read adjacent addresses. Query is broadcast from `__constant__` memory. |
| 3 | `sim_tiled` | 64 queries at once as a shared-memory tiled matmul (`C[Q][N] = Q[Q][D] × Docs^T[D][N]`). |
| ref | cuBLAS `sgemm` | Library reference for the batched case. |

Top-k selection for a query uses Thrust `sort_by_key`.

## Results

Tesla T4 (Colab), N = 1,048,576, D = 128, Q = 64, theoretical peak memory bandwidth 320 GB/s. Every GPU version matches the CPU result (max abs error 8.94e-08).

| Version | Time (ms) | Effective BW (GB/s) | ms / query |
|---------|-----------|---------------------|------------|
| CPU, single thread (1 query) | 134.3 | n/a | 134.3 |
| 1. Naive (row-major) | 8.942 | 60.5 | 8.942 |
| 2. Coalesced + constant memory | 2.131 | 253.9 | 2.131 |
| 3. Tiled, 64 queries | 20.600 | 39.1 | 0.322 |
| cuBLAS SGEMM, 64 queries | 4.484 | 179.6 | 0.070 |

Other measurements:
- Host to device copy of the docs (512 MB): 118 ms (about 4.6 GB/s).
- Batched throughput: tiled kernel about 834 GFLOP/s, cuBLAS about 3.8 TFLOP/s.
- Speedup over the single-thread CPU baseline (query 0): v1 15×, v2 63×.

## What the numbers show

- **Memory access pattern dominated v1.** Moving from row-major to a coalesced layout raised effective bandwidth from 60 to 254 GB/s (about 80% of peak) with no change in arithmetic.
- **Batching raises arithmetic intensity.** With 64 queries, each doc is read once and reused 64 times. Per-query time dropped from 2.13 ms (v2) to 0.32 ms (v3) and 0.07 ms (cuBLAS).
- **The "effective BW" of the tiled kernel is not a DRAM limit.** For the matmul it is bounded by instruction and shared-memory throughput. Each thread computes one output, so it spends two shared-memory loads per FMA. cuBLAS is about 4.6× faster, consistent with register-level tiling (each thread computing a small block of outputs), which this kernel does not do.
- **Transfer cost matters.** Copying the docs took about 55× longer than the v2 kernel, so a GPU only pays off when the docs stay resident and many queries are served. I suspect pageable host memory limited the copy to about 4.6 GB/s; I have not tested pinned memory.

## Build and run

Requires an NVIDIA GPU and the CUDA toolkit. On Google Colab, select a T4 runtime.

```bash
nvcc -O3 -arch=sm_75 gpu_cosine_search.cu -o search -lcublas
./search
```

Change `sm_75` to match your GPU (`sm_80` for A100, `sm_89` for L4).

## Limitations

- Random vectors, so the top-5 results are not meaningful.
- The CPU baseline is single-threaded and unvectorized, so the CPU speedups are against a weak baseline.
- The tiled kernel is a straightforward shared-memory implementation, with no register tiling, vectorized loads or double buffering.
- Brute-force search only; no approximate indexing (FAISS-style IVF/HNSW).
- Timings come from one run on one GPU, averaged over 20 iterations after one warm-up.

## Possible next steps

- Register tiling to close the gap with cuBLAS.
- Pinned host memory and overlapping transfers with compute using streams.
- Profile with Nsight Compute to confirm the memory-bound and compute-bound explanations above.
- GPU top-k instead of a full sort.
