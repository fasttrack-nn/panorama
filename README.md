
# Panorama: Fast-Track Nearest Neighbors

> **Read the paper first:** **[Panorama Technical Report (PDF)](panorama_tr.pdf)** — the complete write-up covering the theory, algorithms, cost model, and full experimental results.

Panorama is a state-of-the-art **refinement** technique for Approximate Nearest-Neighbor Search (ANNS) on high-dimensional neural embeddings. In modern ANNS pipelines the *filtering* phase has been heavily optimized, so candidate **verification** (computing exact distances to the retrieved candidates) has become the dominant cost — and its share of query latency grows monotonically with dimensionality.

Panorama attacks this bottleneck directly. It evaluates each candidate's distance **incrementally** along a PCA basis, maintaining a tight Cauchy–Schwarz lower bound on the full-vector distance and **pruning** any candidate the moment its bound exceeds the running k-th nearest-neighbor distance. The method is deterministic, exact at the strict setting, and has been upstreamed into [Faiss](https://github.com/facebookresearch/faiss) across its major index families.

## Key features

- **Up to 28.9× end-to-end speedup** over Faiss baselines, with gains that grow with embedding dimensionality.
- **Exactness by default, throughput on demand.** A single pruning knob `ε ∈ (0, 1]` controls aggressiveness: `ε = 1` is provably exact (no true neighbor is ever pruned), while `ε < 1` trades a controlled, near-zero recall loss for higher QPS.
- **Drop-in for existing indexes.** Panorama leaves the filtering phase untouched and integrates with **IVFFlat, IVFPQ, HNSW, and FastScan-Refine** in Faiss. It also accelerates any reranking pipeline.
- **Native product-quantization support.** A variance-shaping transform reconciles PCA's energy concentration with PQ's uniform-bit-budget assumption — something prior refinement methods cannot do.
- **Provable cost model.** Expected verification cost scales as `O(N·d / α)`, where `α` is the dataset's empirical PCA spectral-decay rate; the prediction tracks measured pruning within a small constant across six benchmarks.
- **Robust out-of-distribution.** Speedups degrade gracefully under distribution shift (e.g. 25.1× → 12.1× across easy-to-hard workloads) while preserving recall, unlike probabilistic baselines.

## How it works

1. **Accretive refinement (§3).** The squared L2 distance is decomposed over `L` levels via a norm-preserving transform. Partial inner products plus a Cauchy–Schwarz tail bound give a lower bound `LBℓ` that tightens monotonically as more levels are read. A candidate is pruned as soon as its bound exceeds the current k-th best distance. The relaxed score `LBℓ_ε = Aℓ − 2εCℓ` exposes the `ε` knob (exact at `ε = 1`).
2. **PCA transform + variance shaping (§4).** PCA is the energy-optimal orthogonal basis: it leaves the smallest possible tail energy at every level, making the pruning bound as tight as possible. To stay compatible with product quantization, Panorama applies a two-stage **variance-shaping** step — a "Robin Hood" inter-level rotation that spills surplus energy from rich leading levels into poorer trailing ones, followed by an intra-level Haar rotation that isotropizes each block — preserving most of PCA's front-loading while flattening per-block variance for PQ.
3. **Cost model (§5).** Real embeddings exhibit exponential PCA tail decay `E(m) ≈ e^(−α·m/d)`. This yields the central scaling law `E[Cost] ≲ c·N·d / α`, an out-of-distribution extension using the arithmetic mean of query/database decay rates, and an additive `(d/α)·ln ε` shift for the `ε`-relaxation.
4. **Systems co-design (§6).** Panorama is co-designed with the memory hierarchy: a **level-major batched layout** (batch size `B`) for contiguous indexes (IVFFlat/IVFPQ), a **sub-quantizer-major layout** for PQ codes that keeps tiny LUT slices pinned in L1, SIMD bulk-pruning kernels, and a point-centric (`B = 1`) variant for scattered layouts (HNSW, Refine).

## Repository structure

```
panorama/
├── panorama_tr.pdf         # Technical report (full paper: theory, algorithms, results)
├── faiss/                  # Faiss with Panorama: IVFFlat, IVFPQ, HNSW, FastScan-Refine
├── PDX/                    # PDX (pdxearch) vector search library (baseline + layout reference)
├── ADSampling/             # ADSampling baseline (SIGMOD 2023)
├── hephaestus/             # Out-of-distribution query generation
├── benches/                # Benchmarking harness (bench.py) & experiment configs
├── data/
│   └── install_datasets.py # Dataset downloader (from Hugging Face)
├── setup.sh                # One-command build script
└── requirements.txt        # Python dependencies
```

## Prerequisites

| Platform | Requirements |
|----------|-------------|
| **Linux** (Ubuntu/Debian) | `build-essential cmake swig python3-dev python3-numpy libopenblas-dev liblapack-dev libomp-dev` |
| **macOS** | Xcode Command Line Tools, [Homebrew](https://brew.sh), formulae: `cmake swig libomp openblas` |

Python 3.8+ is required. The setup script installs system dependencies automatically.

## Quick start

### 1. Clone the repository

```bash
git clone --recursive https://github.com/fasttrack-nn/panorama.git
cd panorama
```

### 2. Build everything

`setup.sh` handles the full build: it creates a virtual environment, installs Python dependencies, and compiles Faiss and pdxearch from source.

Set `FAISS_OPT_LEVEL` to match your CPU architecture:

| Architecture | Valid opt levels |
|--------------|-----------------|
| x86_64 | `generic` `avx2` `avx512` `avx512_spr` |
| ARM64 (macOS) | `generic` (NEON is always enabled) |
| ARM64 (Linux) | `generic` `sve` (NEON is always enabled) |

```bash
FAISS_OPT_LEVEL=avx512 ./setup.sh
```

### 3. Activate the virtual environment

```bash
source venv/bin/activate
```

### 4. Download datasets

Datasets are hosted on Hugging Face ([PanoramaVLDB/data](https://huggingface.co/datasets/PanoramaVLDB/data/tree/main)) and downloaded as `.fvec` files (split files are reassembled automatically).

```bash
python3 data/install_datasets.py              # saves to data/datasets/
python3 data/install_datasets.py /some/path   # custom destination
```

### 5. Run benchmarks

```bash
cd benches
python3 bench.py --config example_config.json --note "my first run"
```

#### Benchmark CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--config <path>` | *(required)* | JSON or YAML experiment config file |
| `--note <text>` | *(required)* | Free-text note stored with every result row |
| `--db <path>` | `<repo>/data/benchmarks.duckdb` | DuckDB file for results |
| `--datasets-dir <dir>` | `<repo>/data/datasets` | Directory containing `.fvec`/`.fvecs` files |
| `--scale-factor <0-1>` | `1.0` | Fraction of the dataset to use |

Each run is tagged with a unique `run_id`, and results are appended to the DuckDB database.

## Experiment configuration

Benchmarks are driven by a JSON (or YAML) config. Experiments are grouped per dataset; each experiment names an `index_type` and sweeps over arrays of build/search parameters. See `benches/example_config.json` for a full example.

```json
{
  "nq": 100,
  "k": 10,
  "datasets": {
    "gist1m": {
      "experiments": [
        {
          "index_type": "ivf_flat_pano",
          "build_params": { "nlist": [128], "nlevels": [8, 16] },
          "search_params": { "nprobe": [1, 4, 8, 32], "epsilon": [1.0, 0.8] }
        }
      ]
    }
  }
}
```

Parameter values are arrays so a single run can sweep multiple settings.

### Supported index types

Panorama variants (suffix `_pano`) and their Faiss/PDX baselines:

| Index type | Build params | Search params | Description |
|------------|-------------|---------------|-------------|
| `naive` | — | — | Brute-force flat L2 |
| `naive_pano` | `nlevels` | `epsilon` | Brute-force + Panorama |
| `ivf_flat` | `nlist` | `nprobe` | IVF with flat quantizer |
| `ivf_flat_pano` | `nlist`, `nlevels`, `batch_size` | `nprobe`, `epsilon` | IVFFlat + Panorama |
| `ivfpq` | `nlist`, `M` | `nprobe` | IVF with product quantization |
| `ivfpq_pano` | `nlist`, `M`, `nlevels`, `alpha` | `nprobe`, `epsilon` | IVFPQ + Panorama (variance-shaped) |
| `hnsw` | `M_HNSW`, `efConstruction` | `efSearch` | Hierarchical NSW graph |
| `hnsw_pano` | `M_HNSW`, `efConstruction`, `nlevels` | `efSearch`, `epsilon` | HNSW + Panorama |
| `fastscan_refine` | `nlist`, `M`, `bbs` | `nprobe`, `k_factor` | PQ4 FastScan + exact refine |
| `fastscan_refine_pano` | `nlist`, `M`, `bbs`, `nlevels` | `nprobe`, `k_factor`, `epsilon` | FastScan + Panorama refine |
| `rabitq_refine` | `nlist`, `bbs`, `nb_bits` | `nprobe`, `k_factor` | RaBitQ FastScan + exact refine |
| `rabitq_refine_pano` | `nlist`, `bbs`, `nb_bits`, `nlevels` | `nprobe`, `k_factor`, `epsilon` | RaBitQ FastScan + Panorama refine |
| `pdx_ivf` | `nlist` | `nprobe` | PDX IVF baseline |
| `pdx_bond_ivf` | `nlist` | `nprobe` | PDX-BOND IVF baseline |
| `ads_ivf` / `ads_hnsw` | — | — | ADSampling baselines |
| `hnswlib` | `M_HNSW`, `efConstruction` | `efSearch` | hnswlib graph baseline |

Key Panorama parameters: `nlevels` is the number of refinement levels `L`; `batch_size` is the batched-layout size `B` (1024 is the sweet spot for contiguous indexes); `epsilon` is the runtime pruning knob `ε` (1.0 = exact); `alpha` controls the IVFPQ variance-shaping cap.

## Datasets

Dataset files follow the naming convention `<name>_base.fvec` and `<name>_query.fvec`; the name used in the config (e.g. `"gist1m"`) must match these prefixes. The paper evaluates six benchmarks spanning 96–3072 dimensions:

| Dataset | Semantics | Size | Dim. |
|---------|-----------|------|------|
| Deep | Image embeddings | 9.99M | 96 |
| SIFT | Image features | 10M | 128 |
| arXiv | Text embeddings | 2.24M | 768 |
| GIST | Image features | 1M | 960 |
| Ada | Text embeddings (OpenAI `text-embedding-ada-002`) | 900K | 1536 |
| Large | Text embeddings (OpenAI `text-embedding-3-large`) | 900K | 3072 |

## Citation

If you use Panorama, please cite:

```bibtex
@article{panorama2026,
  title   = {Panorama: Fast-Track Nearest Neighbors},
  author  = {Schlomer, Alexis and Nayar, Akash K. and Ramani, Vansh and
             Ranu, Sayan and Patel, Jignesh M. and Karras, Panagiotis},
  journal = {Proceedings of the VLDB Endowment},
  volume  = {14},
  number  = {1},
  year    = {2026}
}
```
