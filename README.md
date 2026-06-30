
# PANORAMA: Fast-Track Nearest Neighbors

> **Read the paper first:** **[PANORAMA Technical Report (PDF)](panorama_tr.pdf)** — the complete write-up covering the theory, algorithms, complexity analysis, and full experimental results.

This repository contains the implementation of PANORAMA, a machine learning-driven approach for accelerating Approximate Nearest-Neighbor Search (ANNS) through data-adaptive learned orthogonal transforms and accretive distance bound refinement.

## Overview

PANORAMA addresses the verification bottleneck in modern ANNS systems, where up to 99% of query time is spent computing distances during the final refinement phase. Our approach uses learned orthogonal transforms that compact over 90% of signal energy into the first half of dimensions, enabling early candidate pruning with partial distance computations.

Key features:
- **2-30x end-to-end speedup** with no recall loss across diverse datasets
- **Universal integration** with existing ANNS methods (IVFPQ, HNSW, MRPT, IVFFlat)
- **Data-adaptive transforms** via PCA and IVFPQ-specific learned projections
- **Theoretical guarantees** with complexity analysis and robustness to out-of-distribution queries

## Repository Structure

```
panorama/
├── panorama_tr.pdf         # Technical report (full paper: theory, algorithms, results)
├── faiss/                  # Modified IVFPQ, IVFFlat, L2Flat, HNSW implementations
├── PDX/                    # PDX (pdxearch) vector search library
├── ADSampling/             # ADSampling baseline (SIGMOD 2023)
├── hephaestus/             # Out-of-distribution query generation
├── benches/                # Benchmarking infrastructure & configs
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

## Quick Start

### 1. Clone the repository

```bash
git clone --recursive <repo-url>
cd panorama
```

### 2. Build everything

The `setup.sh` script handles the full build: creates a virtual environment, installs Python dependencies, and compiles faiss and pdxearch from source.

You must set `FAISS_OPT_LEVEL` to match your CPU architecture:

| Architecture | Valid opt levels |
|--------------|-----------------|
| x86_64 | `generic` `avx2` `avx512` `avx512_spr` |
| ARM64 (macOS) | `generic` (NEON is always enabled) |
| ARM64 (Linux) | `generic` `sve` (NEON is always enabled) |

```bash
FAISS_OPT_LEVEL=generic ./setup.sh
```

### 3. Activate the virtual environment

```bash
source venv/bin/activate
```

### 4. Download datasets

Datasets are hosted on Hugging Face ([PanoramaVLDB/data](https://huggingface.co/datasets/PanoramaVLDB/data/tree/main)) and downloaded as `.fvec` files.

```bash
python3 data/install_datasets.py              # saves to data/datasets/
python3 data/install_datasets.py /some/path   # custom destination
```

### 5. Run benchmarks

```bash
cd benches
python3 bench.py --config example_config.json
```

#### Benchmark CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--config <path>` | *(required)* | JSON experiment config file |
| `--datasets-dir <dir>` | `<repo>/data/datasets` | Directory containing `.fvec`/`.fvecs` files |
| `--db <path>` | `data/benchmarks.duckdb` | DuckDB file for results |
| `--scale-factor <0-1>` | `1.0` | Fraction of the dataset to use |

Results are appended to the DuckDB database specified by `--db`.

## Experiment Configuration

Benchmarks are driven by a JSON config file. See `benches/example_config.json` for a full example.

```json
{
  "datasets": ["gist1m"],
  "nq": 100,
  "k": 10,
  "experiments": [
    {
      "index_type": "hnsw_pano",
      "build_params": { "M_HNSW": [16], "efConstruction": [40], "nlevels": [8] },
      "search_params": { "efSearch": [32, 64, 128, 1024] }
    }
  ]
}
```

### Supported index types

| Index type | Build params | Search params | Description |
|------------|-------------|---------------|-------------|
| `naive` | — | — | Brute-force flat L2 |
| `naive_pano` | `nlevels`, `batch_size` | — | Brute-force with Panorama transform |
| `ivf_flat` | `nlist` | `nprobe` | IVF with flat quantizer |
| `ivf_flat_pano` | `nlist`, `nlevels` | `nprobe` | IVF flat + Panorama |
| `ivfpq` | `nlist`, `M` | `nprobe` | IVF with product quantization |
| `ivfpq_pano` | `nlist`, `M`, `nlevels`, `alpha` | `nprobe` | IVFPQ + Panorama |
| `hnsw` | `M_HNSW`, `efConstruction` | `efSearch` | Hierarchical NSW graph |
| `hnsw_pano` | `M_HNSW`, `efConstruction`, `nlevels` | `efSearch` | HNSW + Panorama |
| `pdx_ivf` | `nlist` | `nprobe` | PDX vector search |

Parameter values are specified as arrays in the config to sweep over multiple settings in a single run.

## Datasets

Dataset files follow the naming convention `<name>_base.fvec` and `<name>_query.fvec`. The dataset name used in the config (e.g. `"gist1m"`) must match these file prefixes.
