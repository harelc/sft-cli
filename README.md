# sft

A fast, interactive terminal browser for `.safetensors` files — with LoRA analysis and compactification.

<p align="center">
  <img src="https://vhs.charm.sh/vhs-6eQ3Cv0oexkfUshZ7PmO3b.gif" alt="sft demo">
</p>

## Why?

If you work with ML models, you've probably found yourself wondering "what's actually in this .safetensors file?" — the layer names, shapes, dtypes, sizes. Maybe you want to check if a model has the layers you expect, compare two checkpoints, or just explore an unfamiliar architecture.

`sft` lets you do that instantly from your terminal. No Python scripts, no notebooks, no waiting for tensors to load into memory. It reads only the file header, so even multi-gigabyte models open in milliseconds.

## Installation

This fork is not published to PyPI — install directly from the GitHub repo.

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv tool install git+https://github.com/harelc/sft-cli
```

This makes `sft` available globally as a command.

Or with pip:

```bash
pip install git+https://github.com/harelc/sft-cli
```

To install a specific branch or tag, append `@<ref>` to the URL (e.g. `…/sft-cli@main`).

For local development:

```bash
git clone https://github.com/harelc/sft-cli
cd sft-cli
uv sync
uv run sft --help
```

## Usage

```bash
sft model.safetensors                                              # interactive browser
sft diff a.safetensors b.safetensors                               # tensor-by-tensor comparison
sft merge a.safetensors b.safetensors -a 0.7 -b 0.3 -o out.safetensors   # weighted LoRA merge
```

Navigate with arrow keys, search with `/`, quit with `q`.

## Features

### Tensor Browser
- **Hierarchical browser** — Tensors grouped by namespace (e.g., `model.layers.0.attention`)
- **Instant startup** — Header-only parsing, works on multi-GB files
- **Search** — Filter tensors by name with `/`
- **Sort** — By name, size, or rank with `s`
- **Inspect** — View full tensor details with `Space`
- **Metadata** — See embedded file metadata with `m`
- **Filter** — Filter by dtype with `f`
- **Read-only** — Never touches your model files

### LoRA Analysis (`l`)
- **Auto-detection** — Finds all LoRA A/B pairs automatically
- **Per-pair stats** — Frobenius norms (||A||, ||B||), mean, min/max range for both tensors
- **Effective rank** — Stable rank computation via fast QR-accelerated SVD
- **SVD spectrum** — Visual histogram of singular values per pair (`Enter`)
- **Sortable** — Sort pairs by name, rank, effective rank, or norms (`s`)
- **Export** — Save full analysis to JSON (`e`)

### Compactify (`c` from LoRA screen)
- **Rank reduction** — Truncate LoRA A/B pairs to a lower rank via SVD, keeping only the most important singular values
- **Fixed rank** — Specify a target rank (e.g., `8`) to truncate all pairs uniformly
- **Auto mode** — Type `auto` to truncate each pair to its effective rank + 1, keeping nearly all energy while minimizing rank
- **Energy tracking** — Shows fraction of Frobenius energy retained per pair
- **Output** — Saves a new `.safetensors` file (e.g., `model_r8.safetensors` or `model_rauto.safetensors`)

### Kohya → PEFT Conversion (`k`)
- **Auto-detect** — Identifies Kohya-format LoRA modules (`lora_down`/`lora_up`/`alpha`)
- **Rename + scale** — Renames to PEFT convention (`lora_A`/`lora_B`) and bakes `alpha/rank` scaling into B, so downstream loaders can drop the `.alpha` tensor
- **Pass-through** — Non-LoRA tensors and already-PEFT modules are copied as-is
- **Output** — Saves `<name>_peft.safetensors` with a `converted_from: kohya_lora_to_peft` metadata marker

### Weighted LoRA Merge (`sft merge`)
- **Weighted sum** — `C_eff = α · A_eff + β · B_eff` per module, where `_eff = lora_B @ lora_A`
- **Rank-aware** — Stacks the two adapters' factors (merged rank = `r_A + r_B`), so no compression is required before adding; ranks don't need to match
- **Optional truncation** — `--target-rank N` runs SVD on the merged pair to compress back to rank `N` (uses the same compactify tooling)
- **Per-side modules** — Modules present in only one file are kept and scaled by that file's coefficient
- **PEFT-form only** — Run the `k` converter on Kohya files first; Kohya modules are reported and skipped
- **CLI** — `sft merge A.safetensors B.safetensors -a 0.7 -b 0.3 -o out.safetensors [--target-rank N]`

### Diff (`D` or `sft diff`)
- **Tensor-by-tensor compare** — Categorizes each key as `equal`, `close`, `differ`, `incompatible` (shape mismatch), `left_only`, or `right_only`
- **Numeric metrics** — For comparable tensors: max-abs-diff, mean-abs-diff, relative L2 (`||a−b||/||a||`), and cosine similarity
- **TUI filters** — In the browser, press `D` to open a diff; toggle views with `a` (all) / `d` (differ) / `e` (equal) / `m` (missing) / `i` (incompatible)
- **CLI mode** — `sft diff a b [--rtol 1e-5] [--atol 1e-8] [--show diff|all|missing|incompatible] [--limit 50]`

## Keybindings

### Main Browser

| Key | Action |
|-----|--------|
| `↑`/`↓` | Navigate |
| `←`/`→` | Collapse/expand tree |
| `Tab` | Switch panels |
| `/` | Search |
| `s` | Cycle sort mode |
| `Space` | Tensor details |
| `m` | File metadata |
| `f` | Filter by dtype |
| `l` | LoRA analysis |
| `k` | Kohya → PEFT convert |
| `D` | Diff against another file |
| `q` | Quit |

### LoRA Analysis

| Key | Action |
|-----|--------|
| `↑`/`↓` | Select pair |
| `Enter` | SVD spectrum |
| `s` | Cycle sort mode |
| `c` | Compactify |
| `e` | Export to JSON |
| `?` | Help |
| `Esc`/`l` | Close |

## How It Works

### Fast Header Parsing
`sft` reads only the safetensors file header (a JSON blob at the start of the file) to extract tensor names, shapes, dtypes, and byte offsets. No tensor data is loaded during browsing.

### QR-Accelerated SVD
For LoRA analysis, computing the SVD of B@A directly would require forming the full (out_features × in_features) matrix — potentially 4096×4096 or larger. Instead, `sft` QR-factors both thin matrices and computes SVD on the small (rank × rank) product, making it orders of magnitude faster.

### Compactification
Rank reduction works by computing the SVD of each LoRA pair's effective matrix B@A, truncating to the top-k singular values, and reconstructing new smaller A' and B' matrices with √σ split equally between them. This is the optimal rank-k approximation (Eckart–Young theorem).

### Diff Metrics
For each common-key, same-shape tensor pair the diff reports:
- `max_abs` — `max |a − b|`, the worst-case elementwise drift
- `mean_abs` — average elementwise drift
- `rel_L2` — `||a − b|| / ||a||`, magnitude of the difference relative to A
- `cos` — cosine similarity; values near 1.0 with non-zero `rel_L2` indicate a pure scale change

## License

MIT
