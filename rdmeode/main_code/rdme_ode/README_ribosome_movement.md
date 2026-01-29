# Ribosome Movement Optimization

This document describes the optimized ribosome movement features in `galactose_rdmeode_combined_ribo_move.py`.

## Overview

The script supports moving 300k+ ribosomes efficiently using two different modes:

1. **Diffusion Mode** (Recommended for speed): Uses the solver's built-in diffusion mechanism
2. **Hook Mode** (For controlled movement): Moves ribosomes in the hook function with batching

## Command-Line Arguments

### Enable Ribosome Movement
```bash
--enable-ribosome-movement
```
Enable ribosome movement functionality.

### Movement Mode
```bash
--ribosome-movement-mode {diffusion,hook}
```
- `diffusion` (default): Fast, handled by solver - recommended for 300k+ ribosomes
- `hook`: Controlled movement in hook function - slower but more control

### Diffusion Mode Options
```bash
--ribosome-diffusion-rate FLOAT
```
Diffusion rate for ribosomes in m²/s (default: 1e-14)

### Hook Mode Options
```bash
--ribosome-move-fraction FLOAT
```
Fraction of ribosomes to move per hook call (0.0-1.0, default: 0.05 = 5%)

```bash
--ribosome-move-interval INT
```
Move ribosomes every N hook calls to reduce overhead (default: 5)

## Usage Examples

### Fast Diffusion Mode (Recommended)
```bash
python galactose_rdmeode_combined_ribo_move.py \
    --enable-ribosome-movement \
    --ribosome-movement-mode diffusion \
    --ribosome-diffusion-rate 1e-14 \
    -id 1 -t 60 -g 11.1
```

### Controlled Hook Mode
```bash
python galactose_rdmeode_combined_ribo_move.py \
    --enable-ribosome-movement \
    --ribosome-movement-mode hook \
    --ribosome-move-fraction 0.05 \
    --ribosome-move-interval 5 \
    -id 1 -t 60 -g 11.1
```

### With ER Support
```bash
python galactose_rdmeode_combined_ribo_move.py \
    --enable-ribosome-movement \
    --ribosome-movement-mode diffusion \
    --enable-er \
    -id 1 -t 60 -g 11.1
```

## Performance Optimizations

### For 300k Ribosomes:

1. **Diffusion Mode**:
   - Handled entirely by the solver (GPU-accelerated)
   - No Python overhead
   - Fastest option
   - Movement constrained to ribosome regions only

2. **Hook Mode**:
   - Cached valid positions (computed once)
   - Batched movement (only moves fraction each hook)
   - Interval-based execution (skips most hook calls)
   - Vectorized operations where possible

### Recommended Settings for 300k Ribosomes:

- **Diffusion mode**: `--ribosome-diffusion-rate 1e-14` to `1e-13`
- **Hook mode**: 
  - `--ribosome-move-fraction 0.05` (5% per movement)
  - `--ribosome-move-interval 10` (move every 10 seconds)

## How It Works

### Diffusion Mode
- Enables limited diffusion within ribosome regions
- Ribosomes can move between sites within the same region
- Movement is constrained - ribosomes cannot leave their region
- Handled by the RDME solver (very fast)

### Hook Mode
1. Caches all valid ribosome positions on first call
2. Every N hook calls (interval), selects a fraction of positions with ribosomes
3. **Swaps particles** between positions to effectively move ribosome regions
4. This preserves particle counts while moving regions to new locations
5. Returns 1 to signal lattice modification (for GPU sync)

**Important**: The hook mode performs particle swapping, not just movement. This means:
- Position A with ribosomes swaps with Position B (which may have other ribosomes or be empty)
- All ribosome particles at each position are swapped together
- This effectively "moves" the ribosome region by switching what's at different positions

## Constraints

- **Regions cannot be moved**: The ribosome regions themselves are fixed
- **Movement within regions only**: Ribosomes can only move to other sites within ribosome regions
- **No cross-region movement**: Ribosomes stay in their assigned region type

## Output

The script will print:
- Ribosome movement status at startup
- Movement statistics during hook mode execution
- Total hook time (includes movement overhead)

Example output:
```
Ribosome movement: ENABLED (mode: diffusion)
  Diffusion rate: 1e-14 m^2/s
```

Or for hook mode:
```
Ribosome movement: ENABLED (mode: hook)
  Move fraction: 0.05, Interval: every 5 hooks
...
Moved 15000 ribosomes in 0.234s (hook 5)
```

## Notes

- The hook mode implementation may need adjustment based on the LM API for removing particles from old positions
- For very large simulations (300k+), diffusion mode is strongly recommended
- Hook mode is useful when you need explicit control over movement patterns

