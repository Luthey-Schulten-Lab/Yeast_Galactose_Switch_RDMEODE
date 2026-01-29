# Migration Guide

This guide explains how to migrate code from `galactose_rdmeode_combined_ribo_move.py` to the modular structure.

## Quick Start

The modular structure is set up at:
```
rdme_ode/galactose_model/
```

Main entry point:
```bash
python galactose_model/main.py -id 1 -t 60 -g 11.1 --enable-er --enable-ribosome-movement
```

## File Mapping

### Original → Modular Structure

| Original Section | Line Range | New Location |
|-----------------|------------|--------------|
| Imports & Setup | 1-150 | `main.py` |
| Argument Parsing | 26-69 | `config/args_parser.py` |
| Output Path Setup | 91-132 | `config/args_parser.py::setup_output_paths()` |
| Memory Monitor | 144-147 | `utils/memory_monitor.py` |
| Lattice Loading | 148-155 | `geometry/lattice_loader.py` |
| Region Definitions | 156-600+ | `geometry/region_builder.py` |
| OdeRdmeHybridSolver | 1038-1431 | `solvers/hybrid_solver.py` |
| Ribosome Movement | 1170-1298 | `movement/ribosome_movement.py` |
| NumpyEncoder | 1028-1036 | `utils/json_encoder.py` |
| Signal Handler | 1433-1442 | `utils/signal_handler.py` |
| Main Execution | 1444-1483 | `main.py::main()` |

## Step-by-Step Migration

### 1. Config Module (COMPLETED ✓)
**Files**: `config/args_parser.py`

- ✓ `parse_arguments()` - All command-line arguments
- ✓ `setup_output_paths()` - Output directory and file naming

**Status**: Ready to use

### 2. Utils Module (COMPLETED ✓)
**Files**: `utils/memory_monitor.py`, `utils/json_encoder.py`, `utils/signal_handler.py`

- ✓ `print_memory_usage()` - Memory monitoring
- ✓ `NumpyEncoder` - JSON encoding for numpy arrays
- ✓ `setup_signal_handler()` - Graceful shutdown

**Status**: Ready to use

### 3. Geometry Module (PARTIAL ⚠️)
**Files**: `geometry/lattice_loader.py`, `geometry/region_builder.py`

**Completed**:
- ✓ `load_lattice_data()` - Load pickle/lzma files
- ✓ `get_bool_lattice()` - Boolean region extraction

**TODO**:
- [ ] Extract region building logic from lines 156-1000
- [ ] Implement `build_regions()` to create RDMESim object
- [ ] Add chromosome region loading
- [ ] Add ER region loading
- [ ] Add ribosome region definitions

**How to migrate**:
1. Open original file, find region building code (lines ~156-1000)
2. Copy region definitions to `geometry/region_builder.py`
3. Organize into functions by feature (basic, chromosome, ER, etc.)

### 4. Movement Module (COMPLETED ✓)
**Files**: `movement/ribosome_movement.py`

**Completed**:
- ✓ `RibosomeMovementManager` class
- ✓ `move_ribosomes_hook()` - Random diffusion with site swapping
- ✓ Region cache building
- ✓ Particle and site type swapping

**Status**: Ready to use - contains your updated ribosome movement code

### 5. Solvers Module (TODO 📝)
**Files**: `solvers/hybrid_solver.py`

**TODO**:
- [ ] Extract `OdeRdmeHybridSolver` class (lines 1038-1431)
- [ ] Split into multiple files:
  - `hybrid_solver.py` - Main class structure
  - `ode_methods.py` - ODE integration methods
  - `rdme_methods.py` - RDME-specific methods
  - `hook_methods.py` - Simulation hooks

**How to migrate**:
1. Copy class definition from original (line 1038)
2. Copy `__init__()` method
3. Copy all methods, organizing by category
4. Update imports in each file

### 6. Main Entry Point (COMPLETED ✓)
**File**: `main.py`

**Completed**:
- ✓ Argument parsing
- ✓ Geometry loading
- ✓ Solver creation
- ✓ Simulation execution
- ✓ Error handling
- ✓ Output summary

**Status**: Ready to use (once dependencies are migrated)

## Testing Strategy

### Phase 1: Unit Testing
Test each module independently:

```python
# Test config
from galactose_model.config import parse_arguments
args = parse_arguments()

# Test geometry loader
from galactose_model.geometry import load_lattice_data
data = load_lattice_data("yeast-lattice.2.pkl.xz")

# Test utils
from galactose_model.utils import print_memory_usage
print_memory_usage()
```

### Phase 2: Integration Testing
Run with simple configuration:

```bash
python galactose_model/main.py -id 99 -t 1 -g 11.1
```

### Phase 3: Full Testing
Run with all features enabled:

```bash
python galactose_model/main.py -id 99 -t 60 -g 11.1 \
    --enable-er --enable-chromosome \
    --enable-effective-ribosome \
    --enable-ribosome-movement \
    --ribosome-movement-mode hook
```

## Current Status Summary

### Ready to Use ✓
- Configuration module
- Utils module (memory, JSON, signals)
- Movement module (ribosome diffusion)
- Main entry point structure

### Needs Implementation 📝
- **Geometry module**: `build_regions()` function
- **Solvers module**: `OdeRdmeHybridSolver` class

### Priority Next Steps

1. **HIGH**: Migrate `OdeRdmeHybridSolver` class
   - This is the core of the simulation
   - Start with `__init__()` and `hookSimulation()`

2. **HIGH**: Implement `build_regions()`
   - Required for simulation to initialize
   - Start with basic regions, then add ER/chromosome

3. **MEDIUM**: Test integration
   - Once both above are done, test end-to-end

4. **LOW**: Add documentation
   - Docstrings for all functions
   - Usage examples

## Benefits Achieved

Even with partial migration:

1. **Ribosome Movement** - Now isolated and easy to modify
2. **Configuration** - All arguments in one place
3. **Utils** - Reusable across projects
4. **Structure** - Clear organization for future work

## Questions?

See `README.md` for overview and `main.py` for usage examples.
