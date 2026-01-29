# Galactose Model - Modular Structure

This directory contains the modularized version of `galactose_rdmeode_combined_ribo_move.py`.

## Directory Structure

```
galactose_model/
├── __init__.py                    # Package initialization
├── README.md                      # This file
│
├── config/                        # Configuration and argument parsing
│   ├── __init__.py
│   └── args_parser.py            # Command-line arguments and output setup
│
├── geometry/                      # Lattice geometry and regions
│   ├── __init__.py
│   ├── lattice_loader.py         # Load lattice data from pickle files
│   └── region_builder.py         # Build spatial regions (ER, chromosomes, etc.)
│
├── solvers/                       # Main simulation solver
│   ├── __init__.py
│   ├── hybrid_solver.py          # OdeRdmeHybridSolver class
│   ├── ode_methods.py            # ODE integration methods
│   ├── rdme_methods.py           # RDME-specific methods
│   └── hook_methods.py           # Simulation hooks and callbacks
│
├── movement/                      # Ribosome movement logic
│   ├── __init__.py
│   ├── ribosome_movement.py      # Main ribosome movement manager
│   ├── diffusion_mode.py         # Diffusion-based movement
│   └── hook_mode.py              # Hook-based movement (region swapping)
│
└── utils/                         # Utility functions
    ├── __init__.py
    ├── memory_monitor.py         # Memory usage tracking
    ├── json_encoder.py           # Custom JSON encoder for numpy
    └── signal_handler.py         # Signal handling for graceful shutdown

## Usage

After modularization, you can use the package like:

```python
from galactose_model import config, geometry, solvers, movement

# Parse arguments
args = config.parse_arguments()

# Load geometry
lattice_data = geometry.load_lattice_data(args.geometry_file)
regions = geometry.build_regions(lattice_data, args)

# Create solver
solver = solvers.OdeRdmeHybridSolver(regions, args)

# Run simulation
solver.run()
```

## Migration Guide

To migrate existing code sections:

1. **Arguments & Configuration** → `config/args_parser.py`
2. **Geometry Loading** → `geometry/lattice_loader.py`
3. **Region Building** → `geometry/region_builder.py`
4. **OdeRdmeHybridSolver Class** → `solvers/hybrid_solver.py`
5. **Ribosome Movement** → `movement/ribosome_movement.py`
6. **Utility Functions** → `utils/`

## Benefits

- **Maintainability**: Each module has a clear responsibility
- **Testability**: Individual components can be tested in isolation
- **Reusability**: Components can be imported and reused
- **Clarity**: Easier to understand and modify specific sections
