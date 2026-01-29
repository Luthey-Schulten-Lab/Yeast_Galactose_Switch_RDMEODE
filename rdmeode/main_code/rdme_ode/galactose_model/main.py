#!/usr/bin/env python
# coding: utf-8
"""Main entry point for the modular galactose RDME/ODE simulation

This is the refactored version of galactose_rdmeode_combined_ribo_move.py
organized into a modular structure for easier maintenance and modification.

Usage:
    python -m galactose_model.main [arguments]

    or directly:

    python galactose_model/main.py [arguments]
"""

import time
import signal
import sys
import os

# Start timing
start_time = time.time()

# Import modular components
from config.args_parser import parse_arguments, setup_output_paths
from geometry.lattice_loader import load_lattice_data, get_bool_lattice
from geometry.region_builder import build_regions
from solvers.hybrid_solver import OdeRdmeHybridSolver
from utils.memory_monitor import print_memory_usage
from utils.signal_handler import setup_signal_handler

# LM imports
from jLM.Solvers import makeSolver
from lm import MGPUMpdRdmeSolver, MpdRdmeSolver


def main():
    """Main simulation function"""

    # Parse command-line arguments
    print("=" * 80)
    print("Modular Galactose RDME/ODE Simulation")
    print("=" * 80)

    args = parse_arguments()

    # Setup output paths
    output_folder, output_dir = setup_output_paths(args)
    print(f"Output folder: {output_folder}")
    print(f"Output directory: {output_dir}")

    # Print configuration summary
    print("\n" + "=" * 80)
    print("Configuration Summary:")
    print("=" * 80)
    print(f"Simulation time: {args.simtime} min")
    print(f"External galactose: {args.galactose} mM")
    print(f"GPUs: {args.gpus}")
    print(f"Geometry file: {args.geometry}")
    print(f"Gene location: {args.gene_location}")
    print(f"Chromosome support: {args.enable_chromosome}")
    print(f"ER support: {args.enable_er}")
    if args.enable_er:
        print(f"  ER tunnels: {args.er_num}")
    print(f"Effective ribosome: {args.enable_effective_ribosome}")
    print(f"RNA tracking: {args.enable_rna_tracking}")

    if args.enable_ribosome_movement:
        print(f"Ribosome movement: ENABLED")
        print(f"  Mode: {args.ribosome_movement_mode}")
        if args.ribosome_movement_mode == 'diffusion':
            print(f"  Diffusion rate: {args.ribosome_diffusion_rate} m^2/s")
        else:
            print(f"  Move fraction: {args.ribosome_move_fraction}")
            print(f"  Move interval: every {args.ribosome_move_interval} hooks")
    else:
        print(f"Ribosome movement: DISABLED")

    print("=" * 80 + "\n")

    # Initial memory check
    print("Initial memory usage:")
    print_memory_usage()

    # Load lattice geometry
    print("\nLoading lattice geometry...")
    lattice_data = load_lattice_data(args.geometry, args.if_dgx)
    print(f"Lattice loaded: {lattice_data['lattice'].shape}")

    # Build regions
    print("\nBuilding spatial regions...")
    sim = build_regions( lattice_data,output_folder, args)

    # Check if simulation object was created
    if sim is None:
        print("\n" + "=" * 80)
        print("ERROR: Simulation object not created")
        print("=" * 80)
        print("\nThe region_builder.py module is a placeholder and needs implementation.")
        print("\nTo complete the modular setup, you need to:")
        print("1. Extract region building code from galactose_rdmeode_combined_ribo_move.py")
        print("   (approximately lines 156-1000)")
        print("2. Implement the build_regions() function in geometry/region_builder.py")
        print("3. Extract OdeRdmeHybridSolver class to solvers/hybrid_solver.py")
        print("   (lines 1038-1431)")
        print("\nSee MIGRATION_GUIDE.md for detailed instructions.")
        print("=" * 80)
        return 1

    print("Regions built successfully")

    # Memory check after geometry loading
    print("\nMemory usage after geometry loading:")
    print_memory_usage()

    # Setup signal handler for graceful shutdown
    solver_container = {'solver': None}
    setup_signal_handler(solver_container)

    # Create solver based on GPU configuration
    print(f"\nCreating solver (GPUs: {args.gpus})...")
    if args.gpus == 1:
        Solver = makeSolver(MpdRdmeSolver, OdeRdmeHybridSolver)
    else:
        Solver = makeSolver(MGPUMpdRdmeSolver, OdeRdmeHybridSolver)

    # Instantiate solver with configuration
    try:
        solver = Solver(sim, args.galactose, output_folder=output_folder, args=args)
        solver_container['solver'] = solver
    except Exception as e:
        print(f"\nERROR creating solver: {e}")
        print("The OdeRdmeHybridSolver class may need additional fixes.")
        print("See solvers/hybrid_solver.py and MIGRATION_GUIDE.md")
        import traceback
        traceback.print_exc()
        return 1

    # Finalize simulation setup
    sim.finalize()
    print("Solver initialized and simulation finalized")

    # Memory check before simulation
    print("\nMemory usage before simulation:")
    print_memory_usage()

    # Run simulation
    print("\n" + "=" * 80)
    print("Starting Simulation")
    print("=" * 80 + "\n")

    traj = None
    try:
        if args.gpus == 1:
            print("Running on single GPU (device 0)...")
            traj = sim.run(solver=solver, cudaDevices=[0])
        else:
            gpu_list = list(range(args.gpus))
            print(f"Running on multiple GPUs: {gpu_list}...")
            traj = sim.run(solver=solver, cudaDevices=gpu_list)

        print("\nSimulation completed successfully!")

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user (Ctrl+C)")

    except Exception as e:
        print(f"\n\nAn error occurred during simulation:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()

    finally:
        # Close output file handles
        if hasattr(solver, 'save_cts_by_region_handle'):
            solver.save_cts_by_region_handle.close()
            print(f"Closed output file: {solver.save_cts_by_region_file}")

        if hasattr(solver, 'save_ode_data_handle'):
            solver.save_ode_data_handle.close()
            print(f"Closed output file: {solver.save_ode_data_file}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Simulation Summary")
    print("=" * 80)

    if traj is not None:
        print(f"Status: SUCCESS")
        if hasattr(solver, 'hook_time'):
            print(f"Total time in hookSimulation: {solver.hook_time:.2f} seconds")
    else:
        print(f"Status: FAILED or INTERRUPTED")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total simulation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    # Final memory usage
    print("\nFinal memory usage:")
    print_memory_usage()

    print("=" * 80)

    return 0 if traj is not None else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
