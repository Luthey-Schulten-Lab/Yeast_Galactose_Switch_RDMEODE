"""Argument parsing and output path configuration

This module handles all command-line arguments and sets up output paths.
"""

import argparse
import os
import datetime


def parse_arguments():
    """Parse command-line arguments for the simulation

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Combined RDME/ODE simulation with optional chromosome and ER support'
    )

    # Basic simulation parameters
    parser.add_argument('-id', '--index', type=int, required=True,
                        help='Index of the output lm files')
    parser.add_argument('-t', '--simtime', type=float, default=60,
                        help='Simulation time in minutes')
    parser.add_argument('-g', '--galactose', type=float, default=11.1,
                        help='External galactose concentration (mM)')

    # Computation parameters
    parser.add_argument('-gpus', '--gpus', type=int, default=1,
                        help='Number of GPUs to use (default: 1)')
    parser.add_argument('-mt', '--max_time', type=float, default=1000,
                        help='Maximum allowed simulation time in hours')

    # File paths and tagging
    parser.add_argument('-tag', '--tag', type=str, default='',
                        help='Tag for the output folder')
    parser.add_argument('-geo', '--geometry', type=str, default='yeast-lattice.2.pkl.xz',
                        help='Geometry file name')
    parser.add_argument('-ckpt', '--checkpoint', type=str, default='',
                        help='Checkpoint file name (empty for no checkpoint)')

    # Gene location
    parser.add_argument('-geloc', '--gene_location', type=str, default='random',
                        help='Location of the genes (random, center, edge, chromosome)')

    # Feature flags
    parser.add_argument('--enable-chromosome', action='store_true', default=False,
                        help='Enable chromosome regions and related functionality')
    parser.add_argument('--enable-er', action='store_true', default=False,
                        help='Enable ER regions and related functionality')
    parser.add_argument('-ernum', '--er_num', type=int, default=4,
                        help='Number of ER tunnels')
    parser.add_argument('--enable-effective-ribosome', action='store_true', default=False,
                        help='Enable effective ribosome case (includes both ribosome and ribosome_dummy)')
    parser.add_argument('--enable_rna_tracking', action='store_true', default=False,
                        help='Enable RNA tracking functionality')

    # Ribosome movement parameters
    parser.add_argument('--enable-ribosome-movement', action='store_true', default=False,
                        help='Enable ribosome movement')
    parser.add_argument('--ribosome-movement-mode', type=str, default='hook',
                        choices=['diffusion', 'hook'],
                        help='Movement mode: diffusion (fast, solver-handled) or hook (controlled, slower)')
    parser.add_argument('--ribosome-diffusion-rate', type=float, default=1e-14,
                        help='Diffusion rate for ribosomes (m^2/s) when using diffusion mode')
    parser.add_argument('--ribosome-move-fraction', type=float, default=1.0,
                        help='Fraction of ribosomes to move per hook call (0.0-1.0) when using hook mode')
    parser.add_argument('--ribosome-move-interval', type=int, default=1,
                        help='Move ribosomes every N hook calls (reduces overhead, default 1)')

    # System flags
    parser.add_argument('--if-dgx', action='store_true', default=False,
                        help='Running on DGX system')

    args = parser.parse_args()

    # Auto-adjust gene location if chromosome is enabled
    if args.enable_chromosome:
        args.gene_location = "chromosome"

    # Validate ER geometry file
    if args.enable_er and "_ER_" not in args.geometry:
        raise ValueError("ER support requires _ER_... geometry file")

    return args


def setup_output_paths(args):
    """Setup output directory and file paths

    Args:
        args: Parsed command-line arguments

    Returns:
        tuple: (output_folder, output_dir)
    """
    date = datetime.datetime.now().strftime("%Y%m%d")
    version = "1.17_modular"

    # Handle checkpoint directory
    if args.checkpoint:
        output_dir = os.path.dirname(args.checkpoint)
        print(f"Using directory from checkpoint file: {output_dir}")
    else:
        output_dir = f"simulation_results_id_{args.index}"

    # Create feature suffix
    feature_suffix = ""
    if args.enable_er:
        feature_suffix += "_ER"
    if args.enable_chromosome:
        feature_suffix += "_CHROMO"
    if args.enable_effective_ribosome:
        feature_suffix += "_EFFRIBO"
    if args.enable_ribosome_movement:
        feature_suffix += f"_RIBO_MOVE_{args.ribosome_movement_mode}"

    # DGX path prefix
    dir_prefix = "workspace/" if args.if_dgx else ""

    # Build base name
    base_name = f"yeast{version}_{date}_{args.index}_t{args.simtime}min_GAE{args.galactose}mM{feature_suffix}{args.tag}"
    if args.gpus > 1:
        base_name += f"_gpu{args.gpus}"

    # Create full output path
    output_folder = base_name + ".lm"

    if args.if_dgx:
        output_folder = os.path.join(dir_prefix, output_folder)
        output_dir = os.path.join(dir_prefix, output_dir)
    else:
        output_folder = os.path.join(output_dir, output_folder)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_folder, output_dir
