#!/usr/bin/env python
"""Script to automatically extract and organize code from galactose_rdmeode_combined_ribo_move.py
into the modular structure"""

import re
import os

# Read the original file
with open('galactose_rdmeode_combined_ribo_move.py', 'r') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Line ranges for different sections (0-indexed)
sections = {
    'geometry/region_builder.py': (155, 1027),  # From region definitions to before OdeRdmeHybridSolver
    'solvers/hybrid_solver.py': (1037, 1431),   # OdeRdmeHybridSolver class
}

# Extract the geometry/region_builder section and wrap it in a function
print("\nExtracting region builder...")
region_builder_lines = lines[155:1027]

# Create the complete region_builder.py
region_builder_content = '''"""Build spatial regions for RDME simulation

This module constructs all spatial regions including:
- Basic cellular compartments (cytoplasm, nucleus, etc.)
- Optional chromosome regions
- Optional ER regions
- Ribosome regions (effective or standard)
- Species definitions
- Reactions
- Initial conditions
- Diffusion coefficients
"""

import numpy as np
import os
import pickle
from jLM.RDME import Sim as RDMESim
from jLM.RegionBuilder import RegionBuilder
from .lattice_loader import get_bool_lattice


def build_regions(lattice_data, args):
    """Build all spatial regions for the simulation

    Args:
        lattice_data (dict): Loaded lattice data
        args: Parsed command-line arguments

    Returns:
        RDMESim: Configured simulation object with all regions defined
    """
    # Extract arguments
    enable_chromosome = args.enable_chromosome
    enable_er = args.enable_er
    enable_effective_ribosome = args.enable_effective_ribosome
    enable_rna_tracking = args.enable_rna_tracking
    er_num = args.er_num
    gene_location = args.gene_location
    checkpoint_file = args.checkpoint
    simtime = args.simtime
    externalGal_input = args.galactose
    output_order = args.index
    gpus = args.gpus
    if_dgx = args.if_dgx

    dir_dgx = "workspace/" if if_dgx else ""

    # Get boolean lattice function
    bool_lattice_func = get_bool_lattice(lattice_data)

    # Create closure to access latticeData
    boolLattice = bool_lattice_func

'''

# Add the extracted region building code
region_builder_content += ''.join(['    ' + line for line in region_builder_lines])

# Add return statement
region_builder_content += '''
    return sim
'''

# Write region_builder.py
output_file = 'galactose_model/geometry/region_builder.py'
with open(output_file, 'w') as f:
    f.write(region_builder_content)

print(f"✓ Created {output_file}")

# Extract solver class
print("\nExtracting hybrid solver...")
solver_lines = lines[1037:1431]

solver_content = '''"""OdeRdmeHybridSolver class

Hybrid solver that combines ODE and RDME simulations for the galactose pathway.
"""

import numpy as np
import scipy.integrate as spint
import json
import time
import os
from jLM.RDME import File as RDMEFile, Sim as RDMESim
from lm_functions import deleteParticle
from movement.ribosome_movement import RibosomeMovementManager


'''

# Add the extracted solver code
solver_content += ''.join(solver_lines)

# Write hybrid_solver.py
output_file = 'galactose_model/solvers/hybrid_solver.py'
with open(output_file, 'w') as f:
    f.write(solver_content)

print(f"✓ Created {output_file}")

print("\n✓ Extraction complete!")
print("\nNext steps:")
print("1. Review the extracted files for any missing imports")
print("2. Fix variable references (e.g., global variables)")
print("3. Test the modular structure")
