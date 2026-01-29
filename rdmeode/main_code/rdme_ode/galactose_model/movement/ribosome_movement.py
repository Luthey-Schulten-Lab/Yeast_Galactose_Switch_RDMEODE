"""Ribosome movement manager

This module handles ribosome movement using random diffusion and site swapping.
Based on the updated implementation that:
1. Builds region cache for ribosome-containing sites
2. Randomly diffuses ribosomes to neighboring sites
3. Swaps site types and particles when ribosomes move to different regions
"""

import time
import random
import numpy as np


class RibosomeMovementManager:
    """Manages ribosome movement through region swapping"""

    def __init__(self, rdme_sim, enable_er=False, enable_effective_ribosome=False):
        """Initialize ribosome movement manager

        Args:
            rdme_sim: RDME simulation object
            enable_er (bool): Whether ER is enabled
            enable_effective_ribosome (bool): Whether using effective ribosomes
        """
        self.rdme = rdme_sim
        self.enable_er = enable_er
        self.enable_effective_ribosome = enable_effective_ribosome
        self.move_counter = 0

    def move_ribosomes_hook(self, lattice, ribo_species, deleteParticle_func):
        """Move ribosomes by random diffusion and site swapping

        This method:
        1. Builds a cache of ribosome region sites
        2. For each ribosome-containing site, randomly picks a neighbor
        3. If neighbor has different site type, swaps site types and moves particles

        Args:
            lattice: Lattice object with particle and site data
            ribo_species (list): List of ribosome species objects
            deleteParticle_func: Function to delete particles

        Returns:
            int: 2 if regions were modified, 0 otherwise
        """
        start_move_time = time.time()
        self.move_counter += 1

        # Get lattice views
        site_lattice = lattice.getSiteLatticeView()
        particle_lattice = lattice.getParticleLatticeView()
        allowed_region = self.rdme.reg.cytoplasm
        # Get ribosome region IDs
        if self.enable_er:
            if self.enable_effective_ribosome:
                ribo_regions_list = [
                    self.rdme.reg.cytoRibosomes,
                    self.rdme.reg.erRibosomes,
                    self.rdme.reg.dum_erRibosomes,
                    self.rdme.reg.dum_cytoRibosomes
                ]
            else:
                ribo_regions_list = [
                    self.rdme.reg.cytoRibosomes,
                    self.rdme.reg.pmaRibosomes,
                    self.rdme.reg.cecRibosomes,
                    self.rdme.reg.tubRibosomes
                ]
        else:
            ribo_regions_list = [self.rdme.reg.ribosomes]

        ribo_region_ids = {r.idx for r in ribo_regions_list}

        # Build region cache: map each region ID to list of its sites
        region_cache = {}
        shape = site_lattice.shape

        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    region_id = site_lattice[x, y, z]
                    if region_id in ribo_region_ids:
                        if region_id not in region_cache:
                            region_cache[region_id] = []
                        region_cache[region_id].append((x, y, z))

        # Track swaps to perform
        swaps = []  # List of (origin, destination, ribo_counts, origin_type, dest_type)
        modified = False

        # For each ribosome region, randomly diffuse ribosomes to neighbors
        for region_id, sites in region_cache.items():
            for origin in sites:
                ox, oy, oz = origin
                particle_data = particle_lattice[ox, oy, oz]

                if particle_data is not None:
                    # Check if this site has ribosomes
                    ribo_counts = {}
                    for ribo_sp in ribo_species:
                        count = particle_data.get(ribo_sp.idx, 0)
                        if count > 0:
                            ribo_counts[ribo_sp] = count

                    if ribo_counts:
                        # Get valid neighboring sites (6-connected)
                        neighbors = []
                        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                            nx, ny, nz = ox + dx, oy + dy, oz + dz
                            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                                neighbors.append((nx, ny, nz))

                        if neighbors:
                            # Randomly pick a neighbor for diffusion
                            destination = random.choice(neighbors)
                            dx, dy, dz = destination

                            # Only swap if site types differ
                            origin_site_type = site_lattice[ox, oy, oz]
                            dest_site_type = site_lattice[dx, dy, dz]

                            if origin_site_type != dest_site_type:
                                swaps.append((
                                    origin,
                                    destination,
                                    ribo_counts,
                                    origin_site_type,
                                    dest_site_type
                                ))
                                modified = True

        # Apply all swaps: swap site types and particles
        particles = lattice.getParticleLattice() if hasattr(lattice, 'getParticleLattice') else None

        for origin, destination, ribo_counts, origin_site_type, dest_site_type in swaps:
            ox, oy, oz = origin
            dx, dy, dz = destination

            # Swap site types
            site_lattice[ox, oy, oz] = dest_site_type
            site_lattice[dx, dy, dz] = origin_site_type

            # Swap particles: move ribosomes from origin to destination
            for ribo_sp, count in ribo_counts.items():
                # Remove from origin
                if particles is not None:
                    pid = ribo_sp.idx
                    for _ in range(count):
                        deleteParticle_func(particles, ox, oy, oz, pid)

                # Add to destination
                ribo_sp.placeParticle(dx, dy, dz, count)

        move_time = time.time() - start_move_time
        if modified:
            print(f"Moved ribosome regions: {len(swaps)} swaps performed "
                  f"in {move_time:.3f}s (hook {self.move_counter})")

        # Return 2 to indicate region movement
        return 2 if modified else 0
