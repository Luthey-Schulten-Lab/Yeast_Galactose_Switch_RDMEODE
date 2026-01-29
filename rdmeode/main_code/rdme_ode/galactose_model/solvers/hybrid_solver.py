"""OdeRdmeHybridSolver class

Hybrid solver that combines ODE and RDME simulations for the galactose pathway.
"""

import numpy as np
import scipy.integrate as spi
import json
import time
import os
import sys
from jLM.RDME import File as RDMEFile, Sim as RDMESim

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lm_functions import deleteParticle, getParticlesInSite

# Import movement manager from sibling package
from movement.ribosome_movement import RibosomeMovementManager
from utils.json_encoder import NumpyEncoder
from utils.memory_monitor import print_memory_usage


class OdeRdmeHybridSolver:
    """Hybrid solver for Galactose switch
    
    Transport reactions and galactose metabolism handled by ODE, rest by RDME. The
    ODE system is coupled to the RDME by updating the protein counts in the ODE 
    each communication step. New proteins are added to the state unbound to
    galactose. If the number of proteins decreases, both bound and unbound 
    states are decreased by the same proportion and the galactose is
    added back as internal or external galactose. The RDME is coupled to the 
    ODE system through the internal galactose concentration. The rate of
    G3 -> G3i is updated with the internal galactose concentration each
    communication step."""
    
    def __init__(self, lmFile, initialExternalGalactose, output_folder=None, args=None):
        super(OdeRdmeHybridSolver, self).__init__()
        self.GAE = initialExternalGalactose #M
        self.odeYs = None
        self.lastOdeEval = 0
        self.odeEvals = []
        self.odeSpNames = ['GAI', 'G1', 'G1GAI', 'G2GAI', 'G2GAE', 'G2']
        if isinstance(lmFile, (RDMEFile, RDMESim)):
            self.rdme = lmFile
        else:
            self.rdme = RDMEFile(lmFile)
        self.cellVol = self.rdme.reg.cytoplasm.volume + self.rdme.reg.nucleoplasm.volume + self.rdme.reg.plasmaMembrane.volume
        self.NAV = 6.022e23*self.cellVol

        self.g3actRidx = self.rdme.reaction(self.rdme.sp.G3, self.rdme.sp.G3i, self.rdme.rc.fi).idx
        self.g3actRc = self.rdme.rc.fi._toLM()

        # Store configuration from args
        if args is not None:
            self.checkpoint_file = args.checkpoint
            self.enable_er = args.enable_er
            self.enable_effective_ribosome = args.enable_effective_ribosome
            self.enable_ribosome_movement = args.enable_ribosome_movement
            self.ribosome_movement_mode = args.ribosome_movement_mode if args.enable_ribosome_movement else None
            self.ribosome_move_interval = args.ribosome_move_interval if args.enable_ribosome_movement else 1
            self.max_time = args.max_time
        else:
            # Defaults if no args provided
            self.checkpoint_file = ""
            self.enable_er = False
            self.enable_effective_ribosome = False
            self.enable_ribosome_movement = False
            self.ribosome_movement_mode = None
            self.ribosome_move_interval = 1
            self.max_time = 1000

        # Open output files
        if output_folder is None:
            output_folder = "simulation_output"
        self.output_folder = output_folder
        self.save_cts_by_region_file = output_folder + "_region.jsonl"
        self.save_cts_by_region_handle = open(self.save_cts_by_region_file, "w")
        self.hook_time = 0

        self.save_ode_data_file = output_folder + "_ode.jsonl"
        self.save_ode_data_handle = open(self.save_ode_data_file, "w")

        # Ribosome movement setup (for hook mode)
        self.ribosome_move_counter = 0
        self.valid_ribo_positions = None
        self.ribo_positions_cached = False
        self.start_time = time.time()  # Store start time for max_time check
    
    def copyInitialConditions(self, cts):
        if self.checkpoint_file == "":
            y = np.zeros(len(self.odeSpNames))
            y[self.odeSpIndex("GAI")] = 0
            y[self.odeSpIndex("G1")] = cts['countBySpecies'][self.rdme.sp.G1]/self.NAV 
            y[self.odeSpIndex("G1GAI")] = 0
            y[self.odeSpIndex("G2")] = cts['countBySpecies'][self.rdme.sp.G2]/self.NAV
            y[self.odeSpIndex("G2GAE")] = 0
            y[self.odeSpIndex("G2GAI")] = 0
        else:
            print(f"using checkpoint:{self.checkpoint_file}")
            checkpoint_ode = self.checkpoint_file + "_ode.jsonl"
           
            with open(checkpoint_ode, 'r') as f:
                last_line = None
                for line in f:
                    last_line = line
                
                if last_line is None:
                    raise RuntimeError(f"ODE checkpoint file {checkpoint_ode} is empty")
                
                last_ode_state = json.loads(last_line.strip())
                if 'species' not in last_ode_state:
                    raise RuntimeError(f"Invalid ODE state format in {checkpoint_ode}")
                
                y = np.zeros(len(self.odeSpNames))
                for i, name in enumerate(self.odeSpNames):
                    y[self.odeSpIndex(name)] = last_ode_state['species'][name]
                
                print(f"Initialized ODE state from time {last_ode_state['time']}")
        
        self.boundGal = self.rdmeGal(cts)
        return y
    
    def rdmeGal(self, cts):
        return (cts['countBySpecies'][self.rdme.sp.G3i] + cts['countBySpecies'][self.rdme.sp.G80d_G3i])/self.NAV

    def rdme2odeConc(self, y0, cts):
        y = y0.copy()
        
        # Update G1 in ODE
        g1ode = y0[self.odeSpIndex("G1")]
        g1gaiode = y0[self.odeSpIndex("G1GAI")]
        g1rdme = cts['countBySpecies'][self.rdme.sp.G1]/self.NAV
        change = g1rdme-g1ode-g1gaiode
        
        if change > 0:
            y[self.odeSpIndex("G1")] = g1ode + change
        else:
            fracChange = g1rdme/(g1ode+g1gaiode) if (g1ode+g1gaiode) > 0 else 0
            y[self.odeSpIndex("G1")] = g1ode*fracChange
            y[self.odeSpIndex("G1GAI")] = g1gaiode*fracChange
            y[self.odeSpIndex("GAI")] += g1gaiode*(1-fracChange)
            
        # Update G2 in ODE
        g2ode = y0[self.odeSpIndex("G2")]
        g2gaiode = y0[self.odeSpIndex("G2GAI")]
        g2gaeode = y0[self.odeSpIndex("G2GAE")]
        g2rdme = cts['countBySpeciesRegion'][self.rdme.sp.G2][self.rdme.reg.plasmaMembrane]/self.NAV
        
        change = g2rdme-g2ode-g2gaiode-g2gaeode
        
        if change >= 0:
            y[self.odeSpIndex("G2")] = g2ode + change
        else:
            total = g2ode+g2gaiode+g2gaeode
            fracChange = g2rdme/total if total > 0 else 0
            y[self.odeSpIndex("G2")] = g2ode*fracChange
            y[self.odeSpIndex("G2GAI")] = g2gaiode*fracChange
            y[self.odeSpIndex("GAI")] += g2gaiode*(1-fracChange)
            y[self.odeSpIndex("G2GAE")] = g2gaeode*fracChange

        # Update internal galactose in ODE
        g0 = self.boundGal
        g1 = self.rdmeGal(cts)
        y[self.odeSpIndex("GAI")] += g1-g0
        self.boundGal = g1
            
        return y
                   
    def _cache_valid_ribo_positions(self, site_lattice):
        """Cache valid ribosome positions once for efficiency"""
        if self.ribo_positions_cached and self.valid_ribo_positions is not None:
            return self.valid_ribo_positions
        
        valid_positions = np.zeros(site_lattice.shape, dtype=np.int32)
        if self.enable_er:
            if self.enable_effective_ribosome:
                ribo_regions_list = [self.rdme.reg.cytoRibosomes, self.rdme.reg.erRibosomes, self.rdme.reg.dum_erRibosomes, self.rdme.reg.dum_cytoRibosomes]
            else:
                ribo_regions_list = [self.rdme.reg.cytoRibosomes, self.rdme.reg.pmaRibosomes, 
                                    self.rdme.reg.cecRibosomes, self.rdme.reg.tubRibosomes]
        else:
            ribo_regions_list = [self.rdme.reg.ribosomes]
        
        for ribo_reg in ribo_regions_list:
            positions = np.argwhere(site_lattice == ribo_reg.idx)
            print(f"positions: {positions}")
            valid_positions[positions] = 1
        
        self.valid_ribo_positions = valid_positions
        # this is saved as 
        self.ribo_positions_cached = True
        print(f"Cached {len(self.valid_ribo_positions)} valid ribosome positions")
        return self.valid_ribo_positions
    
    def _move_ribosomes_hook(self, lattice):
        """Move ribosome regions by tracking diffusion and updating particle positions"""
        if not self.enable_ribosome_movement or self.ribosome_movement_mode != 'hook':
            return 0
        
        # Only move every N hook calls to reduce overhead
        self.ribosome_move_counter += 1
        if self.ribosome_move_counter % self.ribosome_move_interval != 0:
            return 0
        
        start_move_time = time.time()
        particle_lattice = lattice.getParticleLatticeView()
        site_lattice = lattice.getSiteLatticeView()
        
        # Get raw particle array for direct manipulation (needed for deleteParticle)
        # The array shape should be [replicates, z, y, x, particles_per_site]
        
        
        # Get all ribosome species
        ribo_species = [s for s in self.rdme.speciesList if 'ribosome' in s.name.lower()]
        
        # Get ribosome region IDs
        if self.enable_er:
            if self.enable_effective_ribosome:
                ribo_regions_list = [self.rdme.reg.cytoRibosomes, self.rdme.reg.erRibosomes, 
                                    self.rdme.reg.dum_erRibosomes, self.rdme.reg.dum_cytoRibosomes]
            else:
                ribo_regions_list = [self.rdme.reg.cytoRibosomes, self.rdme.reg.pmaRibosomes, 
                                    self.rdme.reg.cecRibosomes, self.rdme.reg.tubRibosomes]
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
        swaps = []  # List of (origin, destination, ribo_counts) tuples
        modified = False

        # For each ribosome region, randomly diffuse ribosomes to neighbors
        for region_id, sites in region_cache.items():
            for origin in sites:
                ox, oy, oz = origin
                
                # Check if this site has ribosomes using getParticlesInSite
                ps = getParticlesInSite(particle_lattice, ox, oy, oz)
                ribo_counts = {}
                for ribo_sp in ribo_species:
                    # Count instances of this ribosome species in the site
                    count = np.sum(ps == ribo_sp.idx)
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
                        import random
                        destination = random.choice(neighbors)
                        dx, dy, dz = destination

                        # Only swap if site types differ (i.e., ribosome is moving to different region)
                        origin_site_type = site_lattice[ox, oy, oz]
                        dest_site_type = site_lattice[dx, dy, dz]

                        if origin_site_type != dest_site_type:
                            swaps.append((origin, destination, ribo_counts, origin_site_type, dest_site_type))
                            modified = True

        # Apply all swaps: swap site types and particles
        for origin, destination, ribo_counts, origin_site_type, dest_site_type in swaps:
            ox, oy, oz = origin
            dx, dy, dz = destination

            # Swap site types
            site_lattice[ox, oy, oz] = dest_site_type
            site_lattice[dx, dy, dz] = origin_site_type

            # Swap particles: move ribosomes from origin to destination
            for ribo_sp, count in ribo_counts.items():
                pid = ribo_sp.idx
                for _ in range(count):
                    # Remove from origin
                    deleteParticle(particle_lattice, ox, oy, oz, pid)

                # Add to destination
                ribo_sp.placeParticle(dx, dy, dz, count)

        move_time = time.time() - start_move_time
        if modified:
            print(f"Moved ribosome regions: {len(swaps)} swaps performed "
                  f"in {move_time:.3f}s (hook {self.ribosome_move_counter})")
        
        # Return 2 to indicate region movement (different from regular particle modification)
        return 2 if modified else 0
    
    def hookSimulation(self, t, lattice):
        print_memory_usage()
        start_time_hook = time.time()
        
        cts = self.rdme.particleStatistics(particleLattice=lattice.getParticleLatticeView(),
                                           siteLattice=lattice.getSiteLatticeView())
        if self.odeYs is None:
            ys0 = self.copyInitialConditions(cts)
        else:
            ys0 = self.rdme2odeConc(self.odeYs, cts)
                   
        dt = t-self.lastOdeEval
        if dt>0:
            ys1 = self.stepOde(dt, ys0)
        else:
            ys1 = ys0
            
        self.odeEvals.append((t,ys1))
        self.odeYs = ys1
        self.lastOdeEval = t
            
        assocRt = max(0,self.g3actRc*ys1[self.odeSpIndex("GAI")])
        self.setReactionRate(self.g3actRidx, assocRt)
        
        # Ribosome movement (hook mode) - tracks diffusion and moves regions
        ribo_move_result = self._move_ribosomes_hook(lattice)
        
        self.save_rdme_cts_by_region(t, cts)
        self.save_ode_data(t, ys1)
        self.print_ode_evals(t,assocRt,cts)
        
        end_time_hook = time.time()
        self.hook_time += end_time_hook - start_time_hook
        
        if self.max_time is not None and (end_time_hook - self.start_time) >= self.max_time * 3600:
            print(f"Maximum simulation time of {self.max_time} hours reached. Stopping simulation.")
            return 3
        
        # Return ribo_move_result (2 for region movement, 0 for no change)
        # This signals the solver that regions were moved
        return ribo_move_result

    def print_ode_evals(self,t,assocRt,cts):
        print("="*80)
        print("t=",t)
        print("ODE")
        for i,n in enumerate(self.odeSpNames):
            print("  {:<16s}{:16.5g}".format(n,self.odeYs[i]))
        print("RDME")
        for n in self.rdme.speciesList:
            print("  {:<16s}{:16d}".format(n.name,cts['countBySpecies'][n]))
        print("new rate g3 activation: {:.3g}".format(assocRt))
        print("-"*80)
        return 

    def odeSpIndex(self, sp):
        return self.odeSpNames.index(sp)
    
    def ode_model(self,conc, ts, GAE):
        NA = 6.02214076e23
        kf_GK = 1.442e5  # M^-1 s^-1 
        kr_GK = 30.708   # s^-1
        kcat_GK = 55.833 # s^-1
        kcat_TR = 72.5   # s^-1
        kr_TR = 39.875   # s^-1
        kf_TR = 1.123e5  # M^-1 s^-1 
        kf_TR_gae = 1.123e5* GAE # s^-1
        
        GAI = conc[self.odeSpIndex("GAI")]
        G2GAI = conc[self.odeSpIndex("G2GAI")]
        G2GAE = conc[self.odeSpIndex("G2GAE")]
        G1GAI = conc[self.odeSpIndex("G1GAI")]
        G1 = conc[self.odeSpIndex("G1")]
        G2 = conc[self.odeSpIndex("G2")]
    
        # GAI
        dGAI_dt = kr_TR*G2GAI - kf_TR*GAI*G2 + kr_GK*G1GAI - kf_GK*G1*GAI
        # G1
        dG1_dt =  kr_GK*G1GAI - kf_GK*G1*GAI + kcat_GK*G1GAI
        # G1GAI
        dG1GAI_dt = kf_GK*G1*GAI - kr_GK*G1GAI - kcat_GK*G1GAI
        # G2
        dG2_dt = kr_TR*G2GAI - kf_TR*G2*GAI + kr_TR*G2GAE - kf_TR_gae*G2
        # G2GAE
        dG2GAE_dt = kf_TR_gae*G2 - kr_TR*G2GAE - kcat_TR*G2GAE + kcat_TR*G2GAI
        # G2GAI
        dG2GAI_dt = kf_TR*G2*GAI - kr_TR*G2GAI - kcat_TR*G2GAI + kcat_TR*G2GAE
       
        dx_dt = [0] * len(self.odeSpNames)
        dx_dt[self.odeSpIndex("GAI")] = dGAI_dt
        dx_dt[self.odeSpIndex("G1")] = dG1_dt
        dx_dt[self.odeSpIndex("G1GAI")] = dG1GAI_dt
        dx_dt[self.odeSpIndex("G2")] = dG2_dt
        dx_dt[self.odeSpIndex("G2GAI")] = dG2GAI_dt
        dx_dt[self.odeSpIndex("G2GAE")] = dG2GAE_dt
        dx_dt_array = np.asarray(dx_dt)
        return (dx_dt_array)
    
    def stepOde(self, dt, ys0):
        odestep = 0.001
        ts = np.linspace(0,dt, int(np.ceil(dt/odestep))+1)
        ys = spi.odeint(self.ode_model, ys0, ts, args=(self.GAE,), hmax=odestep)
        return ys[-1]

    def save_rdme_cts_by_region(self, t, stats):
        counts_by_region = {'time': float(t)}

        for species in self.rdme.speciesList:
            counts_by_region[species.name] = {}
            for region in self.rdme.regionList:
                count = stats['countBySpeciesRegion'][species][region]
                counts_by_region[species.name][region.name] = int(count)

        json.dump(counts_by_region, self.save_cts_by_region_handle, cls=NumpyEncoder)
        self.save_cts_by_region_handle.write('\n')
        self.save_cts_by_region_handle.flush()

        print(f"Data for time {t} appended to {self.save_cts_by_region_file}")
        return counts_by_region

    def save_ode_data(self, t, ys):
        ode_data = {
            'time': float(t),
            'species': {name: float(value) for name, value in zip(self.odeSpNames, ys)}
        }

        json.dump(ode_data, self.save_ode_data_handle, cls=NumpyEncoder)
        self.save_ode_data_handle.write('\n')
        self.save_ode_data_handle.flush()

        print(f"ODE data for time {t} appended to {self.save_ode_data_file}")

