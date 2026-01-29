import json

class OdeRdmeHybridSolver:
    def __init__(self, lmFile, initialExternalGalactose):
        # ... existing initialization code ...

        # Open the output file once during initialization
        self.save_cts_by_region_file = f"{self.rdme.outputFile}_species_counts_by_region.jsonl"
        self.save_cts_by_region_handle = open(self.save_cts_by_region_file, "w")  # Open in write mode to start fresh

    def save_rdme_cts_by_region(self, t):
        # Get the current particle and site lattices
        particle_lattice = self.rdme.particleLattice
        site_lattice = self.rdme.siteLattice

        # Get particle statistics
        stats = self.rdme.particleStatistics(particleLattice=particle_lattice, siteLattice=site_lattice)

        # Initialize a dictionary to store counts by species and region for this time step
        counts_by_region = {'time': t}

        # Iterate through all species and regions
        for species in self.rdme.speciesList:
            counts_by_region[species.name] = {}
            for region in self.rdme.regionList:
                count = stats['countBySpeciesRegion'][species][region]
                counts_by_region[species.name][region.name] = count

        # Write the current time step data to file
        json.dump(counts_by_region, self.save_cts_by_region_handle)
        self.save_cts_by_region_handle.write('\n')  # Add a newline for readability
        self.save_cts_by_region_handle.flush()  # Ensure data is written to disk

        print(f"Data for time {t} appended to {self.save_cts_by_region_file}")

        return counts_by_region

    def finalize(self):
        # Close the output file when the simulation is done
        self.save_cts_by_region_handle.close()
        print(f"Closed output file: {self.save_cts_by_region_file}")

    # ... rest of the class implementation ...