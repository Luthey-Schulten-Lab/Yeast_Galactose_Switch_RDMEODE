#!/usr/bin/env python3
import numpy as np
import pickle
import lzma
import os

def create_combined_geometry_file(base_geometry_file="lattice_ER_tunnels_data_Marie.pkl.xz", 
                                output_file="yeast-lattice-combined.pkl.xz"):
    """
    Create one combined geometry file with all numpy arrays as lattice regions
    """
    print(f"Loading base geometry from: {base_geometry_file}")
    latticeData = pickle.load(lzma.open(base_geometry_file, "rb"))
    
    # Create new lattice (start with zeros)
    new_lattice = np.zeros_like(latticeData['lattice'])
    
    # Define the complete region names list
    region_names = [
        'extracellular', 'cellWall', 'nuclearEnvelope', 'mitochondria', 
        'vacuole', 'plasmaMembrane', 'nucleoplasm', 'nuclearPores', 'cytoplasm', 'ribosomes',
        # ER regions
        'pmaER', 'cecER', 'tubER', 'pmaRibosomes', 'cecRibosomes', 'tubRibosomes', 'cytoRibosomes',
        # Additional numpy array regions
        'gene_masks', 'dummy_chromosome',
        'effective_cyto_ribosomes_ER', 'dummy_cyto_ribosomes_ER', 
        'effective_er_ribosomes_ER', 'dummy_er_ribosomes_ER',
        'dummy_ribosomes_noER', 'effective_ribosomes_noER',
        'pmER_fixed_geometry', 'tubER_fixed_geometry', 'cecER_fixed_geometry',
        'combined_tubes_1', 'combined_tubes_2', 'combined_tubes_3'
    ]
    
    # Create site map
    siteMap_new = {name: i for i, name in enumerate(region_names)}
    print(f"Created site map with {len(region_names)} regions")
    
    # Get original site map for existing regions
    original_siteMap = {n: i for i, n in enumerate(latticeData['names'])}
    
    def original_boolLattice(x):
        if x in original_siteMap:
            return latticeData['lattice'] == original_siteMap[x]
        return np.zeros_like(latticeData['lattice'], dtype=bool)
    
    print("Assigning original regions...")
    
    # Assign original regions
    new_lattice[original_boolLattice('extracellular')] = siteMap_new['extracellular']
    new_lattice[original_boolLattice('cellWall')] = siteMap_new['cellWall']
    new_lattice[original_boolLattice('nuclearEnvelope')] = siteMap_new['nuclearEnvelope']
    new_lattice[original_boolLattice('mitochondria')] = siteMap_new['mitochondria']
    new_lattice[original_boolLattice('vacuole')] = siteMap_new['vacuole']
    new_lattice[original_boolLattice('plasmaMembrane')] = siteMap_new['plasmaMembrane']
    new_lattice[original_boolLattice('nucleoplasm')] = siteMap_new['nucleoplasm']
    new_lattice[original_boolLattice('nuclearPores')] = siteMap_new['nuclearPores']
    new_lattice[original_boolLattice('cytoplasm')] = siteMap_new['cytoplasm']
    new_lattice[original_boolLattice('ribosomes')] = siteMap_new['ribosomes']
    
    # Assign ER regions if they exist in original
    er_regions = ['pmaER', 'cecER', 'tubER', 'pmaRibosomes', 'cecRibosomes', 'tubRibosomes', 'cytoRibosomes']
    for region in er_regions:
        if region in original_siteMap:
            new_lattice[original_boolLattice(region)] = siteMap_new[region]
            print(f"  Assigned original {region}: {np.sum(original_boolLattice(region))} voxels")
    
    print("Loading and assigning numpy arrays...")
    
    # Dictionary of numpy files to load
    numpy_files = {
        'gene_masks': 'gene_masks.npy',
        'dummy_chromosome': 'dummy_chromosome.npy',
        'effective_cyto_ribosomes_ER': 'effective_cyto_ribosomes_ER_Marie.npy',
        'dummy_cyto_ribosomes_ER': 'dummy_cyto_ribosomes_ER_Marie.npy',
        'effective_er_ribosomes_ER': 'effective_er_ribosomes_ER_Marie.npy',
        'dummy_er_ribosomes_ER': 'dummy_er_ribosomes_ER_Marie.npy',
        'dummy_ribosomes_noER': 'dummy_ribosomes_noER.npy',
        'effective_ribosomes_noER': 'effective_ribosomes_noER.npy',
        'pmER_fixed_geometry': 'ER_geometry/pmER_fixed_geometry.npy',
        'tubER_fixed_geometry': 'ER_geometry/tubER_fixed_geometry.npy',
        'cecER_fixed_geometry': 'ER_geometry/cecER_fixed_geometry.npy',
        'combined_tubes_1': 'combined_tubes/combined_tubes_1.npy',
        'combined_tubes_2': 'combined_tubes/combined_tubes_2.npy',
        'combined_tubes_3': 'combined_tubes/combined_tubes_3.npy'
    }
    
    # Load and assign numpy arrays
    for region_name, filepath in numpy_files.items():
        if os.path.exists(filepath):
            print(f"  Loading {filepath}")
            region_array = np.load(filepath).astype(bool)
            
            # Verify shape matches
            if region_array.shape != new_lattice.shape:
                print(f"    Error: {filepath} shape {region_array.shape} doesn't match lattice shape {new_lattice.shape}")
                continue
            
            # Assign to lattice
            new_lattice[region_array] = siteMap_new[region_name]
            voxel_count = np.sum(region_array)
            print(f"    Assigned {region_name}: {voxel_count} voxels to index {siteMap_new[region_name]}")
        else:
            print(f"    Warning: {filepath} not found, skipping {region_name}")
    
    # Update latticeData
    latticeData['lattice'] = new_lattice
    latticeData['names'] = region_names
    
    print(f"Final lattice unique values: {np.unique(new_lattice)}")
    print(f"Total regions: {len(region_names)}")
    
    # Save the combined file
    print(f"Saving combined geometry to: {output_file}")
    with lzma.open(output_file, 'wb') as f:
        pickle.dump(latticeData, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("✓ Successfully created combined geometry file!")
    return output_file

def test_combined_geometry(geometry_file):
    """Test the combined geometry file"""
    print(f"\n=== Testing {geometry_file} ===")
    
    latticeData = pickle.load(lzma.open(geometry_file, "rb"))
    siteMap = {n: i for i, n in enumerate(latticeData['names'])}
    
    def boolLattice(x):
        return latticeData['lattice'] == siteMap[x]
    
    print("Available regions:")
    for name in latticeData['names']:
        region_mask = boolLattice(name)
        voxel_count = np.sum(region_mask)
        if voxel_count > 0:
            print(f"  ✓ {name}: {voxel_count} voxels")
        else:
            print(f"  - {name}: 0 voxels (empty)")
    
    # Test some key regions
    test_regions = ['cytoplasm', 'nucleoplasm', 'gene_masks', 'pmER_fixed_geometry']
    print("\nTesting key regions:")
    for region in test_regions:
        if region in siteMap:
            mask = boolLattice(region)
            count = np.sum(mask)
            print(f"  boolLattice('{region}'): {count} voxels")

if __name__ == "__main__":
    # Create the combined geometry file
    output_file = create_combined_geometry_file()
    
    # Test it
    if os.path.exists(output_file):
        test_combined_geometry(output_file)
    
    print(f"\n=== Usage ===")
    print(f"Use in your simulation with: -geo {output_file}")
    print("All regions now accessible via boolLattice() function!")