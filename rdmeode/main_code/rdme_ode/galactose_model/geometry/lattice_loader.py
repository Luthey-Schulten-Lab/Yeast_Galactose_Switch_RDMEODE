"""Lattice geometry loading from pickle files"""

import pickle
import lzma


def load_lattice_data(geometry_file, if_dgx=False):
    """Load lattice data from compressed pickle file

    Args:
        geometry_file (str): Name of the geometry file
        if_dgx (bool): Whether running on DGX system

    Returns:
        dict: Lattice data containing 'lattice', 'names', etc.
    """
    dir_prefix = "workspace/" if if_dgx else ""
    filepath = dir_prefix + geometry_file

    print(f"Loading lattice from: {filepath}")
    lattice_data = pickle.load(lzma.open(filepath, "rb"))

    return lattice_data


def get_bool_lattice(lattice_data):
    """Create a function to get boolean lattice for a given region name

    Args:
        lattice_data (dict): Lattice data from load_lattice_data

    Returns:
        function: Function that takes region name and returns boolean array
    """
    site_map = {name: idx for idx, name in enumerate(lattice_data['names'])}

    def bool_lattice(region_name):
        """Get boolean lattice for a specific region

        Args:
            region_name (str): Name of the region

        Returns:
            numpy.ndarray: Boolean array where True indicates the region
        """
        return lattice_data['lattice'] == site_map[region_name]

    return bool_lattice
