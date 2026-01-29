"""Geometry module for lattice loading and region definitions"""

from .lattice_loader import load_lattice_data, get_bool_lattice
from .region_builder import build_regions

__all__ = ['load_lattice_data', 'get_bool_lattice', 'build_regions']
