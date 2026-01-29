import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def setup_publication_style(style='default', figure_size='medium', dpi=300):
    """
    Set up matplotlib with publication-ready styling.
    
    Parameters:
    - style: 'default', 'seaborn', or other matplotlib styles
    - figure_size: 'small', 'medium', 'large', or tuple (width, height) in inches
    - dpi: dots per inch for figure resolution
    """
    # Apply base style
    plt.style.use(style)
    
    # Define figure sizes
    size_presets = {
        'small': (6, 4.5),    # ~1800x1350 pixels at 300 DPI
        'medium': (10/3, 875/300),  # From notebook: 1000x875 pixels at 300 DPI  
        'large': (10, 7.5),   # ~3000x2250 pixels at 300 DPI
        'paper': (8.5, 6.4),  # Good for papers
    }
    
    if isinstance(figure_size, str):
        figsize = size_presets.get(figure_size, size_presets['medium'])
    else:
        figsize = figure_size
    
    # Set rcParams
    plt.rcParams.update({
        'figure.dpi': dpi,
        'figure.figsize': figsize,
        'font.family': 'Arial',
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 1.5,
        'lines.linewidth': 1.5,
        'patch.linewidth': 0.5,
        'grid.linewidth': 0.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'axes.prop_cycle': plt.cycler(color=cm.get_cmap('Dark2').colors)
    })
    
    # Return the color cycle for easy access
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return colors

# Usage example:
colors = setup_publication_style(figure_size='medium')
color1, color2, color3 = colors[0], colors[1], colors[2]