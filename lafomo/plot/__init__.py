from .plotter import Plotter
from .misc import plot_spatiotemporal_data, plot_phase
from .colours import Colours
tight_kwargs = dict(bbox_inches='tight', pad_inches=0)


__all__ = [
    'Plotter',
    'plot_spatiotemporal_data',
    'plot_phase',
    'Colours',
    'tight_kwargs',
]
