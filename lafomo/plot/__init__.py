from .plotter1d import Plotter1d
from .plotter2d import Plotter2d
from .misc import plot_spatiotemporal_data, plot_phase
from .colours import Colours
tight_kwargs = dict(bbox_inches='tight', pad_inches=0)


__all__ = [
    'Plotter1d',
    'Plotter2d',
    'plot_spatiotemporal_data',
    'plot_phase',
    'Colours',
    'tight_kwargs',
]
