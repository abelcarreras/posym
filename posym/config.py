from dataclasses import dataclass


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass(frozen=False)
class Configuration(metaclass=Singleton):
    """
    Global configuration singleton for PoSym.

    Set fields before constructing symmetry objects to control
    permutation algorithms, orientation optimization, and labeling
    behavior.

    Example::

        from posym.config import Configuration
        Configuration().algorithm = 'exact'
        Configuration().label_tolerance = 0.1
    """
    fast_optimization: bool = True
    """If True, use a reduced set of symmetry elements during orientation
    optimization for speed."""
    scan_steps: int = 10
    """Number of angle steps for the Euler angle pre-scan."""
    algorithm : str = 'hungarian'
    """Permutation algorithm: ``'hungarian'`` (approximate, fast) or
    ``'exact'`` (brute-force, precise)."""
    label_tolerance: float = 1.0
    """Tolerance (in Angstrom) for merging chemically equivalent atoms
    into the same label group based on distance from the molecular center."""
