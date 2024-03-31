from dataclasses import dataclass


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass(frozen=False)
class Configuration(metaclass=Singleton):
    fast_optimization: bool = True  # speeds up the optimization by reducing the number of symmetry elements to use
    scan_steps: int = 20            # number of angle steps to use in the pre-scan in orientation optimization
    algorithm : str = 'hungarian'   # permutation algorithms: hungarian, exact


@dataclass(frozen=False)
class CustomPerm(metaclass=Singleton):
    perm_list: list = None
