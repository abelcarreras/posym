import numpy as np
import copy


class Permutation:
    def __init__(self, permutation):
        self._permutation = copy.deepcopy(list(permutation))
        self._orbits = None

    def __str__(self):
        return self._permutation.__str__()

    def __iter__(self):
        return self._permutation.__iter__()

    def __getitem__(self, item):
        return self._permutation.__getitem__(item)

    def get_orbits(self):
        if self._orbits is None:
            self._orbits = []
            track_pos = []

            for p in self._permutation:

                if not p in track_pos:
                    track_pos.append(p)
                    orbit = [p]
                    while orbit[0] != self._permutation[p]:
                        p = self._permutation[p]
                        track_pos.append(p)
                        orbit.append(p)
                    self._orbits.append(orbit)

        return self._orbits

    def len_orbits(self):
        len_orbits = []
        for orbit in self.get_orbits():
            len_orbits.append(len(orbit))

        return len_orbits

    def max_orbit(self):
        max_orb = 0
        for orbit in self.get_orbits():
            max_orb = len(orbit) if len(orbit) > max_orb else max_orb

        return max_orb

    def slide(self, n):
        slide_permut = np.array(self._permutation)

        for orbit in self.get_orbits():
            original = list(orbit)
            roll_orbit = np.roll(orbit, -n)

            slide_permut[original] = slide_permut[roll_orbit]

        return slide_permut

    @property
    def raw_list(self):
        return self._permutation

    def __len__(self):
        return len(self._permutation)


def generate_permutation_set(generators, symbols):

    from itertools import permutations, product

    n_atoms = len(symbols)

    def symbols_compatible(len_orbits):
        val = 1
        for lo in len_orbits:
            val *= len(np.unique(np.array(symbols)[lo]))
        if val == 1:
            return True
        return False

    def gen_perm(gen):
        order = gen._order
        determinant = gen._determinant
        for p in permutations(range(n_atoms)):
            p_object = Permutation(p)
            len_orbits = p_object.len_orbits()
            orbits = p_object.get_orbits()

            # check symbols restriction
            if not symbols_compatible(orbits):
                continue

            orbit_mod = np.ones_like(len_orbits)
            orbit_mod = np.multiply(orbit_mod, np.mod(order, len_orbits))

            if determinant < 0:
                orbit_mod = np.multiply(orbit_mod, np.mod(2 * order, len_orbits))
                if order != 2:
                    orbit_mod = np.multiply(orbit_mod, np.mod(2, len_orbits))

            if np.sum(orbit_mod) == 0:
                yield p

    # for perm_set in product(*gen_perm_list):
    for perm_set in product(*[gen_perm(gen) for gen in generators]):
        yield {k: pi for k, pi in zip(generators, perm_set)}


class PermutationSet():
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        if 'all' in self._data:
            return self._data['all']
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __contains__(self, key):
        return 'all' in self._data or key in self._data

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, key, default=None):
        if 'all' in self._data:
            return self._data.get('all', default)

        return self._data.get(key, default)

    def clear(self):
        self._data.clear()

    def update(self, other):
        if isinstance(other, PermutationSet):
            self._data.update(other._data)
        else:
            self._data.update(other)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._data})'
