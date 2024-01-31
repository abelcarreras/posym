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
                    # print('perm', p)
                    while orbit[0] != self._permutation[p]:
                        p = self._permutation[p]
                        # print('p', p)
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


def roll_permutation(permutation, n):
    p = Permutation(permutation)
    return p.slide(n-1)
