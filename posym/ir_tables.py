import pandas as pd
import numpy as np


class CharTable(pd.DataFrame):
    """
    Subclass of DataFrame to add some convenience

    """
    def __init__(self, data, name, rotations, translations, multiplicities):
        super().__init__(data)
        self.name = name

        if np.all([r in data for r in rotations]):
            self.attrs['rotations'] = rotations
        else:
            raise Exception('Rotations representations not in table')

        if np.all([r in data for r in translations]):
            self.attrs['translations'] = translations
        else:
            raise Exception('Translations representations not in table')

        if len(multiplicities) == len(self.index):
            self.attrs['multiplicities'] = multiplicities
        else:
            raise Exception('Multiplicities do not match')

    @property
    def rotations(self):
        return self.attrs['rotations']

    @property
    def translations(self):
        return self.attrs['translations']

    @property
    def multiplicities(self):
        return self.attrs['multiplicities']


ir_table_list = [
    CharTable({'Ag': pd.Series([+1, +1, +1, +1], index=['E', 'C2', 'i', 'sh']),
               'Bg': pd.Series([+1, -1, +1, -1], index=['E', 'C2', 'i', 'sh']),
               'Au': pd.Series([+1, +1, -1, -1], index=['E', 'C2', 'i', 'sh']),
               'Bu': pd.Series([+1, -1, -1, +1], index=['E', 'C2', 'i', 'sh']),
               },
              name='C2h',
              rotations=['Bg', 'Bg', 'Ag'],  # x, y, z
              translations=['Bu', 'Bu', 'Au'],  # Rx, Ry, Rz
              multiplicities=[1, 1, 1, 1]),

    CharTable({'A1': pd.Series([+1, +1, +1, +1], index=['E', 'C2', 'sv_xz', 'sv_yz']),
               'A2': pd.Series([+1, +1, -1, -1], index=['E', 'C2', 'sv_xz', 'sv_yz']),
               'B1': pd.Series([+1, -1, +1, -1], index=['E', 'C2', 'sv_xz', 'sv_yz']),
               'B2': pd.Series([+1, -1, -1, +1], index=['E', 'C2', 'sv_xz', 'sv_yz']),
               },
              name='C2v',
              rotations=[],
              translations=[],
              multiplicities=[1, 1, 1, 1]),

    CharTable({'A1': pd.Series([+1, +1, +1, +1, +1], index=['E', 'C3', 'C2', 'S4', 'sd']),
               'A2': pd.Series([+1, +1, +1, -1, -1], index=['E', 'C3', 'C2', 'S4', 'sd']),
               'E' : pd.Series([+2, -1, +2,  0,  0], index=['E', 'C3', 'C2', 'S4', 'sd']),
               'T1': pd.Series([+3,  0, -1, +1, -1], index=['E', 'C3', 'C2', 'S4', 'sd']),
               'T2': pd.Series([+3,  0, -1, -1, +1], index=['E', 'C3', 'C2', 'S4', 'sd']),
               },
              name='Td',
              rotations=[],
              translations=[],
              multiplicities=[1, 8, 3, 6, 6]),
]
