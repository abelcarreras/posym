

class InvalidRepresentation(Exception):
    def __init__(self, rep, pg):
        self._pg = pg
        self._rep = rep

    def __str__(self):
        if isinstance(self._rep, str):

            return 'Representation {} do not match with group. Available: {}'.format(self._rep,
                                                                                     self._pg.ir_labels)
        return 'Representation do not match with group {}'.format(self._pg.group)


class IncoherenceWarning(UserWarning):
    def __init__(self, message):
        super().__init__(message)
