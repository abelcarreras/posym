

class InvalidRepresentation(Exception):
    """
    Raised when an IR label or representation does not match the point group.
    """
    def __init__(self, rep, pg):
        self._pg = pg
        self._rep = rep

    def __str__(self):
        if isinstance(self._rep, str):

            return 'Representation {} do not match with group. Available: {}'.format(self._rep,
                                                                                      self._pg.ir_labels)
        return 'Representation do not match with group {}'.format(self._pg.group)


class IncoherenceWarning(UserWarning):
    """
    Raised when the symmetrized structure shows permutation
    inconsistency, indicating the CSM may be unreliable.
    """
    def __init__(self, message):
        super().__init__(message)
