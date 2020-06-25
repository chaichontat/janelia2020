class Analyzer:
    """
    Abstract class for data analysis from raw spike data in the form of `SpikeLoader` instance.
    Prevent `SpikeLoader` from being pickled along with the instance.

    All parameters required for a exact replication from an identical `SpikeLoader` instance
    and any data required for a proper functioning of helper functions should be pointed to by an instance variable.

    Any saved analysis should include proper context.

    """

    def __getstate__(self):
        d = self.__dict__.copy()
        try:
            del d['loader']
        except KeyError:
            pass
        return d
