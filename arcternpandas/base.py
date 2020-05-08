class GeoBase(object):

    @property
    def crs(self):
        # todo:
        return self.crs

    @crs.setter
    def crs(self, value):
        self.crs = value

    # -------------------------------------------------------------------------
    # Geometry related methods
    # -------------------------------------------------------------------------

    @property
    def is_valid(self):
        """
        Determine each geometry is valid.
        :return: a ``Series`` of ``dtype('bool')`` with value ``True`` for
        geometries that are valid.
        """
        return
