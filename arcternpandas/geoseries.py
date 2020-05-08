from arcternpandas.base import GeoBase
from pandas import Series
from arcternpandas.geoarry import GeoArry, GeoDtype
import arctern


def is_geometry_arry(data):
    """
    Check if the data is of bytes dtype.
    """
    if isinstance(getattr(data, "dtype", None), GeoDtype):
        # GeoArray, GeoSeries and Series[GeoArray]
        return True
    else:
        return False


def _property_op(f, this, *args, **kwargs):
    # type: (function, GeoSeries, args/kwargs) -> Series[bool/float]
    return Series(f(this.values).values, index=this.index)


class GeoSeries(Series):
    _metadata = ["name"]

    def __init__(self, data=None, index=None, dtype=None, name=None, crs=None, **kwargs):
        if hasattr(data, "crs") and crs:
            if not data.crs:
                data = data.copy()
            else:
                raise ValueError(
                    "CRS mismatch between CRS of the passed geometries and crs."
                )
        if isinstance(data, bytes):
            n = len(index) if index is not None else 1
            data = [data] * n

        if not is_geometry_arry(data):
            s = Series(data, index=index, name=name, **kwargs)
            if not s.dtype == bytes:
                if s.empty:
                    s = s.astype(bytes)
                else:
                    raise TypeError("Can not use no bytes data to construct GeoSeries.")
            data = GeoArry(s.values)
        super().__init__(data, index=index, dtype=dtype, name=name, **kwargs)

    @property
    def isna(self):
        return super().isna()

    # -------------------------------------------------------------------------
    # Geometry related property
    # -------------------------------------------------------------------------

    @property
    def is_valid(self):
        r = _property_op(arctern.ST_IsValid, self)
        return r

    @property
    def length(self):
        return _property_op(arctern.ST_Length, self)

    @property
    def is_simple(self):
        return _property_op(arctern.ST_IsSimple, self)

    @property
    def area(self):
        return _property_op(arctern.ST_Area, self)

    @property
    def geometry_type(self):
        return _property_op(arctern.ST_GeometryType)

    @property
    def centroid(self):
        return _property_op(arctern.ST_Centroid, self)

    @property
    def convex_hull(self):
        return _property_op(arctern.ST_ConvexHull, self)
