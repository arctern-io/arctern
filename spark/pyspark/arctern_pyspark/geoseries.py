import arctern

import databricks.koalas as ks
import pandas as pd
from pyspark.sql import functions as F
from databricks.koalas import DataFrame, get_option
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.internal import _InternalFrame
from databricks.koalas.series import REPR_PATTERN
from pandas.io.formats.printing import pprint_thing

# os.environ['PYSPARK_PYTHON'] = "/home/shengjh/miniconda3/envs/koalas/bin/python"
# os.environ['PYSPARK_DRIVER_PYTHON'] = "/home/shengjh/miniconda3/envs/koalas/bin/python"

ks.set_option('compute.ops_on_diff_frames', True)


# for unary or binary operation, which return koalas Series.
def _column_op(f, *args):
    from arctern_pyspark import _wrapper_func
    return ks.base._column_op(getattr(_wrapper_func, f))(*args)


# for unary or binary operation, which return GeoSeries.
def _column_geo(f, *args):
    from arctern_pyspark import _wrapper_func
    kss = ks.base._column_op(getattr(_wrapper_func, f))(*args)
    return GeoSeries(kss._internal, anchor=kss._kdf)


class GeoSeries(ks.Series):
    def __init__(
            self, data=None, index=None, dtype=None, name=None, copy=False, fastpath=False, anchor=None
    ):
        if isinstance(data, _InternalFrame):
            assert dtype is None
            assert name is None
            assert not copy
            assert not fastpath
            IndexOpsMixin.__init__(self, data, anchor)
        else:
            assert anchor is None
            if isinstance(data, arctern.GeoSeries):
                assert index is None
                assert dtype is None
                assert name is None
                assert not copy
                assert not fastpath
                pds = data.astype(object, copy=False)
            else:
                pds = arctern.GeoSeries(
                    data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath
                )
                pds = pds.astype(object, copy=False)
            kdf = DataFrame(pds)
            IndexOpsMixin.__init__(
                self, kdf._internal.copy(spark_column=kdf._internal.data_spark_columns[0]), kdf
            )

    def __repr__(self):
        max_display_count = get_option("display.max_rows")
        if max_display_count is None:
            pser = pd.Series(arctern.ST_AsText(self._to_internal_pandas()).to_numpy(),
                             name=self.name,
                             index=self.index.to_numpy(),
                             copy=False
                             )
            return pser.to_string(name=self.name, dtype=self.dtype)

        pser = pd.Series(arctern.ST_AsText(self.head(max_display_count + 1)._to_internal_pandas()).to_numpy(),
                         name=self.name,
                         index=self.index.to_numpy(),
                         copy=False
                         )
        pser_length = len(pser)
        pser = pser.iloc[:max_display_count]
        if pser_length > max_display_count:
            repr_string = pser.to_string(length=True)
            rest, prev_footer = repr_string.rsplit("\n", 1)
            match = REPR_PATTERN.search(prev_footer)
            if match is not None:
                length = match.group("length")
                name = str(self.dtype.name)
                footer = "\nName: {name}, dtype: {dtype}\nShowing only the first {length}".format(
                    length=length, name=self.name, dtype=pprint_thing(name)
                )
                return rest + footer
        return pser.to_string(name=self.name, dtype=self.dtype)

    @property
    def area(self):
        return _column_op("ST_Area", self)

    def geom_equals(self, other):
        return _column_op("ST_Equals", self, other)

    def buffer(self, buffer=1.2):
        return _column_geo("ST_Buffer", self, F.lit(buffer))

    def to_crs(self, crs):
        return _column_geo("ST_Transform", crs)

    def intersects(self, other):
        return _column_op("ST_Intersects", self, other)

    def within(self, other):
        return _column_op("ST_Within", self, other)

    def contains(self, other):
        return _column_op("ST_Contains", self, other)

    def crosses(self, other):
        return _column_op("ST_Crosses", self, other)

    def touches(self, other):
        return _column_op("ST_Touches", self, other)

    def overlaps(self, other):
        return _column_op("ST_Overlaps", self, other)

    def distance(self, other):
        return _column_op("ST_Distance", self, other)

    def distance_sphere(self, other):
        return _column_op("ST_DistanceSphere", self, other)

    def hausdorff_distance(self, other):
        return _column_op("ST_HausdorffDistance", self, other)

    def intersection(self, other):
        return _column_geo("ST_Intersection", self, other)

    def polygon_from_envelope(self, min_x, min_y, max_x, max_y, crs=None):
        return _column_geo("ST_PolygonFromEnvelope", min_x, min_y, max_x, max_y, crs)

    def point(self, x, y):
        return _column_geo("ST_Point", x, y)

    def geom_from_geojson(self, json):
        return _column_geo("ST_GeoFromGeoJson", json)


if __name__ == '__main__':
    rows = 2
    data_series = [
                      'POLYGON ((1 1,1 2,2 2,2 1,1 1))',
                      'POLYGON ((0 0,0 4,2 2,4 4,4 0,0 0))',
                      'POLYGON ((0 0,0 4,4 4,0 0))',
                  ] * rows

    data_series2 = [
                       'POLYGON ((1 1,1 2,3 2,2 1,1 1))',
                       'POLYGON ((0 0,0 4,3 2,4 4,4 0,0 0))',
                       'POLYGON ((0 0,0 4,3 4,0 0))',
                   ] * rows

    countries = ['London', 'New York', 'Helsinki'] * rows
    s = ks.Series(data_series, name="haha", index=countries)
    s1 = ks.Series(data_series, name="haha", index=countries)
    s3 = ks.Series(data_series2, name="hehe", index=countries)
    s4 = pd.Series(data_series, name="haha", index=countries)
    s5 = pd.Series(data_series, name="hehe", index=countries)

    s2 = GeoSeries(data_series, name="haha", index=countries)
    s6 = GeoSeries(data_series, name="hehe", index=countries)

    # print(s + 'a')
    # print(s2)
    # print(s2.area)
    # print(s1+s3)
    # print(s2.area)
    # print(s2.geom_equals(s6))
    print(s2.buffer(1.2))
