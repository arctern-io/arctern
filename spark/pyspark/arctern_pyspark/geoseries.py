import arctern
import databricks.koalas as ks
import pandas as pd
from databricks.koalas import DataFrame, get_option
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.internal import _InternalFrame
from databricks.koalas.series import REPR_PATTERN
from pandas.io.formats.printing import pprint_thing
from pyspark.sql import functions as F

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

    # -------------------------------------------------------------------------
    # Geometry related property
    # -------------------------------------------------------------------------

    @property
    def area(self):
        return _column_op("ST_Area", self)

    @property
    def is_valid(self):
        return _column_op("ST_IsValid", self)

    @property
    def length(self):
        return _column_op("ST_Length", self)

    @property
    def is_simple(self):
        return _column_op("ST_IsSimple", self)

    @property
    def geom_type(self):
        return _column_op("ST_GeometryType", self)

    @property
    def centroid(self):
        return _column_geo("ST_Centroid", self)

    @property
    def convex_hull(self):
        return _column_geo("ST_ConvexHull", self)

    @property
    def npoints(self):
        return _column_op("ST_NPoints", self)

    @property
    def envelope(self):
        return _column_geo("ST_Envelope", self)

    # -------------------------------------------------------------------------
    # Geometry related unary methods, which return GeoSeries
    # -------------------------------------------------------------------------

    def curve_to_line(self):
        return _column_geo("ST_CurveToLine", self)

    def simplify(self, tolerance):
        return _column_geo("ST_SimplifyPreserveTopology", self, tolerance)

    def buffer(self, buffer):
        return _column_geo("ST_Buffer", self, F.lit(buffer))

    def precision_reduce(self, precision):
        return _column_geo("ST_PrecisionReduce", self, F.lit(precision))

    def make_valid(self):
        return _column_geo("ST_MakeValid", self)

    def unary_union(self):
        return _column_geo("ST_Union_Aggr", self)

    def envelope_aggr(self):
        return _column_geo("ST_Envelope_Aggr", self)

    def geom_equals(self, other):
        return _column_op("ST_Equals", self, other)
