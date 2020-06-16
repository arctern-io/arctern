import arctern
import os

# os.environ['PYSPARK_PYTHON'] = "/home/shengjh/miniconda3/envs/koalas/bin/python"
# os.environ['PYSPARK_DRIVER_PYTHON'] = "/home/shengjh/miniconda3/envs/koalas/bin/python"

import databricks.koalas as ks
from databricks.koalas import DataFrame, get_option
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.internal import _InternalFrame
from databricks.koalas.series import REPR_PATTERN, _col
from pandas.io.formats.printing import pprint_thing
ks.set_option('compute.ops_on_diff_frames', True)
from arctern_pyspark import _wrapper_func
from pyspark.sql.functions import col, lit

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
                s = data.astype(object, copy=False)
            else:
                s = arctern.GeoSeries(
                    data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath
                )
                s = s.astype(object, copy=False)
            kdf = DataFrame(s)
            IndexOpsMixin.__init__(
                self, kdf._internal.copy(spark_column=kdf._internal.data_spark_columns[0]), kdf
            )

    def __repr__(self):
        max_display_count = get_option("display.max_rows")
        if max_display_count is None:
            return arctern.ST_AsText(self._to_internal_pandas()).to_string(name=self.name, dtype=self.dtype)

        pser = arctern.ST_AsText(self.head(max_display_count + 1)._to_internal_pandas())
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

        _kdf = self.to_dataframe()
        sdf = _kdf.to_spark()

        ret = sdf.select(_wrapper_func.ST_Area(col(self.name)).alias(self.name))  # spark dataframe

        kdf = ret.to_koalas()
        return kdf[self.name]

    def equals(self, other):
        _kdf = ks.DataFrame(data=self)
        _kdf["col2"] = other
        sdf = _kdf.to_spark()

        ret = sdf.select(_wrapper_func.ST_Equals(col("col1"), col("col2")).alias("xixi"))  # spark dataframe

        kdf = ret.to_koalas()

        from databricks.koalas.internal import _InternalFrame
        internal = _InternalFrame(
            kdf._internal.spark_frame,
            index_map=kdf._internal.index_map,
            column_labels=kdf._internal.column_labels,
            column_label_names=kdf._internal.column_label_names,
        )

        return ks.Series(internal, anchor=kdf)


rows = 1

data_series = [
                  'POLYGON ((1 1,1 2,2 2,2 1,1 1))',
                  'POLYGON ((0 0,0 4,2 2,4 4,4 0,0 0))',
                  'POLYGON ((0 0,0 4,4 4,0 0))',
              ] * rows

countries = ['London', 'New York', 'Helsinki'] * rows
s1 = GeoSeries(data_series, name="col1", index=countries)
s2 = GeoSeries(data_series, name="col2", index=countries)

ret1 = s1.area
ret2 = s1.equals(s2)

print(ret1)
print(ret2)
