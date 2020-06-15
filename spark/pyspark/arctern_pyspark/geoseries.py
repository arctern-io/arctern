import arctern
import databricks.koalas as ks
import pandas as pd
from databricks.koalas import DataFrame, get_option
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.internal import _InternalFrame
from databricks.koalas.series import REPR_PATTERN
from pandas.io.formats.printing import pprint_thing

# os.environ['PYSPARK_PYTHON'] = "/home/shengjh/miniconda3/envs/koalas/bin/python"
# os.environ['PYSPARK_DRIVER_PYTHON'] = "/home/shengjh/miniconda3/envs/koalas/bin/python"

ks.set_option('compute.ops_on_diff_frames', True)


# for unary operation, which return a koalas series.
def _unary_op(kss, f):
    from arctern_pyspark import _wrapper_func
    _kdf = kss.to_dataframe()
    ret_col = getattr(_wrapper_func, f)(kss.spark_column)
    return ks.Series(
        _kdf._internal.copy(
            spark_column=ret_col, column_labels=kss._internal.column_labels
        ),
        anchor=_kdf
    )


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
        return _unary_op(self, "ST_Area")


if __name__ == '__main__':
    rows = 1000000
    data_series = [
                      'POLYGON ((1 1,1 2,2 2,2 1,1 1))',
                      'POLYGON ((0 0,0 4,2 2,4 4,4 0,0 0))',
                      'POLYGON ((0 0,0 4,4 4,0 0))',
                  ] * rows

    countries = ['London', 'New York', 'Helsinki'] * rows
    s = ks.Series(data_series, name="haha", index=countries)
    s1 = ks.Series(data_series, name="haha", index=countries)

    s2 = GeoSeries(data_series, name="haha", index=countries)

    # print(s)
    # print(s2)
    print(s2.area)
    # print(s1+ s)
