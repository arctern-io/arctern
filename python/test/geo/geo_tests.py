import pyarrow
import zilliz_gis
import pandas
import numpy

string_ptr=zilliz_gis.ST_Point(pyarrow.array(pandas.Series([1.3,2.5])),pyarrow.array(pandas.Series([3.8,4.9])))
assert len(string_ptr) == 2
assert string_ptr[0] == "POINT (1.3 3.8)"
assert string_ptr[1] == "POINT (2.5 4.9)"
