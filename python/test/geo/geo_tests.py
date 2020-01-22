import pyarrow
import zilliz_gis_core
import pandas
import numpy

bool_arr = zilliz_gis_core.ST_IsValid(pyarrow.array(pandas.Series(["POINT (1.3 2.6)","POINT (2.6 4.7)"])))
assert bool_arr[0] == 1
assert bool_arr[1] == 1

geometries = zilliz_gis_core.ST_PrecisionReduce(pyarrow.array(pandas.Series(["POINT (1.333 2.666)","POINT (2.655 4.447)"])),3)

assert geometries[0] == "POINT (1.33 2.67)"
assert geometries[1] == "POINT (2.65 4.45)"

geometries = zilliz_gis_core.ST_Intersection(pyarrow.array(pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))"])),pyarrow.array(pandas.Series(["POLYGON ((2 1,3 1,3 2,2 2,2 1))"])))

assert geometries[0] == "LINESTRING (2 2,2 1)"

geometries = zilliz_gis_core.ST_Equals(pyarrow.array(pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((1 1,1 2,2 2,2 1,1 1))"])),pyarrow.array(pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((2 1,3 1,3 2,2 2,2 1))"])))

assert geometries[0] == 1
assert geometries[1] == 0

geometries = zilliz_gis_core.ST_Touches(pyarrow.array(pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((1 1,1 2,2 2,2 1,1 1))"])),pyarrow.array(pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((2 1,3 1,3 2,2 2,2 1))"])))

assert geometries[0] == 0
assert geometries[1] == 1

geometries = zilliz_gis_core.ST_Overlaps(pyarrow.array(pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((1 1,1 2,2 2,2 1,1 1))"])),pyarrow.array(pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((2 1,3 1,3 2,2 2,2 1))"])))

assert geometries[0] == 0
assert geometries[1] == 0

geometries = zilliz_gis_core.ST_Crosses(pyarrow.array(pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((1 1,1 2,2 2,2 1,1 1))"])),pyarrow.array(pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((2 1,3 1,3 2,2 2,2 1))"])))

assert geometries[0] == 0
assert geometries[1] == 0

geometries = zilliz_gis_core.ST_IsSimple(pyarrow.array(pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((1 1,1 2,2 2,2 1,1 1))"])))

assert geometries[0] == 1
assert geometries[1] == 1

geometries = zilliz_gis_core.ST_GeometryType(pyarrow.array(pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((1 1,1 2,2 2,2 1,1 1))"])))

assert geometries[0] == "POLYGON"
assert geometries[1] == "POLYGON"


geometries = zilliz_gis_core.ST_MakeValid(pyarrow.array(pandas.Series(["POLYGON ((2 1,3 1,3 2,2 2,2 8,2 1))"])))

geometries[0] == "GEOMETRYCOLLECTION (POLYGON ((2 2,3 2,3 1,2 1,2 2)),LINESTRING (2 2,2 8))"


geometries = zilliz_gis_core.ST_SimplifyPreserveTopology(pyarrow.array(pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((1 1,1 2,2 2,2 1,1 1))"])),10000)

geometries[0] == "POLYGON ((1 1,1 2,2 2,2 1,1 1))"