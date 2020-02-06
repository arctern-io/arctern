import pyarrow
import zilliz_gis
import pandas
import numpy
import pytest

def test_ST_IsValid():
    data = pandas.Series(["POINT (1.3 2.6)","POINT (2.6 4.7)"])
    array = pyarrow.array(data)
    rst = zilliz_gis.ST_IsValid(array)
    assert rst[0]==1
    assert rst[1]==1

def test_ST_PrecisionReduce():
    data = pandas.Series(["POINT (1.333 2.666)","POINT (2.655 4.447)"])
    array = pyarrow.array(data)
    rst = zilliz_gis.ST_PrecisionReduce(array,3)
    assert rst[0] == "POINT (1.333 2.666)"
    assert rst[1] == "POINT (2.655 4.447)"

def test_ST_Intersection():
    data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    data2 = pandas.Series(["POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
    array1 = pyarrow.array(data1)
    array2 = pyarrow.array(data2)
    rst = zilliz_gis.ST_Intersection(array1,array2)
    assert rst[0] == "LINESTRING (2 2,2 1)"

def test_ST_Equals():
    data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    data2 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
    array1 = pyarrow.array(data1)
    array2 = pyarrow.array(data2)
    rst = zilliz_gis.ST_Equals(array1,array2)
    assert rst[0] == 1
    assert rst[1] == 0

def test_ST_Touches():
    data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    data2 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
    rst = zilliz_gis.ST_Touches(pyarrow.array(data1),pyarrow.array(data2))
    assert rst[0] == 0
    assert rst[1] == 1

def test_ST_Overlaps():
    data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    data2 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
    rst = zilliz_gis.ST_Overlaps(pyarrow.array(data1),pyarrow.array(data2))
    assert rst[0] == 0
    assert rst[1] == 0

def test_ST_Crosses():
    data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    data2 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
    rst = zilliz_gis.ST_Crosses(pyarrow.array(data1),pyarrow.array(data2))
    assert rst[0] == 0
    assert rst[1] == 0

def test_ST_IsSimple():
    data = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    rst = zilliz_gis.ST_IsSimple(pyarrow.array(data))
    assert rst[0] == 1
    assert rst[1] == 1

def test_ST_GeometryType():
    data = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    rst = zilliz_gis.ST_GeometryType(pyarrow.array(data))
    assert rst[0] == "POLYGON"
    assert rst[1] == "POLYGON"

def test_ST_MakeValid():
    data = pandas.Series(["POLYGON ((2 1,3 1,3 2,2 2,2 8,2 1))"])
    array = pyarrow.array(data)
    rst = zilliz_gis.ST_MakeValid(array)
    assert rst[0] == "GEOMETRYCOLLECTION (POLYGON ((2 2,3 2,3 1,2 1,2 2)),LINESTRING (2 2,2 8))"

def test_ST_SimplifyPreserveTopology():
    data = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    rst = zilliz_gis.ST_SimplifyPreserveTopology(pyarrow.array(data),10000)
    assert rst[0] == "POLYGON ((1 1,1 2,2 2,2 1,1 1))"

def test_ST_Point():
    data1 = pandas.Series([1.3,2.5])
    data2 = pandas.Series([3.8,4.9])
    string_ptr=zilliz_gis.ST_Point(pyarrow.array(data1),pyarrow.array(data2))
    assert len(string_ptr) == 2
    assert string_ptr[0] == "POINT (1.3 3.8)"
    assert string_ptr[1] == "POINT (2.5 4.9)"
