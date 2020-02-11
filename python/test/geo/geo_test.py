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

# def test_ST_PrecisionReduce():
#     data = pandas.Series(["POINT (1.333 2.666)","POINT (2.655 4.447)"])
#     array = pyarrow.array(data)
#     rst = zilliz_gis.ST_PrecisionReduce(array,3)
#     assert rst[0] == "POINT (1.333 2.666)"
#     assert rst[1] == "POINT (2.655 4.447)"

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

def test_ST_Contains():
    p11 = "POLYGON((0 0,1 0,1 1,0 1,0 0))"
    p12 = "POLYGON((8 0,9 0,9 1,8 1,8 0))"
    p13 = "POINT(2 2)"
    p14 = "POINT(200 2)"
    data1 = pandas.Series([p11,p12,p13,p14])

    p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p23 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p24 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    data2 = pandas.Series([p21,p22,p23,p24])
    
    rst = zilliz_gis.ST_Contains(pyarrow.array(data2),pyarrow.array(data1))
    assert rst[0] == 1
    assert rst[1] == 0
    assert rst[2] == 1
    assert rst[3] == 0

def test_ST_Intersects():
    p11 = "POLYGON((0 0,1 0,1 1,0 1,0 0))"
    p12 = "POLYGON((8 0,9 0,9 1,8 1,8 0))"
    p13 = "LINESTRING(2 2,10 2)"
    p14 = "LINESTRING(9 2,10 2)"
    data1 = pandas.Series([p11,p12,p13,p14])

    p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p23 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p24 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    data2 = pandas.Series([p21,p22,p23,p24])

    rst = zilliz_gis.ST_Intersects(pyarrow.array(data2),pyarrow.array(data1))
    assert rst[0]==1
    assert rst[1]==1
    assert rst[2]==1
    assert rst[3]==0

def test_ST_Within():
    p11 = "POLYGON((0 0,1 0,1 1,0 1,0 0))"
    p12 = "POLYGON((8 0,9 0,9 1,8 1,8 0))"
    p13 = "LINESTRING(2 2,3 2)"
    p14 = "POINT(10 2)"
    data1 = pandas.Series([p11,p12,p13,p14])

    p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p23 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p24 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    data2 = pandas.Series([p21,p22,p23,p24])

    rst = zilliz_gis.ST_Within(pyarrow.array(data2),pyarrow.array(data1))
    assert rst[0]==0
    assert rst[1]==0
    assert rst[2]==0
    assert rst[3]==0

def test_ST_Distance():
    p11 = "LINESTRING(9 0,9 2)"
    p12 = "POINT(10 2)"
    data1 = pandas.Series([p11,p12])

    p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    data2 = pandas.Series([p21,p22])

    rst = zilliz_gis.ST_Distance(pyarrow.array(data2),pyarrow.array(data1))

    assert rst[0]==1.0
    assert rst[1]==2.0

def test_ST_Area():
    data = ["POLYGON((0 0,1 0,1 1,0 1,0 0))","POLYGON((0 0,0 8,8 8,8 0,0 0))"]
    array = pyarrow.array(pandas.Series(data))
    rst = zilliz_gis.ST_Area(array)

    assert rst[0]==1.0
    assert rst[1]==64.0

def test_ST_Centroid():
    data = ["POLYGON((0 0,1 0,1 1,0 1,0 0))","POLYGON((0 0,0 8,8 8,8 0,0 0))"]
    array = pyarrow.array(pandas.Series(data))
    rst = zilliz_gis.ST_Centroid(array)

    assert rst[0]=="POINT (0.5 0.5)"
    assert rst[1]=="POINT (4 4)"

def test_ST_Length():
    data = ["LINESTRING(0 0,0 1)", "LINESTRING(1 1,1 4)"]
    array = pyarrow.array(pandas.Series(data))
    rst = zilliz_gis.ST_Length(array)

    assert rst[0]==1.0
    assert rst[1]==3.0

def test_ST_ConvexHull():
    data = ["POINT (1.1 101.1)"]
    array = pyarrow.array(pandas.Series(data))
    rst = zilliz_gis.ST_ConvexHull(array)

    assert rst[0] == "POINT (1.1 101.1)"

def test_ST_NPoints():
    data = ["LINESTRING(1 1,1 4)"]
    array = pyarrow.array(pandas.Series(data))
    rst = zilliz_gis.ST_NPoints(array)

    assert rst[0] == 2

def test_ST_Envelope():
    data = ["POLYGON((0 0,1 0,1 1,0 0))"]
    array = pyarrow.array(pandas.Series(data))
    rst = zilliz_gis.ST_Envelope(array)

    assert rst[0] == "LINESTRING (0 0,1 0,1 1,0 0)"

def test_ST_Buffer():
    data = ["POLYGON((0 0,1 0,1 1,0 0))"]
    array = pyarrow.array(pandas.Series(data))
    rst =  zilliz_gis.ST_Buffer(array,1.2)
    expect = "POLYGON ((-0.848528137423857 0.848528137423857,0.151471862576143 1.84852813742386,0.19704327236937 1.89177379057287,0.244815530740195 1.93257515374836,0.294657697249032 1.97082039324994,0.346433157981967 2.00640468153451,0.4 2.03923048454133,0.455211400312543 2.06920782902604,0.511916028309039 2.09625454917112,0.569958460545639 2.12029651179664,0.629179606750062 2.14126781955418,0.689417145876974 2.15911099154688,0.750505971018688 2.17377712088057,0.812278641951722 2.18522600871417,0.874565844078815 2.19342627444193,0.937196852508467 2.19835544170549,1.0 2.2,1.06280314749153 2.19835544170549,1.12543415592118 2.19342627444193,1.18772135804828 2.18522600871417,1.24949402898131 2.17377712088057,1.31058285412302 2.15911099154688,1.37082039324994 2.14126781955418,1.43004153945436 2.12029651179664,1.48808397169096 2.09625454917112,1.54478859968746 2.06920782902604,1.6 2.03923048454133,1.65356684201803 2.00640468153451,1.70534230275097 1.97082039324994,1.75518446925981 1.93257515374836,1.80295672763063 1.89177379057287,1.84852813742386 1.84852813742386,1.89177379057287 1.80295672763063,1.93257515374837 1.7551844692598,1.97082039324994 1.70534230275097,2.00640468153451 1.65356684201803,2.03923048454133 1.6,2.06920782902604 1.54478859968746,2.09625454917112 1.48808397169096,2.12029651179664 1.43004153945436,2.14126781955418 1.37082039324994,2.15911099154688 1.31058285412302,2.17377712088057 1.24949402898131,2.18522600871417 1.18772135804828,2.19342627444193 1.12543415592118,2.19835544170549 1.06280314749153,2.2 1.0,2.2 0.0,2.19835544170549 -0.062803147491532,2.19342627444193 -0.125434155921184,2.18522600871417 -0.187721358048277,2.17377712088057 -0.249494028981311,2.15911099154688 -0.310582854123025,2.14126781955418 -0.370820393249937,2.12029651179664 -0.43004153945436,2.09625454917112 -0.48808397169096,2.06920782902604 -0.544788599687456,2.03923048454133 -0.6,2.00640468153451 -0.653566842018033,1.97082039324994 -0.705342302750968,1.93257515374836 -0.755184469259805,1.89177379057287 -0.80295672763063,1.84852813742386 -0.848528137423857,1.80295672763063 -0.891773790572873,1.75518446925981 -0.932575153748365,1.70534230275097 -0.970820393249937,1.65356684201803 -1.00640468153451,1.6 -1.03923048454133,1.54478859968746 -1.06920782902604,1.48808397169096 -1.09625454917112,1.43004153945436 -1.12029651179664,1.37082039324994 -1.14126781955418,1.31058285412302 -1.15911099154688,1.24949402898131 -1.17377712088057,1.18772135804828 -1.18522600871417,1.12543415592118 -1.19342627444193,1.06280314749153 -1.19835544170549,1.0 -1.2,0.0 -1.2,-0.062803147491532 -1.19835544170549,-0.125434155921184 -1.19342627444193,-0.187721358048276 -1.18522600871417,-0.24949402898131 -1.17377712088057,-0.310582854123024 -1.15911099154688,-0.370820393249936 -1.14126781955418,-0.430041539454359 -1.12029651179664,-0.488083971690959 -1.09625454917112,-0.544788599687455 -1.06920782902604,-0.6 -1.03923048454133,-0.653566842018031 -1.00640468153451,-0.705342302750966 -0.970820393249938,-0.755184469259804 -0.932575153748366,-0.802956727630628 -0.891773790572875,-0.848528137423855 -0.848528137423859,-0.891773790572871 -0.802956727630632,-0.932575153748363 -0.755184469259807,-0.970820393249935 -0.70534230275097,-1.00640468153451 -0.653566842018035,-1.03923048454132 -0.6,-1.06920782902604 -0.544788599687459,-1.09625454917112 -0.488083971690964,-1.12029651179664 -0.430041539454364,-1.14126781955418 -0.370820393249941,-1.15911099154688 -0.310582854123029,-1.17377712088057 -0.249494028981315,-1.18522600871416 -0.187721358048281,-1.19342627444193 -0.125434155921189,-1.19835544170549 -0.062803147491537,-1.2 -0.0,-1.19835544170549 0.062803147491527,-1.19342627444193 0.125434155921179,-1.18522600871417 0.187721358048272,-1.17377712088057 0.249494028981306,-1.15911099154688 0.310582854123019,-1.14126781955419 0.370820393249931,-1.12029651179664 0.430041539454355,-1.09625454917112 0.488083971690954,-1.06920782902604 0.54478859968745,-1.03923048454133 0.6,-1.00640468153451 0.653566842018027,-0.970820393249941 0.705342302750962,-0.93257515374837 0.755184469259799,-0.891773790572878 0.802956727630624,-0.848528137423857 0.848528137423857))"

    assert rst[0]==expect

def test_ST_PolygonFromEnvelope():
    x_min = pyarrow.array(pandas.Series([0.0]))
    x_max = pyarrow.array(pandas.Series([1.0]))
    y_min = pyarrow.array(pandas.Series([0.0]))
    y_max = pyarrow.array(pandas.Series([1.0]))

    rst = zilliz_gis.ST_PolygonFromEnvelope(x_min,y_min,x_max,y_max)

    assert rst[0] == "POLYGON ((0 0,1 0,0 1,1 1,0 0))"