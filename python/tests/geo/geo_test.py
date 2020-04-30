# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas
from osgeo import ogr
import arctern


def test_ST_IsValid():
    data = pandas.Series(["POINT (1.3 2.6)", "POINT (2.6 4.7)"])
    rst = arctern.ST_IsValid(arctern.ST_GeomFromText(data))
    assert rst[0]
    assert rst[1]


def test_ST_PrecisionReduce():
    data = pandas.Series(["POINT (1.333 2.666)", "POINT (2.655 4.447)"])
    rst = arctern.ST_AsText(arctern.ST_PrecisionReduce(arctern.ST_GeomFromText(data), 3))
    assert rst[0] == "POINT (1.33 2.67)"
    assert rst[1] == "POINT (2.66 4.45)"


def test_ST_Intersection():
    data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POINT (0 1)"])
    data2 = pandas.Series(["POLYGON ((2 1,3 1,3 2,2 2,2 1))", "POINT (0 1)"])
    rst = arctern.ST_AsText(arctern.ST_Intersection(arctern.ST_GeomFromText(data1), arctern.ST_GeomFromText(data2)))
    assert len(rst) == 2
    assert rst[0] == "LINESTRING (2 2,2 1)"
    assert rst[1] == "POINT (0 1)"

    rst = arctern.ST_AsText(
        arctern.ST_Intersection(arctern.ST_GeomFromText(data1), arctern.ST_GeomFromText("POINT (0 1)")[0]))
    assert len(rst) == 2
    assert rst[0] == "GEOMETRYCOLLECTION EMPTY"
    assert rst[1] == "POINT (0 1)"

    rst = arctern.ST_AsText(
        arctern.ST_Intersection(arctern.ST_GeomFromText("POINT (0 1)")[0], arctern.ST_GeomFromText(data1)))
    assert len(rst) == 2
    assert rst[0] == "GEOMETRYCOLLECTION EMPTY"
    assert rst[1] == "POINT (0 1)"

    rst = arctern.ST_AsText(
        arctern.ST_Intersection(arctern.ST_GeomFromText("POINT (0 1)")[0], arctern.ST_GeomFromText("POINT (0 1)")[0]))
    assert len(rst) == 1
    assert rst[0] == "POINT (0 1)"


def test_ST_Equals():
    data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    data2 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
    rst = arctern.ST_Equals(arctern.ST_GeomFromText(data1), arctern.ST_GeomFromText(data2))
    assert len(rst) == 2
    assert rst[0] == 1
    assert rst[1] == 0

    rst = arctern.ST_Equals(arctern.ST_GeomFromText(data2),
                            arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0])
    assert len(rst) == 2
    assert rst[0] == 1
    assert rst[1] == 0

    rst = arctern.ST_Equals(arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0],
                            arctern.ST_GeomFromText(data2))
    assert len(rst) == 2
    assert rst[0] == 1
    assert rst[1] == 0

    rst = arctern.ST_Equals(arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0],
                            arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0])
    assert len(rst) == 1
    assert rst[0] == 1


def test_ST_Touches():
    data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    data2 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
    rst = arctern.ST_Touches(arctern.ST_GeomFromText(data1), arctern.ST_GeomFromText(data2))
    assert len(rst) == 2
    assert rst[0] == 0
    assert rst[1] == 1

    rst = arctern.ST_Touches(arctern.ST_GeomFromText(data2),
                             arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0])
    assert len(rst) == 2
    assert rst[0] == 0
    assert rst[1] == 1

    rst = arctern.ST_Touches(arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0],
                             arctern.ST_GeomFromText(data2))
    assert len(rst) == 2
    assert rst[0] == 0
    assert rst[1] == 1

    rst = arctern.ST_Touches(arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0],
                             arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0])
    assert len(rst) == 1
    assert rst[0] == 0


def test_ST_Overlaps():
    data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    data2 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
    rst = arctern.ST_Overlaps(arctern.ST_GeomFromText(data1), arctern.ST_GeomFromText(data2))
    assert len(rst) == 2
    assert rst[0] == 0
    assert rst[1] == 0

    rst = arctern.ST_Overlaps(arctern.ST_GeomFromText(data2),
                              arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0])
    assert len(rst) == 2
    assert rst[0] == 0
    assert rst[1] == 0

    rst = arctern.ST_Overlaps(arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0],
                              arctern.ST_GeomFromText(data2))
    assert len(rst) == 2
    assert rst[0] == 0
    assert rst[1] == 0

    rst = arctern.ST_Overlaps(arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0],
                              arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0])
    assert len(rst) == 1
    assert rst[0] == 0


def test_ST_Crosses():
    data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    data2 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
    rst = arctern.ST_Crosses(arctern.ST_GeomFromText(data1), arctern.ST_GeomFromText(data2))
    assert len(rst) == 2
    assert rst[0] == 0
    assert rst[1] == 0

    rst = arctern.ST_Crosses(arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0],
                             arctern.ST_GeomFromText(data2))
    assert len(rst) == 2
    assert rst[0] == 0
    assert rst[1] == 0

    rst = arctern.ST_Crosses(arctern.ST_GeomFromText(data2),
                             arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0])
    assert len(rst) == 2
    assert rst[0] == 0
    assert rst[1] == 0

    rst = arctern.ST_Crosses(arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0],
                             arctern.ST_GeomFromText("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0])
    assert len(rst) == 1
    assert rst[0] == 0


def test_ST_IsSimple():
    data = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    rst = arctern.ST_IsSimple(arctern.ST_GeomFromText(data))
    assert rst[0] == 1
    assert rst[1] == 1


def test_ST_GeometryType():
    data = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    rst = arctern.ST_GeometryType(arctern.ST_GeomFromText(data))
    assert rst[0] == "ST_POLYGON"
    assert rst[1] == "ST_POLYGON"


def test_ST_MakeValid():
    data = pandas.Series(["POLYGON ((2 1,3 1,3 2,2 2,2 8,2 1))"])
    rst = arctern.ST_AsText(arctern.ST_MakeValid(arctern.ST_GeomFromText(data)))
    assert rst[0] == "GEOMETRYCOLLECTION (POLYGON ((2 2,3 2,3 1,2 1,2 2)),LINESTRING (2 2,2 8))"


def test_ST_SimplifyPreserveTopology():
    data = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    rst = arctern.ST_AsText(arctern.ST_SimplifyPreserveTopology(arctern.ST_GeomFromText(data), 10000))
    assert rst[0] == "POLYGON ((1 1,1 2,2 2,2 1,1 1))"


def test_ST_Point():
    data1 = pandas.Series([1.3, 2.5])
    data2 = pandas.Series([3.8, 4.9])
    string_ptr = arctern.ST_AsText(arctern.ST_Point(data1, data2))
    assert len(string_ptr) == 2
    assert string_ptr[0] == "POINT (1.3 3.8)"
    assert string_ptr[1] == "POINT (2.5 4.9)"

    string_ptr = arctern.ST_AsText(arctern.ST_Point(pandas.Series([1, 2], dtype='double'), 5))
    assert len(string_ptr) == 2
    assert string_ptr[0] == "POINT (1 5)"
    assert string_ptr[1] == "POINT (2 5)"

    string_ptr = arctern.ST_AsText(arctern.ST_Point(5, pandas.Series([1, 2], dtype='double')))
    assert len(string_ptr) == 2
    assert string_ptr[0] == "POINT (5 1)"
    assert string_ptr[1] == "POINT (5 2)"

    string_ptr = arctern.ST_AsText(arctern.ST_Point(5.0, 1.0))
    assert len(string_ptr) == 1
    assert string_ptr[0] == "POINT (5 1)"


def test_ST_GeomFromGeoJSON():
    j0 = "{\"type\":\"Point\",\"coordinates\":[1,2]}"
    j1 = "{\"type\":\"LineString\",\"coordinates\":[[1,2],[4,5],[7,8]]}"
    j2 = "{\"type\":\"Polygon\",\"coordinates\":[[[0,0],[0,1],[1,1],[1,0],[0,0]]]}"
    data = pandas.Series([j0, j1, j2])
    str_ptr = arctern.ST_AsText(arctern.ST_GeomFromGeoJSON(data))
    assert str_ptr[0] == "POINT (1 2)"
    assert str_ptr[1] == "LINESTRING (1 2,4 5,7 8)"
    assert str_ptr[2] == "POLYGON ((0 0,0 1,1 1,1 0,0 0))"


def test_ST_AsGeoJSON():
    j0 = "{\"type\":\"Point\",\"coordinates\":[1,2]}"
    j1 = "{\"type\":\"LineString\",\"coordinates\":[[1,2],[4,5],[7,8]]}"
    j2 = "{\"type\":\"Polygon\",\"coordinates\":[[[0,0],[0,1],[1,1],[1,0],[0,0]]]}"
    data = pandas.Series([j0, j1, j2])
    str_ptr = arctern.ST_AsGeoJSON(arctern.ST_GeomFromGeoJSON(data))
    assert str_ptr[0] == '{ "type": "Point", "coordinates": [ 1.0, 2.0 ] }'
    assert str_ptr[1] == '{ "type": "LineString", "coordinates": [ [ 1.0, 2.0 ], [ 4.0, 5.0 ], [ 7.0, 8.0 ] ] }'
    assert str_ptr[
               2] == '{ "type": "Polygon", "coordinates": [ [ [ 0.0, 0.0 ], [ 0.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ] ] ] }'


def test_ST_GeomFromText():
    p0 = "POINT (1 2)"
    p1 = "LINESTRING (1 2,4 5,7 8)"
    p2 = "POLYGON ((0 0,0 1,1 1,1 0,0 0))"
    data = pandas.Series([p0, p1, p2])
    str_ptr = arctern.ST_AsText(arctern.ST_GeomFromText(data))
    assert str_ptr[0] == "POINT (1 2)"
    assert str_ptr[1] == "LINESTRING (1 2,4 5,7 8)"
    assert str_ptr[2] == "POLYGON ((0 0,0 1,1 1,1 0,0 0))"

    wkb_array = arctern.ST_GeomFromText('POINT (1 2)')
    wkt_array = arctern.ST_AsText(wkb_array[0])
    assert wkt_array[0] == "POINT (1 2)"


def test_ST_Contains():
    p11 = "POLYGON((0 0,1 0,1 1,0 1,0 0))"
    p12 = "POLYGON((8 0,9 0,9 1,8 1,8 0))"
    p13 = "POINT(2 2)"
    p14 = "POINT(200 2)"
    data1 = pandas.Series([p11, p12, p13, p14])

    p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p23 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p24 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    data2 = pandas.Series([p21, p22, p23, p24])
    rst = arctern.ST_Contains(arctern.ST_GeomFromText(data2), arctern.ST_GeomFromText(data1))
    assert len(rst) == 4
    assert rst[0] == 1
    assert rst[1] == 0
    assert rst[2] == 1
    assert rst[3] == 0

    rst = arctern.ST_Contains(arctern.ST_GeomFromText(data2),
                              arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0])
    assert len(rst) == 4
    assert rst[0] == 1
    assert rst[1] == 1
    assert rst[2] == 1
    assert rst[3] == 1

    rst = arctern.ST_Contains(arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0],
                              arctern.ST_GeomFromText(data1))
    assert len(rst) == 4
    assert rst[0] == 1
    assert rst[1] == 0
    assert rst[2] == 1
    assert rst[3] == 0

    rst = arctern.ST_Contains(arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0],
                              arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0])
    assert len(rst) == 1
    assert rst[0] == 1


def test_ST_Intersects():
    p11 = "POLYGON((0 0,1 0,1 1,0 1,0 0))"
    p12 = "POLYGON((8 0,9 0,9 1,8 1,8 0))"
    p13 = "LINESTRING(2 2,10 2)"
    p14 = "LINESTRING(9 2,10 2)"
    data1 = pandas.Series([p11, p12, p13, p14])

    p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p23 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p24 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    data2 = pandas.Series([p21, p22, p23, p24])

    rst = arctern.ST_Intersects(arctern.ST_GeomFromText(data2), arctern.ST_GeomFromText(data1))
    assert rst[0] == 1
    assert rst[1] == 1
    assert rst[2] == 1
    assert rst[3] == 0

    rst = arctern.ST_Intersects(arctern.ST_GeomFromText(data1),
                                arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0])
    assert len(rst) == 4
    assert rst[0] == 1
    assert rst[1] == 1
    assert rst[2] == 1
    assert rst[3] == 0

    rst = arctern.ST_Intersects(arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0],
                                arctern.ST_GeomFromText(data1))
    assert len(rst) == 4
    assert rst[0] == 1
    assert rst[1] == 1
    assert rst[2] == 1
    assert rst[3] == 0

    rst = arctern.ST_Contains(arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0],
                              arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0])
    assert len(rst) == 1
    assert rst[0] == 1


def test_ST_Within():
    p11 = "POLYGON((0 0,1 0,1 1,0 1,0 0))"
    p12 = "POLYGON((8 0,9 0,9 1,8 1,8 0))"
    p13 = "LINESTRING(2 2,3 2)"
    p14 = "POINT(10 2)"
    data1 = pandas.Series([p11, p12, p13, p14])

    p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p23 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p24 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    data2 = pandas.Series([p21, p22, p23, p24])

    rst = arctern.ST_Within(arctern.ST_GeomFromText(data2), arctern.ST_GeomFromText(data1))
    assert len(rst) == 4
    assert rst[0] == 0
    assert rst[1] == 0
    assert rst[2] == 0
    assert rst[3] == 0

    rst = arctern.ST_Within(arctern.ST_GeomFromText(data1),
                            arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0])
    assert len(rst) == 4
    assert rst[0] == 1
    assert rst[1] == 0
    assert rst[2] == 1
    assert rst[3] == 0

    rst = arctern.ST_Within(arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0],
                            arctern.ST_GeomFromText(data1))
    assert len(rst) == 4
    assert rst[0] == 0
    assert rst[1] == 0
    assert rst[2] == 0
    assert rst[3] == 0

    rst = arctern.ST_Within(arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0],
                            arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0])
    assert len(rst) == 1
    assert rst[0] == 1


def test_ST_Distance():
    p11 = "LINESTRING(9 0,9 2)"
    p12 = "POINT(10 2)"
    data1 = pandas.Series([p11, p12])

    p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    data2 = pandas.Series([p21, p22])

    rst = arctern.ST_Distance(arctern.ST_GeomFromText(data2), arctern.ST_GeomFromText(data1))
    assert len(rst) == 2
    assert rst[0] == 1.0
    assert rst[1] == 2.0

    rst = arctern.ST_Distance(arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0],
                              arctern.ST_GeomFromText(data1))
    assert len(rst) == 2
    assert rst[0] == 1.0
    assert rst[1] == 2.0

    rst = arctern.ST_Distance(arctern.ST_GeomFromText(data1),
                              arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0])
    assert len(rst) == 2
    assert rst[0] == 1.0
    assert rst[1] == 2.0

    rst = arctern.ST_Distance(arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0],
                              arctern.ST_GeomFromText("POLYGON((0 0,0 8,8 8,8 0,0 0))")[0])
    assert len(rst) == 1
    assert rst[0] == 0.0


def test_ST_DistanceSphere():
    import math
    p11 = "POINT(-73.981153 40.741841)"
    p12 = "POINT(200 10)"
    data1 = pandas.Series([p11, p12])

    p21 = "POINT(-73.99016751859183 40.729884354626904)"
    p22 = "POINT(10 2)"
    data2 = pandas.Series([p21, p22])

    rst = arctern.ST_DistanceSphere(arctern.ST_GeomFromText(data2), arctern.ST_GeomFromText(data1))
    assert len(rst) == 2
    assert abs(rst[0] - 1531) < 1
    assert math.isnan(rst[1])

    data = pandas.Series(["POINT(0 0)"])
    rst = arctern.ST_DistanceSphere(arctern.ST_GeomFromText(data), arctern.ST_GeomFromText("POINT(0 0)")[0])
    assert len(rst) == 1
    assert math.isclose(rst[0], 0.0, rel_tol=1e-5)

    data = pandas.Series(["POINT(0 0)"])
    rst = arctern.ST_DistanceSphere(arctern.ST_GeomFromText("POINT(0 0)")[0], arctern.ST_GeomFromText(data))
    assert len(rst) == 1
    assert math.isclose(rst[0], 0.0, rel_tol=1e-5)

    data = pandas.Series(["POINT(0 0)"])
    rst = arctern.ST_DistanceSphere(arctern.ST_GeomFromText("POINT(0 0)")[0], arctern.ST_GeomFromText("POINT(0 0)")[0])
    assert len(rst) == 1
    assert math.isclose(rst[0], 0.0, rel_tol=1e-5)


def test_ST_Area():
    data = ["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"]
    data = pandas.Series(data)
    rst = arctern.ST_Area(arctern.ST_GeomFromText(data))

    assert rst[0] == 1.0
    assert rst[1] == 64.0


def test_ST_Centroid():
    data = ["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"]
    data = pandas.Series(data)
    rst = arctern.ST_AsText(arctern.ST_Centroid(arctern.ST_GeomFromText(data)))

    assert rst[0] == "POINT (0.5 0.5)"
    assert rst[1] == "POINT (4 4)"


def test_ST_Length():
    data = ["LINESTRING(0 0,0 1)", "LINESTRING(1 1,1 4)"]
    data = pandas.Series(data)
    rst = arctern.ST_Length(arctern.ST_GeomFromText(data))

    assert rst[0] == 1.0
    assert rst[1] == 3.0


def test_ST_HausdorffDistance():
    import math
    data1 = ["POLYGON((0 0 ,0 1, 1 1, 1 0, 0 0))", "POINT(0 0)"]
    data2 = ["POLYGON((0 0 ,0 2, 1 1, 1 0, 0 0))", "POINT(0 1)"]
    data1 = pandas.Series(data1)
    data2 = pandas.Series(data2)
    rst = arctern.ST_HausdorffDistance(arctern.ST_GeomFromText(data1), arctern.ST_GeomFromText(data2))
    assert len(rst) == 2
    assert rst[0] == 1
    assert rst[1] == 1

    rst = arctern.ST_HausdorffDistance(arctern.ST_GeomFromText(data1), arctern.ST_GeomFromText("POINT(0 0)")[0])
    assert len(rst) == 2
    assert math.isclose(rst[0], math.sqrt(2), rel_tol=1e-5)
    assert rst[1] == 0

    rst = arctern.ST_HausdorffDistance(arctern.ST_GeomFromText("POINT(0 0)")[0], arctern.ST_GeomFromText(data2))
    assert len(rst) == 2
    assert math.isclose(rst[0], 2, rel_tol=1e-5)
    assert rst[1] == 1.0

    rst = arctern.ST_HausdorffDistance(arctern.ST_GeomFromText("POINT(0 0)")[0],
                                       arctern.ST_GeomFromText("POINT(0 0)")[0])
    assert len(rst) == 1
    assert rst[0] == 0


def test_ST_ConvexHull():
    data = ["POINT (1.1 101.1)"]
    data = pandas.Series(data)
    rst = arctern.ST_AsText(arctern.ST_ConvexHull(arctern.ST_GeomFromText(data)))

    assert rst[0] == "POINT (1.1 101.1)"


def test_ST_Transform():
    data = ["POINT (10 10)"]
    data = pandas.Series(data)
    rst = arctern.ST_AsText(arctern.ST_Transform(arctern.ST_GeomFromText(data), "EPSG:4326", "EPSG:3857"))

    wkt = rst[0]
    rst_point = ogr.CreateGeometryFromWkt(str(wkt))
    assert abs(rst_point.GetX() - 1113194.90793274 < 0.01)
    assert abs(rst_point.GetY() - 1118889.97485796 < 0.01)


def test_ST_CurveToLine():
    data = ["CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0))"]
    data = pandas.Series(data)
    rst = arctern.ST_AsText(arctern.ST_CurveToLine(arctern.ST_GeomFromText(data)))

    assert str(rst[0]).startswith("POLYGON")


def test_ST_NPoints():
    data = ["LINESTRING(1 1,1 4)"]
    data = pandas.Series(data)
    rst = arctern.ST_NPoints(arctern.ST_GeomFromText(data))

    assert rst[0] == 2


def test_ST_Envelope():
    p1 = "point (10 10)"
    p2 = "linestring (0 0 , 0 10)"
    p3 = "linestring (0 0 , 10 0)"
    p4 = "linestring (0 0 , 10 10)"
    p5 = "polygon ((0 0, 10 0, 10 10, 0 10, 0 0))"
    p6 = "multipoint (0 0, 10 0, 5 5)"
    p7 = "multilinestring ((0 0, 5 5), (6 6, 6 7, 10 10))"
    p8 = "multipolygon (((0 0, 10 0, 10 10, 0 10, 0 0), (11 11, 20 11, 20 20, 20 11, 11 11)))"
    data = [p1, p2, p3, p4, p5, p6, p7, p8]
    data = pandas.Series(data)
    rst = arctern.ST_AsText(arctern.ST_Envelope(arctern.ST_GeomFromText(data)))

    assert rst[0] == "POINT (10 10)"
    assert rst[1] == "LINESTRING (0 0,0 10)"
    assert rst[2] == "LINESTRING (0 0,10 0)"
    assert rst[3] == "POLYGON ((0 0,0 10,10 10,10 0,0 0))"
    assert rst[4] == "POLYGON ((0 0,0 10,10 10,10 0,0 0))"
    assert rst[5] == "POLYGON ((0 0,0 5,10 5,10 0,0 0))"
    assert rst[6] == "POLYGON ((0 0,0 10,10 10,10 0,0 0))"
    assert rst[7] == "POLYGON ((0 0,0 20,20 20,20 0,0 0))"


def test_ST_Buffer():
    data = ["POLYGON((0 0,1 0,1 1,0 0))"]
    data = pandas.Series(data)
    rst = arctern.ST_AsText(arctern.ST_Buffer(arctern.ST_GeomFromText(data), 1.2))
    expect = "POLYGON ((-0.848528137423857 0.848528137423857,0.151471862576143 1.84852813742386,0.19704327236937 1.89177379057287,0.244815530740195 1.93257515374836,0.294657697249032 1.97082039324994,0.346433157981967 2.00640468153451,0.4 2.03923048454133,0.455211400312543 2.06920782902604,0.511916028309039 2.09625454917112,0.569958460545639 2.12029651179664,0.629179606750062 2.14126781955418,0.689417145876974 2.15911099154688,0.750505971018688 2.17377712088057,0.812278641951722 2.18522600871417,0.874565844078815 2.19342627444193,0.937196852508467 2.19835544170549,1.0 2.2,1.06280314749153 2.19835544170549,1.12543415592118 2.19342627444193,1.18772135804828 2.18522600871417,1.24949402898131 2.17377712088057,1.31058285412302 2.15911099154688,1.37082039324994 2.14126781955418,1.43004153945436 2.12029651179664,1.48808397169096 2.09625454917112,1.54478859968746 2.06920782902604,1.6 2.03923048454133,1.65356684201803 2.00640468153451,1.70534230275097 1.97082039324994,1.75518446925981 1.93257515374836,1.80295672763063 1.89177379057287,1.84852813742386 1.84852813742386,1.89177379057287 1.80295672763063,1.93257515374837 1.7551844692598,1.97082039324994 1.70534230275097,2.00640468153451 1.65356684201803,2.03923048454133 1.6,2.06920782902604 1.54478859968746,2.09625454917112 1.48808397169096,2.12029651179664 1.43004153945436,2.14126781955418 1.37082039324994,2.15911099154688 1.31058285412302,2.17377712088057 1.24949402898131,2.18522600871417 1.18772135804828,2.19342627444193 1.12543415592118,2.19835544170549 1.06280314749153,2.2 1.0,2.2 0.0,2.19835544170549 -0.062803147491532,2.19342627444193 -0.125434155921184,2.18522600871417 -0.187721358048277,2.17377712088057 -0.249494028981311,2.15911099154688 -0.310582854123025,2.14126781955418 -0.370820393249937,2.12029651179664 -0.43004153945436,2.09625454917112 -0.48808397169096,2.06920782902604 -0.544788599687456,2.03923048454133 -0.6,2.00640468153451 -0.653566842018033,1.97082039324994 -0.705342302750968,1.93257515374836 -0.755184469259805,1.89177379057287 -0.80295672763063,1.84852813742386 -0.848528137423857,1.80295672763063 -0.891773790572873,1.75518446925981 -0.932575153748365,1.70534230275097 -0.970820393249937,1.65356684201803 -1.00640468153451,1.6 -1.03923048454133,1.54478859968746 -1.06920782902604,1.48808397169096 -1.09625454917112,1.43004153945436 -1.12029651179664,1.37082039324994 -1.14126781955418,1.31058285412302 -1.15911099154688,1.24949402898131 -1.17377712088057,1.18772135804828 -1.18522600871417,1.12543415592118 -1.19342627444193,1.06280314749153 -1.19835544170549,1.0 -1.2,0.0 -1.2,-0.062803147491532 -1.19835544170549,-0.125434155921184 -1.19342627444193,-0.187721358048276 -1.18522600871417,-0.24949402898131 -1.17377712088057,-0.310582854123024 -1.15911099154688,-0.370820393249936 -1.14126781955418,-0.430041539454359 -1.12029651179664,-0.488083971690959 -1.09625454917112,-0.544788599687455 -1.06920782902604,-0.6 -1.03923048454133,-0.653566842018031 -1.00640468153451,-0.705342302750966 -0.970820393249938,-0.755184469259804 -0.932575153748366,-0.802956727630628 -0.891773790572875,-0.848528137423855 -0.848528137423859,-0.891773790572871 -0.802956727630632,-0.932575153748363 -0.755184469259807,-0.970820393249935 -0.70534230275097,-1.00640468153451 -0.653566842018035,-1.03923048454132 -0.6,-1.06920782902604 -0.544788599687459,-1.09625454917112 -0.488083971690964,-1.12029651179664 -0.430041539454364,-1.14126781955418 -0.370820393249941,-1.15911099154688 -0.310582854123029,-1.17377712088057 -0.249494028981315,-1.18522600871416 -0.187721358048281,-1.19342627444193 -0.125434155921189,-1.19835544170549 -0.062803147491537,-1.2 -0.0,-1.19835544170549 0.062803147491527,-1.19342627444193 0.125434155921179,-1.18522600871417 0.187721358048272,-1.17377712088057 0.249494028981306,-1.15911099154688 0.310582854123019,-1.14126781955419 0.370820393249931,-1.12029651179664 0.430041539454355,-1.09625454917112 0.488083971690954,-1.06920782902604 0.54478859968745,-1.03923048454133 0.6,-1.00640468153451 0.653566842018027,-0.970820393249941 0.705342302750962,-0.93257515374837 0.755184469259799,-0.891773790572878 0.802956727630624,-0.848528137423857 0.848528137423857))"

    assert rst[0] == expect

def test_ST_PolygonFromEnvelope2():
    x_min = pandas.Series([0.0 for x in range(1, 40000001)])
    x_max = pandas.Series([1.0 for x in range(1, 40000001)])
    y_min = pandas.Series([2.0 for x in range(1, 40000001)])
    y_max = pandas.Series([3.0 for x in range(1, 40000001)])

    rst = arctern.ST_PolygonFromEnvelope(x_min, y_min, x_max, y_max)
    assert len(rst) == 40000000

def test_ST_PolygonFromEnvelope():
    x_min = pandas.Series([0.0])
    x_max = pandas.Series([1.0])
    y_min = pandas.Series([0.0])
    y_max = pandas.Series([1.0])

    rst = arctern.ST_AsText(arctern.ST_PolygonFromEnvelope(x_min, y_min, x_max, y_max))

    assert rst[0] == "POLYGON ((0 0,0 1,1 1,1 0,0 0))"


def test_ST_Union_Aggr():
    p1 = "POLYGON ((1 1,1 2,2 2,2 1,1 1))"
    p2 = "POLYGON ((2 1,3 1,3 2,2 2,2 1))"
    data = pandas.Series([p1, p2])
    rst = arctern.ST_AsText(arctern.ST_Union_Aggr(arctern.ST_GeomFromText(data)))
    assert rst[0] == "POLYGON ((1 1,1 2,2 2,3 2,3 1,2 1,1 1))"

    p1 = "POLYGON ((0 0,4 0,4 4,0 4,0 0))"
    p2 = "POLYGON ((3 1,5 1,5 2,3 2,3 1))"
    data = pandas.Series([p1, p2])
    rst = arctern.ST_AsText(arctern.ST_Union_Aggr(arctern.ST_GeomFromText(data)))
    assert rst[0] == "POLYGON ((4 1,4 0,0 0,0 4,4 4,4 2,5 2,5 1,4 1))"

    p1 = "POLYGON ((0 0,4 0,4 4,0 4,0 0))"
    p2 = "POLYGON ((5 1,7 1,7 2,5 2,5 1))"
    data = pandas.Series([p1, p2])
    rst = arctern.ST_AsText(arctern.ST_Union_Aggr(arctern.ST_GeomFromText(data)))
    assert rst[0] == "MULTIPOLYGON (((0 0,4 0,4 4,0 4,0 0)),((5 1,7 1,7 2,5 2,5 1)))"

    p1 = "POLYGON ((0 0,0 4,4 4,4 0,0 0))"
    p2 = "POINT (2 3)"

    data = pandas.Series([p1, p2])
    rst = arctern.ST_AsText(arctern.ST_Union_Aggr(arctern.ST_GeomFromText(data)))
    assert rst[0] == p1


def test_ST_Envelope_Aggr():
    p1 = "POLYGON ((0 0,4 0,4 4,0 4,0 0))"
    p2 = "POLYGON ((5 1,7 1,7 2,5 2,5 1))"
    data = pandas.Series([p1, p2])
    rst = arctern.ST_AsText(arctern.ST_Envelope_Aggr(arctern.ST_GeomFromText(data)))
    assert rst[0] == "POLYGON ((0 0,0 4,7 4,7 0,0 0))"
