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

import pandas as pd
from osgeo import ogr
from databricks.koalas import Series
from arctern_spark.geoseries import GeoSeries


def test_ST_IsValid():
    data = GeoSeries(["POINT (1.3 2.6)", "POINT (2.6 4.7)"])
    rst = data.is_valid
    assert rst[0]
    assert rst[1]


def test_ST_IsEmpty():
    data = GeoSeries(["LINESTRING EMPTY", "POINT (100 200)"])
    rst = data.is_empty
    assert rst[0] == 1
    assert rst[1] == 0


def test_ST_Boundary():
    data = GeoSeries(["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", None, "LINESTRING (0 0, 0 1, 1 1)",
                      "POINT (1 0)", "POINT EMPTY"])
    rst = data.boundary.to_wkt()
    assert rst[0] == "LINEARRING (0 0, 1 0, 1 1, 0 1, 0 0)"
    assert rst[1] is None
    assert rst[2] == "MULTIPOINT ((0 0), (1 1))"
    assert rst[3] == "GEOMETRYCOLLECTION EMPTY"
    assert rst[4] == "GEOMETRYCOLLECTION EMPTY"


def test_ST_Difference():
    data1 = GeoSeries(["LINESTRING (0 0,5 0)", "MULTIPOINT ((4 0),(6 0))"])
    data2 = GeoSeries(["LINESTRING (4 0,6 0)", "POINT (4 0)"])
    rst = data1.difference(data2).to_wkt()
    assert rst[0] == "LINESTRING (0 0, 4 0)"
    assert rst[1] == "POINT (6 0)"


def test_ST_SymDifference():
    data1 = GeoSeries(["LINESTRING (0 0,5 0)", "MULTIPOINT ((4 0),(6 0))"])
    data2 = GeoSeries(["LINESTRING (4 0,6 0)", "POINT (4 0)"])
    rst = data1.symmetric_difference(data2).to_wkt()
    assert rst[0] == "MULTILINESTRING ((0 0, 4 0), (5 0, 6 0))"
    assert rst[1] == "POINT (6 0)"


def test_ST_ExteriorRing():
    data = GeoSeries(
        ["LINESTRING (4 0,6 0)", "POLYGON ((0 0,1 0,1 1,0 1,0 0))"])
    rst = data.exterior.to_wkt()
    assert rst[0] == "LINESTRING (4 0, 6 0)"
    assert rst[1] == "LINEARRING (0 0, 1 0, 1 1, 0 1, 0 0)"m慢慢慢慢慢慢慢慢慢慢慢慢慢慢慢慢慢慢慢慢                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 


def test_ST_Translate():
    data = GeoSeries(
        ["POINT (1 6)", "LINESTRING (0 0,0 1,1 1)", "POLYGON ((0 0,0 1,1 1,0 0))"])
    rst = data.translate(1.2, 0.3).to_wkt()
    assert len(rst) == 3
    assert rst[0] == "POINT (2.2 6.3)"
    assert rst[1] == "LINESTRING (1.2 0.3, 1.2 1.3, 2.2 1.3)"
    assert rst[2] == "POLYGON ((1.2 0.3, 1.2 1.3, 2.2 1.3, 1.2 0.3))"


def test_ST_Scale2():
    data = GeoSeries(["LINESTRING (0 0,5 0)", "MULTIPOINT ((4 0),(6 0))"])
    rst = data.scale(2, 2, origin=(0, 0)).to_wkt()
    assert rst[0] == "LINESTRING (0 0, 10 0)"
    assert rst[1] == "MULTIPOINT ((8 0), (12 0))"


def test_ST_Scale():
    data = GeoSeries(["LINESTRING (0 0,5 0)", "MULTIPOINT ((4 0),(6 0))"])
    rst = data.scale(2, 2, origin="center").to_wkt()
    assert rst[0] == "LINESTRING (-2.5 0, 7.5 0)"
    assert rst[1] == "MULTIPOINT ((3 0), (7 0))"

def test_ST_Affine():
    data = GeoSeries(["LINESTRING (0 0,5 0)", "MULTIPOINT ((4 0),(6 0))"])
    matrix = (2, 2, 2, 2, 2, 2)
    rst = data.affine(*matrix).to_wkt()
    assert rst[0] == "LINESTRING (2 2, 12 12)"
    assert rst[1] == "MULTIPOINT ((10 10), (14 14))"


def test_ST_Rotate():
    p1 = "Point(1 2)"
    p2 = "LineString (1 1, 2 2, 1 2)"
    p3 = "Polygon ((3 3, 3 5, 5 5, 5 3, 3 3))"

    data = GeoSeries([p1, p2, p3])
    rst1 = data.rotate(90, (0, 0)).precision_reduce(3).to_wkt()
    assert rst1[0] == "POINT (-2 1)"
    assert rst1[1] == "LINESTRING (-1 1, -2 2, -2 1)"
    assert rst1[2] == "POLYGON ((-3 3, -5 3, -5 5, -3 5, -3 3))"

    rst2 = data.rotate(90, "centroid").precision_reduce(3).to_wkt()
    assert rst2[0] == "POINT (1 2)"
    assert rst2[1] == "LINESTRING (2.207 1.207, 1.207 2.207, 1.207 1.207)"
    assert rst2[2] == "POLYGON ((5 3, 3 3, 3 5, 5 5, 5 3))"

    rst3 = data.rotate(90).precision_reduce(3).to_wkt()
    assert rst3[0] == "POINT (1 2)"
    assert rst3[1] == "LINESTRING (2 1, 1 2, 1 1)"
    assert rst3[2] == "POLYGON ((5 3, 3 3, 3 5, 5 5, 5 3))"


def test_ST_Disjoint():
    p11 = "POLYGON((0 0,1 0,1 1,0 1,0 0))"
    p12 = "POLYGON((8 0,9 0,9 1,8 1,8 0))"
    p13 = "LINESTRING(2 2,3 2)"
    p14 = "POINT(10 2)"
    data1 = GeoSeries([p11, p12, p13, p14])

    p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p23 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p24 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    data2 = GeoSeries([p21, p22, p23, p24])

    rst = data2.disjoint(data1)
    assert rst[0] == 0
    assert rst[1] == 0
    assert rst[2] == 0
    assert rst[3] == 1


def test_ST_Union():
    p11 = "POINT (0 1)"
    p12 = "LINESTRING (0 0, 0 1, 1 1)"
    p13 = "LINESTRING (0 0, 1 0, 1 1, 0 0)"
    p14 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
    p15 = "MULTIPOINT (0 0, 1 0, 1 2, 1 2)"
    p16 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,1 -3,-2 1) )"
    p17 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )"
    data1 = GeoSeries([p11, p12, p13, p14, p15, p16, p17])

    p21 = "POLYGON ((0 0,0 2,2 2,0 0))"
    p22 = "LINESTRING (0 0, 0 1, 1 2)"
    p23 = "POINT (2 3)"
    p24 = "MULTIPOINT (0 0, 1 0, 1 2, 1 2)"
    p25 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,1 -3,-2 1) )"
    p26 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )"
    p27 = "POINT (1 5)"
    data2 = GeoSeries([p21, p22, p23, p24, p25, p26, p27])

    rst = data1.union(data2).to_wkt()
    assert rst[0] == "POLYGON ((0 0, 0 2, 2 2, 0 0))"
    assert rst[1] == "MULTILINESTRING ((0 0, 0 1), (0 1, 1 1), (0 1, 1 2))"
    assert rst[2] == "GEOMETRYCOLLECTION (POINT (2 3), LINESTRING (0 0, 1 0, 1 1, 0 0))"
    assert rst[3] == "GEOMETRYCOLLECTION (POINT (1 2), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))"
    assert rst[4] == "MULTILINESTRING ((0 0, 1 2), (0 0, 1 0, 1 1), (-1 2, 3 4, 1 -3, -2 1))"
    assert rst[5] == "GEOMETRYCOLLECTION (LINESTRING (-1 2, 0.7142857142857143 2.857142857142857), LINESTRING (1 3, 3 4, 1 -3, -2 1), POLYGON ((1 0, 0 0, 0.7142857142857143 2.857142857142857, 1 4, 1 3, 1 2, 1 1, 1 0)))"
    assert rst[6] == "GEOMETRYCOLLECTION (POINT (1 5), POLYGON ((0 0, 1 4, 1 0, 0 0)))"


def test_ST_PrecisionReduce():
    data = GeoSeries(["POINT (1.333 2.666)", "POINT (2.655 4.447)"])
    rst = data.precision_reduce(3).to_wkt()
    assert rst[0] == "POINT (1.333 2.666)"
    assert rst[1] == "POINT (2.655 4.447)"


def test_ST_Intersection():
    data1 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POINT (0 1)"])
    data2 = GeoSeries(["POLYGON ((2 1,3 1,3 2,2 2,2 1))", "POINT (0 1)"])
    rst = data1.intersection(data2).to_wkt()
    assert len(rst) == 2
    assert rst[0] == "LINESTRING (2 2, 2 1)"
    assert rst[1] == "POINT (0 1)"

    rst = data1.intersection(GeoSeries("POINT (0 1)")[0]).to_wkt()
    assert len(rst) == 2
    assert rst[0] == "GEOMETRYCOLLECTION EMPTY"
    assert rst[1] == "POINT (0 1)"


def test_ST_Equals():
    data1 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))",
                       "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    data2 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))",
                       "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
    rst = data1.geom_equals(data2)
    assert len(rst) == 2
    assert rst[0] == 1
    assert rst[1] == 0

    rst = data2.geom_equals(GeoSeries("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0])
    assert len(rst) == 2
    assert rst[0] == 1
    assert rst[1] == 0


def test_ST_Touches():
    data1 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))",
                       "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    data2 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))",
                       "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
    rst = data1.touches(data2)
    assert len(rst) == 2
    assert rst[0] == 0
    assert rst[1] == 1

    rst = data2.touches(GeoSeries("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0])
    assert len(rst) == 2
    assert rst[0] == 0
    assert rst[1] == 1


def test_ST_Overlaps():
    data1 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))",
                       "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    data2 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))",
                       "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
    rst = data1.overlaps(data2)
    assert len(rst) == 2
    assert rst[0] == 0
    assert rst[1] == 0

    rst = data2.overlaps(data1[0])
    assert len(rst) == 2
    assert rst[0] == 0
    assert rst[1] == 0


def test_ST_Crosses():
    data1 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))",
                       "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    data2 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))",
                       "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
    rst = data1.crosses(data2)
    assert len(rst) == 2
    assert rst[0] == 0
    assert rst[1] == 0

    rst = data2.crosses(data2[0])
    assert len(rst) == 2
    assert rst[0] == 0
    assert rst[1] == 0


def test_ST_IsSimple():
    data = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))",
                      "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    rst = data.is_simple
    assert rst[0] == 1
    assert rst[1] == 1


def test_ST_GeometryType():
    data = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))",
                      "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    rst = data.geom_type
    assert rst[0] == "Polygon"
    assert rst[1] == "Polygon"


def test_ST_MakeValid():
    data = GeoSeries(["POLYGON ((2 1,3 1,3 2,2 2,2 8,2 1))"])
    rst = data.make_valid().to_wkt()
    assert rst[0] == "POLYGON ((2 1, 3 1, 3 2, 2 2, 2 8, 2 1))"


def test_ST_SimplifyPreserveTopology():
    data = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))",
                      "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
    rst = data.simplify(10000).to_wkt()
    assert rst[0] == "POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))"


def test_ST_Point():
    data1 = [1.3, 2.5]
    data2 = [3.8, 4.9]
    string_ptr = GeoSeries.point(data1, data2).to_wkt()
    assert len(string_ptr) == 2
    assert string_ptr[0] == "POINT (1.3 3.8)"
    assert string_ptr[1] == "POINT (2.5 4.9)"

    # data is koalas series
    string_ptr = GeoSeries.point(Series([1, 2], dtype='double'), 5).to_wkt()
    assert len(string_ptr) == 2
    assert string_ptr[0] == "POINT (1 5)"
    assert string_ptr[1] == "POINT (2 5)"

    string_ptr = GeoSeries.point(5, Series([1, 2], dtype='double')).to_wkt()
    assert len(string_ptr) == 2
    assert string_ptr[0] == "POINT (5 1)"
    assert string_ptr[1] == "POINT (5 2)"

    # data is pandas series
    string_ptr = GeoSeries.point(pd.Series([1, 2], dtype='double'), 5).to_wkt()
    assert len(string_ptr) == 2
    assert string_ptr[0] == "POINT (1 5)"
    assert string_ptr[1] == "POINT (2 5)"

    string_ptr = GeoSeries.point(5, pd.Series([1, 2], dtype='double')).to_wkt()
    assert len(string_ptr) == 2
    assert string_ptr[0] == "POINT (5 1)"
    assert string_ptr[1] == "POINT (5 2)"

    # data is literal
    string_ptr = GeoSeries.point(5.0, 1.0).to_wkt()
    assert len(string_ptr) == 1
    assert string_ptr[0] == "POINT (5 1)"


def test_ST_GeomFromGeoJSON():
    # data is koalas series
    j0 = "{\"type\":\"Point\",\"coordinates\":[1,2]}"
    j1 = "{\"type\":\"LineString\",\"coordinates\":[[1,2],[4,5],[7,8]]}"
    j2 = "{\"type\":\"Polygon\",\"coordinates\":[[[0,0],[0,1],[1,1],[1,0],[0,0]]]}"
    data = Series([j0, j1, j2])
    str_ptr = GeoSeries.geom_from_geojson(data).to_wkt()
    assert str_ptr[0] == "POINT (1 2)"
    assert str_ptr[1] == "LINESTRING (1 2, 4 5, 7 8)"
    assert str_ptr[2] == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"

    # data is pandas series
    data = pd.Series([j0, j1, j2])
    str_ptr = GeoSeries.geom_from_geojson(data).to_wkt()
    assert str_ptr[0] == "POINT (1 2)"
    assert str_ptr[1] == "LINESTRING (1 2, 4 5, 7 8)"
    assert str_ptr[2] == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"


def test_ST_AsGeoJSON():
    j0 = "{\"type\":\"Point\",\"coordinates\":[1,2]}"
    j1 = "{\"type\":\"LineString\",\"coordinates\":[[1,2],[4,5],[7,8]]}"
    j2 = "{\"type\":\"Polygon\",\"coordinates\":[[[0,0],[0,1],[1,1],[1,0],[0,0]]]}"
    data = Series([j0, j1, j2])
    str_ptr = GeoSeries.geom_from_geojson(data).as_geojson()
    assert str_ptr[0] == '{"type":"Point","coordinates":[1.0,2.0]}'
    assert str_ptr[1] == '{"type":"LineString","coordinates":[[1.0,2.0],[4.0,5.0],[7.0,8.0]]}'
    assert str_ptr[2] == '{"type":"Polygon","coordinates":[[[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]]]}'


def test_ST_Contains():
    p11 = "POLYGON((0 0,1 0,1 1,0 1,0 0))"
    p12 = "POLYGON((8 0,9 0,9 1,8 1,8 0))"
    p13 = "POINT(2 2)"
    p14 = "POINT(200 2)"
    data1 = GeoSeries([p11, p12, p13, p14])

    p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p23 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p24 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    data2 = GeoSeries([p21, p22, p23, p24])
    rst = data2.contains(data1)
    assert len(rst) == 4
    assert rst[0] == 1
    assert rst[1] == 0
    assert rst[2] == 1
    assert rst[3] == 0

    rst = data2.contains(data2[0])
    assert len(rst) == 4
    assert rst[0] == 1
    assert rst[1] == 1
    assert rst[2] == 1
    assert rst[3] == 1


def test_ST_Intersects():
    p11 = "POLYGON((0 0,1 0,1 1,0 1,0 0))"
    p12 = "POLYGON((8 0,9 0,9 1,8 1,8 0))"
    p13 = "LINESTRING(2 2,10 2)"
    p14 = "LINESTRING(9 2,10 2)"
    data1 = GeoSeries([p11, p12, p13, p14])

    p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p23 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p24 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    data2 = GeoSeries([p21, p22, p23, p24])

    rst = data2.intersects(data1)
    assert rst[0] == 1
    assert rst[1] == 1
    assert rst[2] == 1
    assert rst[3] == 0

    rst = data1.intersects(data2[0])
    assert len(rst) == 4
    assert rst[0] == 1
    assert rst[1] == 1
    assert rst[2] == 1
    assert rst[3] == 0


def test_ST_Within():
    p11 = "POLYGON((0 0,1 0,1 1,0 1,0 0))"
    p12 = "POLYGON((8 0,9 0,9 1,8 1,8 0))"
    p13 = "LINESTRING(2 2,3 2)"
    p14 = "POINT(10 2)"
    data1 = GeoSeries([p11, p12, p13, p14])

    p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p23 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p24 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    data2 = GeoSeries([p21, p22, p23, p24])

    rst = data2.within(data1)
    assert len(rst) == 4
    assert rst[0] == 0
    assert rst[1] == 0
    assert rst[2] == 0
    assert rst[3] == 0

    rst = data1.within(data2[0])
    assert len(rst) == 4
    assert rst[0] == 1
    assert rst[1] == 0
    assert rst[2] == 1
    assert rst[3] == 0


def test_ST_Distance():
    p11 = "LINESTRING(9 0,9 2)"
    p12 = "POINT(10 2)"
    data1 = GeoSeries([p11, p12])

    p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
    data2 = GeoSeries([p21, p22])

    rst = data2.distance(data1)
    assert len(rst) == 2
    assert rst[0] == 1.0
    assert rst[1] == 2.0

    rst = data1.distance(data2[0])
    assert len(rst) == 2
    assert rst[0] == 1.0
    assert rst[1] == 2.0


def test_ST_DistanceSphere():
    import math
    p11 = "POINT(-73.981153 40.741841)"
    p12 = "POINT(200 10)"
    data1 = GeoSeries([p11, p12], crs="EPSG:4326")

    p21 = "POINT(-73.99016751859183 40.729884354626904)"
    p22 = "POINT(10 2)"
    data2 = GeoSeries([p21, p22], crs="EPSG:4326")

    rst = data2.distance_sphere(data1)
    assert len(rst) == 2
    assert abs(rst[0] - 1531) < 1
    assert math.isnan(rst[1])

    data = GeoSeries(["POINT(0 0)"], crs="EPSG:4326")
    rst = data.distance_sphere(data[0])
    assert len(rst) == 1
    assert math.isclose(rst[0], 0.0, rel_tol=1e-5)


def test_ST_Area():
    data = ["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"]
    data = GeoSeries(data)
    rst = data.area

    assert rst[0] == 1.0
    assert rst[1] == 64.0


def test_ST_Centroid():
    data = ["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"]
    data = GeoSeries(data)
    rst = data.centroid.to_wkt()

    assert rst[0] == "POINT (0.5 0.5)"
    assert rst[1] == "POINT (4 4)"


def test_ST_Length():
    data = ["LINESTRING(0 0,0 1)", "LINESTRING(1 1,1 4)"]
    data = GeoSeries(data)
    rst = data.length

    assert rst[0] == 1.0
    assert rst[1] == 3.0


def test_ST_HausdorffDistance():
    import math
    data1 = ["POLYGON((0 0 ,0 1, 1 1, 1 0, 0 0))", "POINT(0 0)"]
    data2 = ["POLYGON((0 0 ,0 2, 1 1, 1 0, 0 0))", "POINT(0 1)"]
    data1 = GeoSeries(data1)
    data2 = GeoSeries(data2)
    rst = data1.hausdorff_distance(data2)
    assert len(rst) == 2
    assert rst[0] == 1
    assert rst[1] == 1

    rst = data1.hausdorff_distance(data1[1])
    assert len(rst) == 2
    assert math.isclose(rst[0], math.sqrt(2), rel_tol=1e-5)
    assert rst[1] == 0


def test_ST_ConvexHull():
    data = ["POINT (1.1 101.1)"]
    data = GeoSeries(data)
    rst = data.convex_hull.to_wkt()

    assert rst[0] == "POINT (1.1 101.1)"


def test_ST_Transform():
    data = ["POINT (10 10)"]
    data = GeoSeries(data, crs="EPSG:4326")
    rst = data.to_crs("EPSG:3857").to_wkt()

    wkt = rst[0]
    rst_point = ogr.CreateGeometryFromWkt(str(wkt))
    assert abs(rst_point.GetX() - 1113194.90793274 < 0.01)
    assert abs(rst_point.GetY() - 1118889.97485796 < 0.01)


def test_ST_CurveToLine():
    data = ["CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0))"]
    data = GeoSeries(data)
    rst = data.curve_to_line().to_wkt()

    assert str(rst[0]).startswith("POLYGON")


def test_ST_NPoints():
    data = ["LINESTRING(1 1,1 4)"]
    data = GeoSeries(data)
    rst = data.npoints
    assert rst[0] == 2


def test_ST_Envelope():
    p1 = "point (10 10)"
    p2 = "linestring (0 0 , 0 10)"
    p3 = "linestring (0 0 , 10 0)"
    p4 = "linestring (0 0 , 10 10)"
    p5 = "polygon ((0 0, 10 0, 10 10, 0 10, 0 0))"
    p6 = "multipoint (0 0, 10 0, 5 5)"
    p7 = "multilinestring ((0 0, 5 5), (6 6, 6 7, 10 10))"
    p8 = "multipolygon (((0 0, 10 0, 10 10, 0 10, 0 0)), ((11 11, 20 11, 20 20, 20 11, 11 11)))"
    data = [p1, p2, p3, p4, p5, p6, p7, p8]
    data = GeoSeries(data)
    rst = data.envelope.to_wkt()

    assert rst[0] == "POINT (10 10)"
    assert rst[1] == "LINESTRING (0 0, 0 10)"
    assert rst[2] == "LINESTRING (0 0, 10 0)"
    assert rst[3] == "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))"
    assert rst[4] == "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))"
    assert rst[5] == "POLYGON ((0 0, 0 5, 10 5, 10 0, 0 0))"
    assert rst[6] == "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))"
    assert rst[7] == "POLYGON ((0 0, 0 20, 20 20, 20 0, 0 0))"


def test_ST_Buffer():
    data = ["POLYGON((0 0,1 0,1 1,0 0))"]
    data = GeoSeries(data)
    rst = data.buffer(1.2).to_wkt()
    expect = "POLYGON ((-0.8485281374238569 0.8485281374238569, 0.1514718625761431 1.848528137423857, 0.3333157203764777 1.9977635347630542, 0.5407798811618924 2.1086554390135444, 0.7658916135806462 2.1769423364838767, 1 2.2, 1.234108386419354 2.1769423364838767, 1.4592201188381078 2.1086554390135444, 1.6666842796235226 1.9977635347630542, 1.8485281374238571 1.848528137423857, 1.9977635347630542 1.6666842796235226, 2.1086554390135444 1.4592201188381078, 2.1769423364838767 1.2341083864193538, 2.2 1, 2.2 0, 2.1769423364838767 -0.2341083864193539, 2.1086554390135444 -0.4592201188381077, 1.9977635347630542 -0.6666842796235226, 1.8485281374238571 -0.8485281374238569, 1.6666842796235226 -0.9977635347630542, 1.4592201188381078 -1.1086554390135441, 1.234108386419354 -1.1769423364838765, 1 -1.2, 0 -1.2, -0.2341083864193541 -1.1769423364838765, -0.4592201188381076 -1.1086554390135441, -0.6666842796235223 -0.9977635347630543, -0.8485281374238569 -0.848528137423857, -0.9977635347630543 -0.6666842796235226, -1.1086554390135441 -0.4592201188381079, -1.1769423364838765 -0.2341083864193543, -1.2 -0.0000000000000001, -1.1769423364838765 0.234108386419354, -1.1086554390135441 0.4592201188381076, -0.9977635347630545 0.6666842796235223, -0.8485281374238569 0.8485281374238569))"

    assert rst[0] == expect


def test_ST_PolygonFromEnvelope():
    x_min = Series([0.0])
    x_max = Series([1.0])
    y_min = Series([0.0])
    y_max = Series([1.0])

    rst = GeoSeries.polygon_from_envelope(x_min, y_min, x_max, y_max).to_wkt()

    assert rst[0] == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"


def test_ST_Union_Aggr():
    p1 = "POLYGON ((1 1,1 2,2 2,2 1,1 1))"
    p2 = "POLYGON ((2 1,3 1,3 2,2 2,2 1))"
    data = GeoSeries([p1, p2])
    rst = data.unary_union().to_wkt()
    assert rst[0] == "POLYGON ((1 1, 1 2, 2 2, 3 2, 3 1, 2 1, 1 1))"

    p1 = "POLYGON ((0 0,4 0,4 4,0 4,0 0))"
    p2 = "POLYGON ((3 1,5 1,5 2,3 2,3 1))"
    data = GeoSeries([p1, p2])
    rst = data.unary_union().to_wkt()
    assert rst[0] == "POLYGON ((4 1, 4 0, 0 0, 0 4, 4 4, 4 2, 5 2, 5 1, 4 1))"

    p1 = "POLYGON ((0 0,4 0,4 4,0 4,0 0))"
    p2 = "POLYGON ((5 1,7 1,7 2,5 2,5 1))"
    data = GeoSeries([p1, p2])
    rst = data.unary_union().to_wkt()
    assert rst[0] == "MULTIPOLYGON (((0 0, 0 4, 4 4, 4 0, 0 0)), ((5 1, 5 2, 7 2, 7 1, 5 1)))"

    p1 = "POLYGON ((0 0,0 4,4 4,4 0,0 0))"
    p2 = "POINT (2 3)"

    data = GeoSeries([p1, p2])
    rst = data.unary_union().to_wkt()
    assert rst[0] == "GEOMETRYCOLLECTION (POLYGON ((0 0, 0 4, 4 4, 4 0, 0 0)))"


def test_ST_Envelope_Aggr():
    p1 = "POLYGON ((0 0,4 0,4 4,0 4,0 0))"
    p2 = "POLYGON ((5 1,7 1,7 2,5 2,5 1))"
    data = GeoSeries([p1, p2])
    rst = data.envelope_aggr().to_wkt()
    assert rst[0] == "POLYGON ((0 0, 0 4, 7 4, 7 0, 0 0))"
