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


def test_ST_Intersection():
    geo1 = "POLYGON ((113.66220266388723 22.39277623851494, 114.58136061218778 22.39277623851494, 114.58136061218778 22.92800492531275 ,113.66220266388723 22.92800492531275, 113.66220266388723 22.39277623851494))"
    geo2 = "POINT (1 1)"
    geo_wkb1 = arctern.ST_GeomFromText(geo1)[0]
    geo_wkb2 = arctern.ST_GeomFromText(geo2)[0]

    arr1 = [geo_wkb1 for x in range(1, 40000001)]
    arr2 = [geo_wkb2 for x in range(1, 40000001)]

    data1 = pandas.Series(arr1)
    data2 = pandas.Series(arr2)
    rst = arctern.ST_Intersection(data1, data2)
    assert len(rst) == 40000000

def test_ST_Equals():
    geo1 = "POLYGON ((113.66220266388723 22.39277623851494, 114.58136061218778 22.39277623851494, 114.58136061218778 22.92800492531275 ,113.66220266388723 22.92800492531275, 113.66220266388723 22.39277623851494))"
    geo2 = "POINT (1 1)"
    geo_wkb1 = arctern.ST_GeomFromText(geo1)[0]
    geo_wkb2 = arctern.ST_GeomFromText(geo2)[0]
    arr1 = [geo_wkb1 for x in range(1, 40000001)]
    arr2 = [geo_wkb2 for x in range(1, 40000001)]

    data1 = pandas.Series(arr1)
    data2 = pandas.Series(arr2)
    rst = arctern.ST_Equals(data1, data2)
    assert len(rst) == 40000000


def test_ST_Touches():
    geo1 = "POLYGON ((113.66220266388723 22.39277623851494, 114.58136061218778 22.39277623851494, 114.58136061218778 22.92800492531275 ,113.66220266388723 22.92800492531275, 113.66220266388723 22.39277623851494))"
    geo2 = "POINT (1 1)"
    geo_wkb1 = arctern.ST_GeomFromText(geo1)[0]
    geo_wkb2 = arctern.ST_GeomFromText(geo2)[0]
    arr1 = [geo_wkb1 for x in range(1, 40000001)]
    arr2 = [geo_wkb2 for x in range(1, 40000001)]

    data1 = pandas.Series(arr1)
    data2 = pandas.Series(arr2)
    rst = arctern.ST_Touches(data1, data2)
    assert len(rst) == 40000000


def test_ST_Overlaps():
    geo1 = "POLYGON ((113.66220266388723 22.39277623851494, 114.58136061218778 22.39277623851494, 114.58136061218778 22.92800492531275 ,113.66220266388723 22.92800492531275, 113.66220266388723 22.39277623851494))"
    geo2 = "POINT (1 1)"
    geo_wkb1 = arctern.ST_GeomFromText(geo1)[0]
    geo_wkb2 = arctern.ST_GeomFromText(geo2)[0]
    arr1 = [geo_wkb1 for x in range(1, 40000001)]
    arr2 = [geo_wkb2 for x in range(1, 40000001)]

    data1 = pandas.Series(arr1)
    data2 = pandas.Series(arr2)
    rst = arctern.ST_Overlaps(data1, data2)
    assert len(rst) == 40000000


def test_ST_Crosses():
    geo1 = "POLYGON ((113.66220266388723 22.39277623851494, 114.58136061218778 22.39277623851494, 114.58136061218778 22.92800492531275 ,113.66220266388723 22.92800492531275, 113.66220266388723 22.39277623851494))"
    geo2 = "POINT (1 1)"
    geo_wkb1 = arctern.ST_GeomFromText(geo1)[0]
    geo_wkb2 = arctern.ST_GeomFromText(geo2)[0]
    arr1 = [geo_wkb1 for x in range(1, 40000001)]
    arr2 = [geo_wkb2 for x in range(1, 40000001)]

    data1 = pandas.Series(arr1)
    data2 = pandas.Series(arr2)
    rst = arctern.ST_Crosses(data1, data2)
    assert len(rst) == 40000000


def test_ST_Point():
    geo1 = 1.1
    geo2 = 2.1
    arr1 = [geo1 for x in range(1, 40000001)]
    arr2 = [geo2 for x in range(1, 40000001)]

    data1 = pandas.Series(arr1)
    data2 = pandas.Series(arr2)
    rst = arctern.ST_Point(data1, data2)
    assert len(rst) == 40000000

def test_ST_Contains():
    geo1 = "POLYGON ((113.66220266388723 22.39277623851494, 114.58136061218778 22.39277623851494, 114.58136061218778 22.92800492531275 ,113.66220266388723 22.92800492531275, 113.66220266388723 22.39277623851494))"
    geo2 = "POINT (1 1)"
    geo_wkb1 = arctern.ST_GeomFromText(geo1)[0]
    geo_wkb2 = arctern.ST_GeomFromText(geo2)[0]
    arr1 = [geo_wkb1 for x in range(1, 40000001)]
    arr2 = [geo_wkb2 for x in range(1, 40000001)]

    data1 = pandas.Series(arr1)
    data2 = pandas.Series(arr2)
    rst = arctern.ST_Contains(data1, data2)
    assert len(rst) == 40000000


def test_ST_Intersects():
    import sys
    geo1 = "POLYGON ((113.66220266388723 22.39277623851494, 114.58136061218778 22.39277623851494, 114.58136061218778 22.92800492531275 ,113.66220266388723 22.92800492531275, 113.66220266388723 22.39277623851494))"
    geo2 = "POINT (1 1)"
    geo_wkb1 = arctern.ST_GeomFromText(geo1)[0]
    geo_wkb2 = arctern.ST_GeomFromText(geo2)[0]
    arr1 = [geo_wkb1 for x in range(1, 40000001)]
    arr2 = [geo_wkb2 for x in range(1, 40000001)]

    data1 = pandas.Series(arr1)
    data2 = pandas.Series(arr2)
    print('------>  size: {:.2f}GB'.format(sys.getsizeof(data1)/1024**3))
    rst = arctern.ST_Intersects(data1, data2)
    assert len(rst) == 40000000


def test_ST_Within():
    geo1 = "POLYGON ((113.66220266388723 22.39277623851494, 114.58136061218778 22.39277623851494, 114.58136061218778 22.92800492531275 ,113.66220266388723 22.92800492531275, 113.66220266388723 22.39277623851494))"
    geo2 = "POINT (1 1)"
    geo_wkb1 = arctern.ST_GeomFromText(geo1)[0]
    geo_wkb2 = arctern.ST_GeomFromText(geo2)[0]
    arr1 = [geo_wkb1 for x in range(1, 40000001)]
    arr2 = [geo_wkb2 for x in range(1, 40000001)]

    data1 = pandas.Series(arr1)
    data2 = pandas.Series(arr2)
    rst = arctern.ST_Within(data1, data2)
    assert len(rst) == 40000000


def test_ST_Distance():
    geo1 = "POLYGON ((113.66220266388723 22.39277623851494, 114.58136061218778 22.39277623851494, 114.58136061218778 22.92800492531275 ,113.66220266388723 22.92800492531275, 113.66220266388723 22.39277623851494))"
    geo2 = "POINT (1 1)"
    geo_wkb1 = arctern.ST_GeomFromText(geo1)[0]
    geo_wkb2 = arctern.ST_GeomFromText(geo2)[0]
    arr1 = [geo_wkb1 for x in range(1, 40000001)]
    arr2 = [geo_wkb2 for x in range(1, 40000001)]

    data1 = pandas.Series(arr1)
    data2 = pandas.Series(arr2)
    rst = arctern.ST_Distance(data1, data2)
    assert len(rst) == 40000000


def test_ST_DistanceSphere():
    geo1 = "POLYGON ((113.66220266388723 22.39277623851494, 114.58136061218778 22.39277623851494, 114.58136061218778 22.92800492531275 ,113.66220266388723 22.92800492531275, 113.66220266388723 22.39277623851494))"
    geo2 = "POINT (1 1)"
    geo_wkb1 = arctern.ST_GeomFromText(geo1)[0]
    geo_wkb2 = arctern.ST_GeomFromText(geo2)[0]
    arr1 = [geo_wkb1 for x in range(1, 40000001)]
    arr2 = [geo_wkb2 for x in range(1, 40000001)]

    data1 = pandas.Series(arr1)
    data2 = pandas.Series(arr2)
    rst = arctern.ST_DistanceSphere(data1, data2)
    assert len(rst) == 40000000


def test_ST_HausdorffDistance():
    geo1 = "POLYGON ((113.66220266388723 22.39277623851494, 114.58136061218778 22.39277623851494, 114.58136061218778 22.92800492531275 ,113.66220266388723 22.92800492531275, 113.66220266388723 22.39277623851494))"
    geo2 = "POINT (1 1)"
    geo_wkb1 = arctern.ST_GeomFromText(geo1)[0]
    geo_wkb2 = arctern.ST_GeomFromText(geo2)[0]
    arr1 = [geo_wkb1 for x in range(1, 40000001)]
    arr2 = [geo_wkb2 for x in range(1, 40000001)]

    data1 = pandas.Series(arr1)
    data2 = pandas.Series(arr2)
    rst = arctern.ST_HausdorffDistance(data1, data2)
    assert len(rst) == 40000000

def test_ST_PolygonFromEnvelope():
    x_min = pandas.Series([0.0 for x in range(1, 4000001)])
    x_max = pandas.Series([1.0 for x in range(1, 4000001)])
    y_min = pandas.Series([2.0 for x in range(1, 4000001)])
    y_max = pandas.Series([3.0 for x in range(1, 4000001)])

    rst = arctern.ST_PolygonFromEnvelope(x_min, y_min, x_max, y_max)
    assert len(rst) == 4000000
