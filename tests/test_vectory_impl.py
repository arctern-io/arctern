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
import arctern


def test_suite():
    from multiprocessing import Process
    import time
    p1 = Process(target=ST_Intersection)
    p2 = Process(target=ST_Equals)
    p3 = Process(target=ST_Touches)
    p4 = Process(target=ST_Overlaps)
    p5 = Process(target=ST_Crosses)
    p6 = Process(target=ST_Point)
    p7 = Process(target=ST_Contains)
    p8 = Process(target=ST_Intersects)
    p9 = Process(target=ST_Distance)
    p10 = Process(target=ST_DistanceSphere)
    p11 = Process(target=ST_HausdorffDistance)
    p12 = Process(target=ST_PolygonFromEnvelope)
    start = time.time()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()
    p10.start()
    p11.start()
    p12.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
    p10.join()
    p11.join()
    p12.join()
    end = time.time()
    print('Task runs %0.2f seconds.' % ((end - start)))


def ST_Intersection():
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


def ST_Equals():
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


def ST_Touches():
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


def ST_Overlaps():
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


def ST_Crosses():
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


def ST_Point():
    geo1 = 1.1
    geo2 = 2.1
    arr1 = [geo1 for x in range(1, 40000001)]
    arr2 = [geo2 for x in range(1, 40000001)]

    data1 = pandas.Series(arr1)
    data2 = pandas.Series(arr2)
    rst = arctern.ST_Point(data1, data2)
    assert len(rst) == 40000000


def ST_Contains():
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


def ST_Intersects():
    geo1 = "POLYGON ((113.66220266388723 22.39277623851494, 114.58136061218778 22.39277623851494, 114.58136061218778 22.92800492531275 ,113.66220266388723 22.92800492531275, 113.66220266388723 22.39277623851494))"
    geo2 = "POINT (1 1)"
    geo_wkb1 = arctern.ST_GeomFromText(geo1)[0]
    geo_wkb2 = arctern.ST_GeomFromText(geo2)[0]
    arr1 = [geo_wkb1 for x in range(1, 40000001)]
    arr2 = [geo_wkb2 for x in range(1, 40000001)]

    data1 = pandas.Series(arr1)
    data2 = pandas.Series(arr2)
    rst = arctern.ST_Intersects(data1, data2)
    assert len(rst) == 40000000


def ST_Within():
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


def ST_Distance():
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


def ST_DistanceSphere():
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


def ST_HausdorffDistance():
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


def ST_PolygonFromEnvelope():
    x_min = pandas.Series([0.0 for x in range(1, 40000001)])
    x_max = pandas.Series([1.0 for x in range(1, 40000001)])
    y_min = pandas.Series([2.0 for x in range(1, 40000001)])
    y_max = pandas.Series([3.0 for x in range(1, 40000001)])

    rst = arctern.ST_PolygonFromEnvelope(x_min, y_min, x_max, y_max)
    assert len(rst) == 40000000


if __name__ == "__main__":
    test_suite()
