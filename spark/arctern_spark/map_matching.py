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

# pylint: disable=protected-access

from databricks.koalas import DataFrame, Series
from arctern_spark import GeoSeries

__all__ = [
    "near_road",
    "nearest_road",
    "nearest_location_on_road",
]


def _invoke_scala_udf(func, points, roads, *args):
    assert isinstance(roads, GeoSeries)
    assert isinstance(points, GeoSeries)

    if roads.crs != points.crs or roads.crs != "EPSG:4326":
        raise ValueError("Only can calculate with 'EPSG:4326' crs.")
    from arctern_spark import scala_wrapper
    sdf = getattr(scala_wrapper, func)(points._kdf.spark.frame(), roads._kdf.spark.frame(), *args)
    kdf = DataFrame(sdf)
    return kdf


def near_road(roads, points, distance=100.0):
    # TODO: FIX the doc
    """
    Tests whether there is a road within the given ``distance`` of all ``points``. The points do not need to be part of a continuous path.
    Parameters
    ----------
    roads : GeoSeries
        Sequence of LINGSTRING objects.
    points : GeoSeries
        Sequence of POINT objects.
    distance : double, optional
        Searching distance around the points, by default 100.0.

    Returns
    -------
    Series
        A Series that contains only boolean value that indicates whether there is a road within the given ``distance`` of all ``points``.
        * *True*: The road exists.
        * *False*: The road does not exist.
    Examples
    -------
    >>> import arctern_spark
    >>> data1 = arctern_spark.GeoSeries(["LINESTRING (1 2,1 3)"])
    >>> data2 = arctern_spark.GeoSeries(["POINT (1.0001 2.5)"])
    >>> rst = arctern_spark.near_road(data1, data2)
    >>> rst
    0    True
    dtype: object
    """
    return Series(_invoke_scala_udf("near_road", points, roads, distance), index=("near_road",))


def nearest_road(roads, points):
    """
    Returns the road in ``roads`` closest to the ``points``. The points do not need to be part of a continuous path.
    Parameters
    ----------
    roads : GeoSeries
        Sequence of LINGSTRING objects.
    points : Series
        Sequence of POINT objects.

    Returns
    -------
    GeoSeries
        Sequence of LINGSTRING objects.

    Examples
    -------
    >>> import arctern_spark
    >>> data1 = arctern_spark.GeoSeries(["LINESTRING (1 2,1 3)"])
    >>> data2 = arctern_spark.GeoSeries(["POINT (1.001 2.5)"])
    >>> rst = arctern_spark.nearest_road(data1, data2)
    >>> rst
        0    LINESTRING (1 2,1 3)
        dtype: object
    """
    return GeoSeries(_invoke_scala_udf("nearest_road", points, roads), index=('nearest_road',), crs=roads.crs)


def nearest_location_on_road(roads, points):
    """
    Returns the location on ``roads`` closest to the ``points``. The points do not need to be part of a continuous path.
    Parameters
    ----------
    roads : Series
        Sequence of LINGSTRING objects.
    points : Series
        Sequence of POINT objects.
    Returns
    -------
    GeoSeries
        Sequence of POINT objects.
    Examples
    -------
    >>> import arctern_spark
    >>> data1 = arctern_spark.GeoSeries(["LINESTRING (1 2,1 3)"])
    >>> data2 = arctern_spark.GeoSeries(["POINT (1.001 2.5)"])
    >>> rst = arctern_spark.nearest_location_on_road(data1, data2)
    >>> rst
    0    POINT (1.0 2.5)
    dtype: object
    """
    return GeoSeries(_invoke_scala_udf("nearest_location_on_road", points, roads),
                     index=('nearest_location_on_road',), crs=roads.crs)
