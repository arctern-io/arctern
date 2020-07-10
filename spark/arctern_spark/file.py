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

# pylint: disable=too-many-lines
# pylint: disable=too-many-public-methods, unused-argument, redefined-builtin, unidiomatic-typecheck
# pylint: disable=wrong-import-order
import json
import warnings

import numpy as np
import pandas as pd
import fiona
from databricks.koalas import DataFrame

from arctern_spark.geodataframe import GeoDataFrame
from arctern_spark.geoseries import GeoSeries


def _read_file(filename, bbox=None, mask=None, rows=None, **kwargs):
    with fiona.Env():
        with fiona.open(filename, "r", **kwargs) as features:
            crs = (
                features.crs["init"]
                if features.crs and "init" in features.crs
                else features.crs_wkt
            )
            properties = list(features.schema["properties"])

            if mask is not None:
                if isinstance(mask, (str, bytes)):
                    mask = GeoSeries(mask)
                if not isinstance(mask, GeoSeries):
                    raise TypeError(f"unsupported mask type {type(mask)}")
                mask = mask.unary_union().as_geojson()
            if bbox is not None and isinstance(bbox, GeoSeries):
                bbox = bbox.envelope_aggr().bbox[0]
            if isinstance(rows, (int, type(None))):
                rows = (rows,)
            elif isinstance(rows, slice):
                rows = (rows.start, rows.stop, rows.step)
            else:
                raise TypeError(f"unsupported rows type {type(rows)}")
            features = features.filter(*rows, bbox=bbox, mask=mask)

            if kwargs.get("ignore_geometry", False):
                return DataFrame(
                    [record["properties"] for record in features], columns=properties
                )
            return _from_features(features, crs=crs, properties=properties, geometry_col="geometry")


def _from_features(features, crs=None, properties=None, geometry_col=None):
    data = {}
    for property_namer in properties:
        data[property_namer] = []
    geometries = []

    for feature in features:
        if hasattr(feature, "__geo_interface__"):
            feature = feature.__geo_interface__
        geometry = json.dumps(feature["geometry"]) if feature["geometry"] else None
        geometries.append(geometry)

        for property_name, value in feature["properties"].items():
            data[property_name].append(value)

    geos = GeoSeries.geom_from_geojson(geometries, crs=crs)
    gdf = GeoDataFrame(data, columns=properties)
    gdf[geometry_col] = geos
    return gdf


def read_file(*args, **kwargs):
    """
    Returns a GeoDataFrame from a file or URL.
    Parameters
        -----------
        filename : str
            File path or file handle to read from.
        bbox : tuple or arctern_spark.GeoSeries, default None
            Filter features by given bounding box, GeoSeries. Cannot be used
            with mask.
        mask : dict | arctern_spark.GeoSeries | dicr, default None
            Filter for features that intersect with the given dict-like geojson
            geometry, GeoSeries. Cannot be used with bbox.
        rows : int or slice, default None
            Load in specific rows by passing an integer (first `n` rows) or a
            slice() object.
        **kwargs :
        Keyword args to be passed to the `open` or `BytesCollection` method
        in the fiona library when opening the file. For more information on
        possible keywords, type:
        ``import fiona; help(fiona.open)``
    Returns
    --------
    GeoDataFrame
        An arctern_spark.GeoDataFrame object.
    """
    return _read_file(*args, **kwargs)


def infer_schema(df, geo_col):
    from collections import OrderedDict

    types = {"Int64": "int", "string": "str", "boolean": "bool"}

    def convert_type(in_type):
        if in_type == object:
            return "str"
        if in_type.name.startswith("datatime64"):
            return "datetime"
        if str(in_type) in types:
            out_type = types[str(in_type)]
        else:
            out_type = type(np.zeros(1, in_type).item()).__name__
        if out_type == "long":
            out_type = "int"
        return out_type

    properties = OrderedDict(
        [
            (col, convert_type(_type))
            for col, _type in zip(df.columns, df.dtypes)
            if col != geo_col
        ]
    )
    schema = {"geometry": "Unknown", "properties": properties}

    return schema


def _to_file(
        df,
        filename,
        driver="ESRI Shapefile",
        schema=None,
        index=None,
        mode="w",
        crs=None,
        geometry=None,
        **kwargs
):
    if index is None:
        index = list(df.index.names) != [None] or type(df.index.to_pandas()) not in (
            pd.RangeIndex,
            pd.Int64Index,
        )
    if index:
        df = df.reset_index(drop=False)
    if schema is None:
        schema = infer_schema(df, geometry)
    if not crs:
        crs = df[geometry].crs

    if driver == "ESRI Shapefile" and any([len(c) > 10 for c in df.columns]):
        warnings.warn(
            "Column names longer than 10 characters will be truncated when saved to "
            "ESRI Shapefile.",
            stacklevel=3,
        )

    with fiona.Env():
        with fiona.open(
                filename, mode=mode, driver=driver, crs_wkt=crs, schema=schema, **kwargs
        ) as colxn:
            colxn.writerecords(df.iterfeatures(geometry=geometry))


def to_file(*args, **kwargs):
    """
    Write this GeoDataFrame to an OGR data source
    Parameters
    ----------
    df : GeoDataFrame to be written
    filename : str
        File path or file handle to write to.
    driver : string, default 'ESRI Shapefile'
        The OGR format driver used to write the vector file.
    schema : dict, default None
        If specified, the schema dictionary is passed to Fiona to
        better control how the file is written. If None, GeoPandas
        will determine the schema based on each column's dtype.
    index : bool, default None
        If True, write index into one or more columns (for MultiIndex).
        Default None writes the index into one or more columns only if
        the index is named, is a MultiIndex, or has a non-integer data
        type. If False, no index is written.
    mode : str, default 'w'
        The write mode, 'w' to overwrite the existing file and 'a' to append.
    crs : str, default None
        If specified, the CRS is passed to Fiona to
        better control how the file is written. If None, GeoPandas
        will determine the crs based on crs df attribute.
    geomrtry : str, default None
        Specify geometry column.
    Examples
    ---------
    >>> from arctern_spark import GeoDataFrame
    >>> import numpy as np
    >>> data = {
    ...     "A": range(5),
    ...     "B": np.arange(5.0),
    ...     "other_geom": range(5),
    ...     "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
    ...     "geo2": ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"],
    ...     "geo3": ["POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)", "POINT (6 6)"],
    ... }
    >>> gdf = GeoDataFrame(data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
    >>> gdf.to_file(filename="/tmp/test.shp", col="geo1", crs="epsg:3857")
    >>> read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp")
    >>> read_gdf
    A    B  other_geom         geo2         geo3     geometry
    0  0  0.0           0  POINT (1 1)  POINT (2 2)  POINT (0 0)
    1  1  1.0           1  POINT (2 2)  POINT (3 3)  POINT (1 1)
    2  2  2.0           2  POINT (3 3)  POINT (4 4)  POINT (2 2)
    3  3  3.0           3  POINT (4 4)  POINT (5 5)  POI`NT (3 3)
    4  4  4.0           4  POINT (5 5)  POINT (6 6)  POINT (4 4)
    """
    return _to_file(*args, **kwargs)
