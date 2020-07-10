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

# pylint: disable=protected-access,access-member-before-definition,attribute-defined-outside-init

import json
from itertools import zip_longest

import numpy as np
import pandas as pd
from databricks.koalas import DataFrame, Series, get_option
from databricks.koalas.frame import REPR_PATTERN

import arctern_spark
from arctern_spark.geoseries import GeoSeries
from arctern_spark.scala_wrapper import GeometryUDT

_crs_dtype = str


class GeoDataFrame(DataFrame):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, geometries=None, crs=None):
        # (col_name, crs) dict to store crs data of those columns are GeoSeries
        self._crs_for_cols = {}
        self._geometry_column_names = set()

        if isinstance(data, GeoSeries):
            self._crs_for_cols[data.name] = data.crs
            self._geometry_column_names.add(data.name)
        elif isinstance(data, DataFrame):
            for col in data.columns:
                if isinstance(data[col].spark.data_type, GeometryUDT):
                    self._crs_for_cols[col] = None
                    self._geometry_column_names.add(col)
                if isinstance(data[col], GeoSeries):
                    self._crs_for_cols[col] = data[col].crs
                    self._geometry_column_names.add(col)
            data = data._internal_frame

        super(GeoDataFrame, self).__init__(data, index, columns, dtype, copy)

        if geometries is None:
            if "geometry" in self.columns:
                geometries = ["geometry"]
            else:
                geometries = []
        self._set_geometries(geometries, crs=crs)

    # only for internal use
    def _set_geometries(self, cols, crs=None):
        assert isinstance(cols, list), "cols must be list"
        if len(cols) == 0:
            return
        if crs is None or isinstance(crs, _crs_dtype):
            crs = [crs] * len(cols)
        assert isinstance(crs, list), "crs must be list or scalar value"
        assert len(cols) >= len(crs), "The length of crs should less than geometries!"

        # align crs and cols, simply fill None to crs
        for col, _crs in zip_longest(cols, crs):
            if col not in self._geometry_column_names:
                # This set_item operation will lead some BUG in koalas(v1.0.0),
                # see https://github.com/databricks/koalas/issues/1633
                self[col] = GeoSeries(self[col], crs=_crs)
                self._crs_for_cols[col] = _crs
                self._geometry_column_names.add(col)

    def set_geometry(self, col, crs):
        self._set_geometries([col], crs)

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(result, Series) and isinstance(result.spark.data_type, GeometryUDT):
            result.__class__ = GeoSeries
            result._gdf = self
            result.set_crs(self._crs_for_cols.get(result.name, None))
        if isinstance(result, DataFrame):
            crs = {}
            geometry_column_names = []
            geometry_crs = []

            for col in self._crs_for_cols:
                if col in result.columns:
                    crs[col] = self._crs_for_cols.get(col, None)

            for col in self._geometry_column_names:
                if col in result.columns:
                    geometry_column_names.append(col)
                    geometry_crs.append(crs[col])

            if crs or geometry_column_names:
                result.__class__ = GeoDataFrame
                result._crs_for_cols = crs
                result._geometry_column_names = geometry_column_names
                # result._set_geometries(geometry_column_names, geometry_crs)

        return result

    def __setitem__(self, key, value):
        super(GeoDataFrame, self).__setitem__(key, value)
        if isinstance(value, GeoSeries):
            self._crs_for_cols[key] = value.crs
            self._geometry_column_names.add(key)
        elif isinstance(value, GeoDataFrame):
            for col in value._crs_for_cols.keys():
                v = value[col]
                if hasattr(v, "crs"):
                    self._crs_for_cols[col] = v.crs
                    self._geometry_column_names.add(col)
        else:
            if isinstance(key, list):
                pass
            else:
                key = [key]
            for col in key:
                self._crs_for_cols.pop(col)

    def dissolve(self, by, col="geometry", aggfunc="first", as_index=True):
        if col not in self._geometry_column_names:
            raise ValueError(f"`col` {col} must be a geometry column whose data type is GeometryUDT,"
                             f"use `set_geometry` to set this column as geometry column.")
        agg_dict = {aggfunc_col: aggfunc for aggfunc_col in self.columns.tolist() if aggfunc_col not in [col, by]}
        agg_dict[col] = "ST_Union_Aggr"
        aggregated = self.groupby(by=by, as_index=as_index).aggregate(agg_dict)
        gdf = GeoDataFrame(aggregated, geometries=[col], crs=[self._crs_for_cols[col]])
        return gdf

    def merge(
            self,
            right,
            how="inner",
            on=None,
            left_on=None,
            right_on=None,
            left_index=False,
            right_index=False,
            suffixes=("_x", "_y")
    ):
        result = super().merge(right, how, on, left_on, right_on, left_index, right_index, suffixes)
        result = GeoDataFrame(result)

        lsuffix, rsuffix = suffixes
        for col in result.columns:
            kser = result[col]
            if isinstance(kser, GeoSeries):
                pick = self
                if col.endswith(lsuffix) and col not in self.columns:
                    col = col[:-len(lsuffix)]
                elif col.endswith(rsuffix) and col not in right.columns:
                    col = col[:-len(rsuffix)]
                    pick = right
                elif col in right.columns:
                    pick = right

                kser.set_crs(pick._crs_for_cols.get(col, None))

        return result

    def iterfeatures(self, na="null", show_bbox=False, geometry='geometry'):
        if na not in ["null", "drop", "keep"]:
            raise ValueError("Unknown na method {0}".format(na))
        if geometry not in self._geometry_column_names:
            raise ValueError("{} is not a geometry column".format(geometry))
        ids = self.index.to_pandas()
        geometries = self[geometry].as_geojson().to_pandas()
        geometries_bbox = self[geometry].bbox
        properties_cols = self.columns.difference([geometry]).tolist()

        if len(properties_cols) > 0:
            properties = self[properties_cols].to_pandas()
            property_geo_cols = self._geometry_column_names.difference([geometry])

            # since it could be more than one geometry columns in GeoDataFrame,
            # we transform those geometry columns as wkt formed string except column `col`.
            for property_geo_col in property_geo_cols:
                properties[property_geo_col] = self[property_geo_col].to_wkt().to_pandas()
            if na == "null":
                properties[pd.isnull(properties).values] = np.nan

            properties = properties.values
            for i, row in enumerate(properties):
                geom = geometries[i]
                if na == "drop":
                    properties_items = {
                        k: v for k, v in zip(properties_cols, row) if not pd.isnull(v)
                    }
                else:
                    properties_items = dict(zip(properties_cols, row))

                feature = {
                    "id": str(ids[i]),
                    "type": "Feature",
                    "properties": properties_items,
                    "geometry": json.loads(geom) if geom else None,
                }

                if show_bbox:
                    feature["bbox"] = geometries_bbox[i]
                yield feature

        else:
            for fid, geom, bbox in zip(ids, geometries, geometries_bbox):
                feature = {
                    "id": str(fid),
                    "type": "Feature",
                    "properties": {},
                    "geometry": geom,
                }
                if show_bbox:
                    feature["bbox"] = bbox if geom else None
                yield feature

    def reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill=""):
        df = super(GeoDataFrame, self).reset_index(level, drop, inplace, col_level, col_fill)
        if not inplace:
            gdf = GeoDataFrame(df)
            gdf._crs_for_cols = self._crs_for_cols
            return gdf
        return None

    @classmethod
    def from_file(cls, filename, **kwargs):
        """
        Alternate constructor to create a ``GeoDataFrame`` from a file or url.

        Parameters
        -----------
        filename : str
            File path or file handle to read from.
        bbox : tuple(minx, miny, maxx, maxy) or arctern.GeoSeries, default None
            Filter features by given bounding box, GeoSeries. Cannot be used
            with mask.
        mask : dict | arctern.GeoSeries | wkt str | wkb bytes, default None
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
            An GeoDataFrame object.
        """
        return arctern_spark.file.read_file(filename, **kwargs)

    def to_file(self, filename, driver="ESRI Shapefile", geometry=None, schema=None, index=None, crs=None, **kwargs):
        """
        Write the ``GeoDataFrame`` to a file.

        Parameters
        ----------
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
        geometry : str, default None
            Specify geometry column.

        Notes
        -----
        The extra keyword arguments ``**kwargs`` are passed to fiona.open and
        can be used to write to multi-layer data, store data within archives
        (zip files), etc.

        The format drivers will attempt to detect the encoding of your data, but
        may fail. In this case, the proper encoding can be specified explicitly
        by using the encoding keyword parameter, e.g. ``encoding='utf-8'``.

        Examples
        --------
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
        >>> gdf.to_file(filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
        >>> read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp")
        >>> read_gdf
           A    B  other_geom         geo2         geo3     geometry
        0  0  0.0           0  POINT (1 1)  POINT (2 2)  POINT (0 0)
        1  1  1.0           1  POINT (2 2)  POINT (3 3)  POINT (1 1)
        3  2  2.0           2  POINT (3 3)  POINT (4 4)  POINT (3 3)
        2  3  3.0           3  POINT (4 4)  POINT (5 5)  POINT (2 2)
        4  4  4.0           4  POINT (5 5)  POINT (6 6)  POINT (4 4)
        """
        arctern_spark.file.to_file(self, filename=filename, driver=driver, schema=schema,
                                   index=index, crs=crs, geometry=geometry, **kwargs)

    def _to_geo(self, **kwargs):
        geo = {
            "type": "FeatureCollection",
            "features": list(self.iterfeatures(**kwargs))
        }

        if kwargs.get("show_bbox", False):
            # calculate bbox of GeoSeries got from GeoDataFrame will failed,
            # see https://github.com/databricks/koalas/issues/1633
            raise NotImplementedError("show bbox is not implemented yet.")
            # geo["bbox"] = self[kwargs.get("geometry")].envelope_aggr()

        return geo

    # pylint: disable=arguments-differ
    def to_json(self, na="null", show_bbox=False, geometry='geometry', **kwargs):
        """
        Returns a GeoJSON representation of the ``GeoDataFrame`` as a string.

        Parameters
        ----------

        na : {'null', 'drop', 'keep'}, default 'null'
            Indicates how to output missing (NaN) values in the GeoDataFrame.
            See below.
        show_bbox : bool, optional, default: False
            Include bbox (bounds) in the geojson
        geometry : str, optional, default 'geometry'
            Specify geometry column.

        Returns
        -------
        Series
            Sequence of geometries in GeoJSON format.

        Note
        ----
        The remaining *kwargs* are passed to json.dumps().

        Missing (NaN) values in the GeoDataFrame can be represented as follows:

        - ``null``: output the missing entries as JSON null.
        - ``drop``: remove the property from the feature. This applies to each
          feature individually so that features may have different properties.
        - ``keep``: output the missing entries as NaN.

        Examples
        --------
        >>> from arctern_spark import GeoDataFrame
        >>> import numpy as np
        >>> data = {
        ...     "A": range(1),
        ...     "B": np.arange(1.0),
        ...     "other_geom": range(1),
        ...     "geometry": ["POINT (0 0)"],
        ... }
        >>> gdf = GeoDataFrame(data, geometries=["geometry"], crs=["epsg:4326"])
        >>> print(gdf.to_json(geometry="geometry"))
        {"type": "FeatureCollection", "features": [{"id": "0", "type": "Feature", "properties": {"A": 0.0, "B": 0.0, "other_geom": 0.0}, "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}}]}
        """
        return json.dumps(self._to_geo(na=na, show_bbox=show_bbox, geometry=geometry), **kwargs)
