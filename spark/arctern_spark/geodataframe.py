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

import pandas as pd
from databricks.koalas import DataFrame, Series, get_option
from databricks.koalas.frame import REPR_PATTERN

import arctern_spark
from arctern_spark.geoseries import GeoSeries

_crs_dtype = str


class GeoDataFrame(DataFrame):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, geometries=None, crs=None):
        """
        A GeoDataFrame object is a Koalas.DataFrame that has columns with geometries.

        Parameters
        ----------
        data : numpy ndarray (structured or homogeneous), dict, pandas DataFrame, Spark DataFrame \
            Koalas Series, or GeoSeries
            Dict can contain Series, GeoSeries, arrays, constants, or list-like objects.
            If data is a dict, argument order is maintained for Python 3.6
            and later.
            Note that if ``data`` is a pandas DataFrame, a Spark DataFrame, a Koalas Series, and a GeoSeries,
            other arguments should not be used.
        index : Index or array-like
            Index to use for resulting frame. Will default to RangeIndex if
            no indexing information part of input data and no index provided.
        columns : Index or array-like
            Column labels to use for resulting frame. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        dtype : dtype, default None
            Data type to force. Only a single dtype is allowed. If None, infers the data type based on the data.
        copy : bool, default False
            Copy data from inputs. Only affects DataFrame or 2d ndarray input.
        geometries : list
            The name of columns which are setten as geometry columns.
        crs : str or list, default None
            Coordinate Reference System of the geometry objects.
            If ``crs`` is a string, sets all ``geometries`` columns' crs to param ``crs``.
            If ``crs`` is a list, sets ``geometries`` columns' crs with param ``crs`` elementwise.

        Examples
        ---------
        >>> from arctern_spark import GeoDataFrame
        >>> import numpy as np
        >>> data = {
        ...     "A": range(5),
        ...     "B": np.arange(5.0),
        ...     "other_geom": [1, 1, 1, 2, 2],
        ...     "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        ... }
        >>> gdf = GeoDataFrame(data, geometries=["geo1"], crs=["epsg:4326"])
        >>> gdf
           A    B  other_geom         geo1
        0  0  0.0           1  POINT (0 0)
        1  1  1.0           1  POINT (1 1)
        2  2  2.0           1  POINT (2 2)
        3  3  3.0           2  POINT (3 3)
        4  4  4.0           2  POINT (4 4)
        """

        # (col_name, crs) dict to store crs data of those columns are GeoSeries
        self._crs_for_cols = {}
        self._geometry_column_names = set()

        if isinstance(data, GeoSeries):
            self._crs_for_cols[data.name] = data.crs
            self._geometry_column_names.add(data.name)
        elif isinstance(data, DataFrame):
            for col in data.columns:
                from arctern_spark.scala_wrapper import GeometryUDT
                if isinstance(data[col].spark.data_type, GeometryUDT):
                    self._crs_for_cols[col] = None
                    self._geometry_column_names.add(col)
                if isinstance(data[col], GeoSeries):
                    self._crs_for_cols[col] = data[col].crs
                    self._geometry_column_names.add(col)
            data = data._internal_frame

        super(GeoDataFrame, self).__init__(data, index, columns, dtype, copy)

        if geometries is not None:
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
                self[col] = GeoSeries(self[col], crs=_crs)
                self._crs_for_cols[col] = _crs
                self._geometry_column_names.add(col)

    def set_geometry(self, col, inplace=False, crs=None):
        """
        Sets an existing column in the GeoDataFrame to a geometry column, which is used to perform geometric calculations later.

        Setting an column to a geometry column will attemp to construct geometry for each row of this column,
        so it should be WKT or WKB formed data.

        Parameters
        ----------
        col: str
            The name of column to be setten as a geometry column.
        inplace: bool, default false
            Whether to modify the GeoDataFrame in place.

            * *True:* Modifies the GeoDataFrame in place (does not create a new object).
            * *False:* Does not modifies the GeoDataFrame in place.

        crs: str
            The coordinate reference system to use.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame object.

        Examples
        --------
        >>> from arctern_spark import GeoDataFrame
        >>> import numpy as np
        >>> data = {
        ...    "A": range(5),
        ...    "B": np.arange(5.0),
        ...    "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        ...    "geo2": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        ... }
        >>> gdf = GeoDataFrame(data, geometries=["geo1"], crs=["epsg:4326"])
        >>> gdf.geometries_name
        {'geo1'}
        >>> gdf.set_geometry(col="geo2", crs="epsg:4326", inplace=True)
        >>> gdf.geometries_name # doctest: +SKIP
        {'geo1','geo2'}
        """
        if inplace:
            frame = self
        else:
            frame = self.copy()
        frame._set_geometries([col], crs)

    @property
    def geometries_name(self):
        return self._geometry_column_names

    def __getitem__(self, item):
        result = super().__getitem__(item)
        from arctern_spark.scala_wrapper import GeometryUDT
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

    def _get_or_create_repr_pandas_cache(self, n):
        if not hasattr(self, "_repr_pandas_cache") or n not in self._repr_pandas_cache:
            pdf = self.head(n + 1)._to_internal_pandas()
            for col in self._geometry_column_names:
                pdf[col] = self[col].to_wkt()._to_internal_pandas()
            self._repr_pandas_cache = {n: pdf}
        return self._repr_pandas_cache[n]

    def __repr__(self):
        max_display_count = get_option("display.max_rows")
        if max_display_count is None:
            pdf = self.to_pandas()
            for col in self._geometry_column_names:
                pdf[col] = self[col].to_wkt()._to_internal_pandas()
            return pdf.to_string()

        pdf = self._get_or_create_repr_pandas_cache(max_display_count)
        pdf_length = len(pdf)
        pdf = pdf.iloc[:max_display_count]
        if pdf_length > max_display_count:
            repr_string = pdf.to_string(show_dimensions=True)
            match = REPR_PATTERN.search(repr_string)
            if match is not None:
                nrows = match.group("rows")
                ncols = match.group("columns")
                footer = "\n\n[Showing only the first {nrows} rows x {ncols} columns]".format(
                    nrows=nrows, ncols=ncols
                )
                return REPR_PATTERN.sub(footer, repr_string)
        return pdf.to_string()

    def copy(self, deep=None):
        gdf = GeoDataFrame(self._internal)
        gdf._crs_for_cols = self._crs_for_cols
        gdf._geometry_column_names = self._geometry_column_names
        return gdf

    def dissolve(self, by, col="geometry", aggfunc="first", as_index=True):
        """
        Dissolves geometries within ``by`` into a single observation.

        This is accomplished by applying the ``unary_union`` method to all geometries within a group.

        Observations associated with each ``by`` group will be aggregated using the ``aggfunc``.

        Parameters
        ----------
        by: str
            Column whose values define groups to be dissolved, by default None.
        aggfunc: function or str
            Aggregation function for manipulation of data associated with each group, by default "first". Passed to Koalas ``groupby.agg`` method.
        as_index: bool
            Whether to use the ``by`` column as the index of result, by default True.

            * *True:* The ``by`` column becomes the index of result.
            * *False:* The result uses the default ascending index that starts from 0.
        Returns
        -------
        GeoDataFrame
            A GeoDataFrame object.

        Examples
        --------
        >>> from arctern_spark import GeoDataFrame
        >>> import numpy as np
        >>> data = {
        ...     "A": range(5),
        ...     "B": np.arange(5.0),
        ...     "other_geom": [1, 1, 1, 2, 2],
        ...     "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        ... }
        >>> gdf = GeoDataFrame(data, geometries=["geo1"], crs=["epsg:4326"])
        >>> gdf.dissolve(by="other_geom", col="geo1") # doctest: +NORMALIZE_WHITESPACE
                    A    B                              geo1
        other_geom
        1           0  0.0  MULTIPOINT ((0 0), (1 1), (2 2))
        2           3  3.0         MULTIPOINT ((3 3), (4 4))
        """

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
        """
        Merge GeoDataFrame objects with a database-style join.

        The index of the resulting GeoDataFrame will be one of the following:
            - 0...n if no index is used for merging
            - Index of the left GeoDataFrame if merged only on the index of the right GeoDataFrame
            - Index of the right GeoDataFrame if merged only on the index of the left GeoDataFrame
            - All involved indices if merged using the indices of both GeoDataFrames
                e.g. if `left` with indices (a, x) and `right` with indices (b, x), the result will
                be an index (x, a, b)

        Parameters
        ----------
        right: Object to merge with.
        how: Type of merge to be performed.
            {'left', 'right', 'outer', 'inner'}, default 'inner'

            * *left:* use only keys from left frame, similar to a SQL left outer join; preserve key
                order.
            * *right:* use only keys from right frame, similar to a SQL right outer join; preserve key
                order.
            * *outer:* use union of keys from both frames, similar to a SQL full outer join; sort keys
                lexicographically.
            * *inner:* use intersection of keys from both frames, similar to a SQL inner join;
                preserve the order of the left keys.
        on: Column or index level names to join on. These must be found in both GeoDataFrames. If ``on``
            is None and not merging on indexes then this defaults to the intersection of the
            columns in both GeoDataFrames.
        left_on: Column or index level names to join on in the left GeoDataFrame. Can also
            be an array or list of arrays of the length of the left GeoDataFrame.
            These arrays are treated as if they are columns.
        right_on: Column or index level names to join on in the right GeoDataFrame. Can also
            be an array or list of arrays of the length of the right GeoDataFrame.
            These arrays are treated as if they are columns.
        left_index: Uses the index from the left GeoDataFrame as the join key(s). If it is a
            MultiIndex, the number of keys in the other GeoDataFrame (either the index or a number of
            columns) must match the number of levels.
        right_index: Uses the index from the right GeoDataFrame as the join key. Same caveats as
            left_index.
        suffixes: Suffix to apply to overlapping column names in the left and right side,
            respectively.

        Returns
        -------
            GeoDataFrame
            Returns a merged GeoDataFrame.

        Examples
        -------
        >>> from arctern_spark import GeoDataFrame
        >>> import numpy as np
        >>> data1 = {
        ...      "A": range(5),
        ...      "B": np.arange(5.0),
        ...      "other_geom": range(5),
        ...      "geometry": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        ... }
        >>> gdf1 = GeoDataFrame(data1, geometries=["geometry"], crs=["epsg:4326"])
        >>> data2 = {
        ...      "A": range(5),
        ...      "location": ["POINT (3 0)", "POINT (1 6)", "POINT (2 4)", "POINT (3 4)", "POINT (4 2)"],
        ... }
        >>> gdf2 = GeoDataFrame(data2, geometries=["location"], crs=["epsg:4326"])
        >>> gdf1.merge(gdf2, left_on="A", right_on="A") # doctest: +NORMALIZE_WHITESPACE
           A    B  other_geom     geometry     location
        0  0  0.0           0  POINT (0 0)  POINT (3 0)
        1  1  1.0           1  POINT (1 1)  POINT (1 6)
        2  3  3.0           3  POINT (3 3)  POINT (3 4)
        3  2  2.0           2  POINT (2 2)  POINT (2 4)
        4  4  4.0           4  POINT (4 4)  POINT (4 2)
        """
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

        gdf = self.copy()
        gdf['bbox'] = gdf[geometry].envelope
        gdf[geometry] = gdf[geometry].as_geojson()
        properties_cols = gdf.columns.difference([geometry, 'bbox']).tolist()

        if len(properties_cols) > 0:
            property_geo_cols = gdf._geometry_column_names.difference([geometry, 'bbox'])
            # since it could be more than one geometry columns in GeoDataFrame,
            # we transform those geometry columns as wkt formed string except column `geometry`.
            for property_geo_col in property_geo_cols:
                gdf[property_geo_col] = gdf[property_geo_col].to_wkt()

            gdf = gdf.to_pandas()
            geometries = gdf[geometry].values
            geometries_bbox = gdf['bbox'].apply(GeoSeries._calculate_bbox_from_wkb)
            ids = gdf.index

            properties = gdf[properties_cols].astype(object)
            if na == "null":
                properties[pd.isnull(properties).values] = None
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
            gdf = gdf.to_pandas()
            ids = gdf.index
            geometries = gdf[geometry].values
            geometries_bbox = gdf['bbox'].apply(GeoSeries._calculate_bbox_from_wkb)
            for fid, geom, bbox in zip(ids, geometries, geometries_bbox):
                feature = {
                    "id": str(fid),
                    "type": "Feature",
                    "properties": {},
                    "geometry": json.loads(geom) if geom else None,
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
        Constructs a GeoDataFrame from a file or URL.

        Parameters
        -----------
        filename: str
            File path or file handle to read from.
        bbox: tuple or GeoSeries
            Filters for geometries that spatially intersect with the provided bounding box. The bounding box can be a tuple ``(min_x, min_y, max_x, max_y)``, or a GeoSeries.

            * min_x: The minimum x coordinate of the bounding box.
            * min_y: The minimum y coordinate of the bounding box.
            * max_x: The maximum x coordinate of the bounding box.
            * max_y: The maximum y coordinate of the bounding box.

        mask: dict, GeoSeries
            Filters for geometries that spatially intersect with the geometries in ``mask``. ``mask`` should have the same crs with the GeoSeries that calls this method.
        rows: int or slice
            Rows to load.

            * If ``rows`` is an integer *n*, this function loads the first *n* rows.
            * If ``rows`` is a slice object (for example, *[start, end, step]*), this function loads rows by skipping over rows.

                * *start:* The position to start the slicing, by default 0.
                * *end:* The position to end the slicing.
                * *step:* The step of the slicing, by default 1.

        **kwargs:
            Parameters to be passed to the ``open`` or ``BytesCollection`` method in the fiona library when opening the file. For more information on possible keywords, type ``import fiona; help(fiona.open)``.

        Notes
        -------
        ``bbox`` and ``mask`` cannot be used together.

        Returns
        --------
        GeoDataFrame
            A GeoDataFrame read from file.
        """
        return arctern_spark.file.read_file(filename, **kwargs)

    def to_file(self, filename, driver="ESRI Shapefile", geometry=None, schema=None, index=None, crs=None, **kwargs):
        """
        Write a GeoDataFrame to a file.

        Parameters
        ----------
        filename: str
            File path or file handle to write to.
        driver: str
            The OGR format driver used to write the vector file, by default 'ESRI Shapefile'.
        schema: dict
            Data schema.

            * If specified, the schema dictionary is passed to Fiona to better control how the file is written.
            * If None (default), this function determines the schema based on each column's dtype.
        index: bool
            * If None (default), writes the index into one or more columns only if the index is named, is a MultiIndex, or has a non-integer data type.
            * If True, writes index into one or more columns (for MultiIndex).
            * If False, no index is written.
        mode: str
            Mode of writing data to file.
            * 'a': Append
            * 'w' (default): Write
        crs: str
            * If specified, the CRS is passed to Fiona to better control how the file is written.
            * If None (default), this function determines the crs based on crs df attribute.
        geometry: str
            Specifys geometry column, by default None.

        **kwargs:
        Parameters to be passed to ``fiona.open``. Can be used to write to multi-layer data, store data within archives (zip files), etc.

        Notes
        -----
        The format drivers will attempt to detect the encoding of your data, but may fail. In this case, the proper encoding can be specified explicitly by using the encoding keyword parameter, e.g. ``encoding='utf-8'``.

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
        >>> read_gdf.sort_index(inplace=True)
        >>> read_gdf
           A    B  other_geom         geo2         geo3     geometry
        0  0  0.0           0  POINT (1 1)  POINT (2 2)  POINT (0 0)
        1  1  1.0           1  POINT (2 2)  POINT (3 3)  POINT (1 1)
        2  2  2.0           2  POINT (3 3)  POINT (4 4)  POINT (2 2)
        3  3  3.0           3  POINT (4 4)  POINT (5 5)  POINT (3 3)
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
            # raise NotImplementedError("show bbox is not implemented yet.")
            geo["bbox"] = self[kwargs.get("geometry")].envelope_aggr().bbox[0]

        return geo

    # pylint: disable=arguments-differ
    def to_json(self, na="null", show_bbox=False, geometry='geometry', **kwargs):
        """
        Return a GeoJSON representation of the ``GeoDataFrame`` as a string.

        Parameters
        ----------
        na : {'null', 'drop', 'keep'}
            Indicates how to output missing (NaN) values in the GeoDataFrame, by default 'null'.

            * 'null': Outputs the missing entries as JSON null.
            * 'drop': Removes the property from the feature. This applies to each feature individually so that features may have different properties.
            * 'keep': Outputs the missing entries as NaN.

        show_bbox : bool, optional, default: False
            Include bbox (bounds) in the geojson
        geometry : str, optional, default 'geometry'
            Specify geometry column.

        **kwargs:
            Parameters to pass to `jump.dumps`.

        Returns
        -------
        Series
            Sequence of geometries in GeoJSON format.

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
        {"type": "FeatureCollection", "features": [{"id": "0", "type": "Feature", "properties": {"A": 0, "B": 0.0, "other_geom": 0}, "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}}]}
        """
        return json.dumps(self._to_geo(na=na, show_bbox=show_bbox, geometry=geometry), **kwargs)
