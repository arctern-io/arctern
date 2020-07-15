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
# pylint: disable=too-many-public-methods, unused-argument, redefined-builtin,protected-access
import json

from itertools import zip_longest
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from arctern import GeoSeries
import arctern.tools


class GeoDataFrame(DataFrame):

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, geometries=None, crs=None):
        geometry_column_names = []
        crs_for_cols = {}
        if isinstance(data, GeoSeries):
            crs_for_cols[data.name] = data.crs
            geometry_column_names.append(data.name)
        elif isinstance(data, DataFrame):
            for col in data.columns:
                if isinstance(data[col], GeoSeries):
                    crs_for_cols[col] = data[col].crs
                    geometry_column_names.append(col)
        super(GeoDataFrame, self).__init__(data, index, columns, dtype, copy)

        self._geometry_column_names = None
        self._crs_for_cols = None
        self._geometry_column_names = geometry_column_names
        self._crs_for_cols = crs_for_cols
        if geometries is None:
            geometries = []
        if crs is None or isinstance(crs, str):
            crs = [crs] * len(geometries)
        if not isinstance(crs, list):
            raise TypeError("The type of crs should be str or list!")
        if len(geometries) < len(crs):
            raise ValueError("The length of crs should less than geometries!")

        # align crs and cols, simply fill None to crs
        for col, _crs in zip_longest(geometries, crs):
            if col not in self._geometry_column_names:
                self[col] = GeoSeries(self[col], crs=_crs)
                self[col].invalidate_sindex()
                self._crs_for_cols[col] = _crs
                self._geometry_column_names.append(col)

    # pylint: disable=protected-access
    def set_geometry(self, col, inplace=False, crs=None):
        """
        Sets an existing column in the GeoDataFrame to a geometry column, which is used to perform geometric calculations later.

        Parameters
        ----------
        col: list
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
        >>> from arctern import GeoDataFrame
        >>> import numpy as np
        >>> data = {
        ...    "A": range(5),
        ...    "B": np.arange(5.0),
        ...    "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        ...    "geo2": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        ... }
        >>> gdf = GeoDataFrame(data, geometries=["geo1"], crs=["epsg:4326"])
        >>> print(gdf.geometries_name)
        ['geo1']
        >>> gdf.set_geometry(col="geo2",crs="epsg:4326",inplace=True)
        ['geo1','geo2']
        """
        if inplace:
            frame = self
        else:
            frame = self.copy()
            frame._geometry_column_names = self.geometries_name
            frame._crs_for_cols = self.crs

        geometry_cols = frame._geometry_column_names
        if not isinstance(frame[col], GeoSeries):
            frame[col] = GeoSeries(frame[col], crs=crs)
            geometry_cols.append(col)
            self._crs_for_cols[col] = crs
        if col in geometry_cols:
            if crs is not None:
                self._crs_for_cols[col] = crs
                frame[col].set_crs(crs)

        return frame

    # pylint: disable=arguments-differ
    def to_json(self, na="null", show_bbox=False, geometry=None, **kwargs):
        """
        Returns a GeoJSON string representation of the GeoDataFrame.

        Parameters
        ----------
        na: {'null', 'drop', 'keep'}
            Indicates how to output missing (NaN) values in the GeoDataFrame, by default 'null'.
            * 'null': Outputs the missing entries as JSON null.
            * 'drop': Removes the property from the feature. This applies to each feature individually so that features may have different properties.
            * 'keep': Outputs the missing entries as NaN.
        show_bbow: bool, optional
            Indicates whether to include bbox (bounding box) in the GeoJSON string, by default False.
            * *True:* Includes bounding box in the GeoJSON string.
            * *False:* Do not include bounding box in the GeoJSON string.

        **kwargs:
            Parameters to pass to `jump.dumps`.

        Returns
        -------
        Series
            Sequence of geometries in GeoJSON format.

        Examples
        --------
        >>> from arctern import GeoDataFrame
        >>> import numpy as np
        >>> data = {
        ...     "A": range(1),
        ...     "B": np.arange(1.0),
        ...     "other_geom": range(1),
        ...     "geometry": ["POINT (0 0)"],
        ... }
        >>> gdf = GeoDataFrame(data, geometries=["geometry"], crs=["epsg:4326"])
        >>> print(gdf.to_json(col="geometry"))
        {"type": "FeatureCollection", "features": [{"id": "0", "type": "Feature", "properties": {"A": 0, "B": 0.0, "other_geom": 0}, "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}}]}
        """
        return json.dumps(self._to_geo(na=na, show_bbox=show_bbox, geometry=geometry), **kwargs)

    def _to_geo(self, na="null", show_bbox=False, geometry=None, **kwargs):
        geo = {
            "type": "FeatureCollection",
            "features": list(self.iterfeatures(na=na, show_bbox=show_bbox, geometry=geometry, **kwargs))
        }
        if show_bbox is True:
            geo["bbox"] = self[geometry].envelope_aggr().bbox[0]

        return geo

    def iterfeatures(self, na="null", show_bbox=False, geometry=None):
        if na not in ["null", "drop", "keep"]:
            raise ValueError("Unknown na method {0}".format(na))
        if geometry not in self._geometry_column_names:
            raise ValueError("{} is not a geometry column".format(geometry))
        ids = np.array(self.index, copy=False)
        geometries = self[geometry].as_geojson()
        geometries_bbox = self[geometry].bbox

        propertries_cols = self.columns.difference([geometry])

        if len(propertries_cols) > 0:
            properties = self[propertries_cols].astype(object).values
            if na == "null":
                properties[pd.isnull(self[propertries_cols]).values] = None

            for i, row in enumerate(properties):
                geom = geometries[i]

                if na == "drop":
                    propertries_items = {
                        k: v for k, v in zip(propertries_cols, row) if not pd.isnull(v)
                    }
                else:
                    propertries_items = {}
                    for k, v in zip(propertries_cols, row):
                        propertries_items[k] = v
                    # propertries_items = {
                    #     k: v for k, v in zip(propertries_cols, row)
                    # }

                feature = {
                    "id": str(ids[i]),
                    "type": "Feature",
                    "properties": propertries_items,
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

    # pylint: disable=protected-access
    def reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill=""):
        df = super(GeoDataFrame, self).reset_index(level, drop, inplace, col_level, col_fill)
        if not inplace:
            gdf = GeoDataFrame(df)
            gdf._geometry_column_names = self._geometry_column_names
            gdf._crs_for_cols = self._crs_for_cols
            return gdf
        return None

    def to_geopandas(self):
        """
        Transforms an arctern.GeoDataFrame object to a geopandas.GeoDataFrame object.

        Returns
        --------
        geopandas.GeoDataFrame
            A geopandas.GeoDataFrame object.

        Examples
        --------
        >>> from arctern import GeoDataFrame
        >>> import numpy as np
        >>> import geopandas
        >>> data = {
        ...     "A": range(5),
        ...     "B": np.arange(5.0),
        ...     "other_geom": range(5),
        ...     "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        ...     "geo2": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        ... }
        >>> gdf = GeoDataFrame(data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
        >>> pdf = gdf.to_geopandas()
        >>> pdf.set_geometry("geo1", inplace=True)
        >>> print(pdf.geometry.name)
        "geo1"
        >>> type(pdf["geo1"])
        <class 'geopandas.geoseries.GeoSeries'>
        """
        import geopandas
        copy_df = self.copy()
        if len(self._geometry_column_names) > 0:
            for col in self._geometry_column_names:
                copy_df[col] = Series(copy_df[col].to_geopandas())
        return geopandas.GeoDataFrame(copy_df.values, columns=copy_df.columns.values.tolist())

    # pylint: disable=protected-access
    @classmethod
    def from_geopandas(cls, pdf):
        """
        Constructs an arctern.GeoSeries object from a geopandas.GeoSeries object.

        Parameters
        ----------
        pdf: geopandas.GeoDataFrame
            A geopandas.GeoDataFrame object.

        Returns
        -------
        GeoDataFrame
            An arctern.GeoDataFrame object.

        Examples
        --------
        >>> from arctern import GeoDataFrame
        >>> import geopandas
        >>> from shapely.geometry import Point,LineString
        >>> import numpy as np
        >>> data = {
        ...     "A": range(5),
        ...     "B": np.arange(5.0),
        ...     "other_geom": range(5),
        ...     "geometry": [Point(x, y) for x, y in zip(range(5), range(5))],
        ...     "copy_geo": [Point(x + 1, y + 1) for x, y in zip(range(5), range(5))],
        ... }
        >>> pdf = geopandas.GeoDataFrame(data, geometry="geometry", crs='epsg:4326')
        >>> gdf = GeoDataFrame.from_geopandas(pdf)
        >>> gdf.geometries_name
        ["geometry", "copy_geo"]
        >>> type(gdf["geometry"])
        <class 'arctern.geoseries.geoseries.GeoSeries'>
        >>> gdf["geometry"].crs
        'EPSG:4326'
        """
        import geopandas
        import shapely
        if not isinstance(pdf, geopandas.GeoDataFrame):
            raise TypeError(f"pdf must be {geopandas.GeoSeries}, got {type(pdf)}")
        result = cls(pdf.values, columns=pdf.columns.values.tolist())
        column_names = pdf.columns.values.tolist()
        for col in column_names:
            if isinstance(pdf[col][0], shapely.geometry.base.BaseGeometry):
                geo_col = GeoSeries.from_geopandas(geopandas.GeoSeries(pdf[col]))
                result[col] = geo_col
                result._geometry_column_names.append(col)
                if isinstance(pdf[col], geopandas.GeoSeries):
                    result._crs_for_cols[col] = pdf[col].crs
                else:
                    result._crs_for_cols[col] = None
        return result

    def dissolve(self, by=None, col="geometry", aggfunc="first", as_index=True):
        """
        Dissolves geometries within `by` into a single observation.

        This is accomplished by applying the `unary_union` method to all geometries within a group.

        Observations associated with each `by` group will be aggregated using the `aggfunc`.

        Parameters
        ----------
        by: str
            Column whose values define groups to be dissolved, by default None.
        aggfunc: function or str
            Aggregation function for manipulation of data associated with each group, by default "first". Passed to pandas `groupby.agg` method.
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
        >>> from arctern import GeoDataFrame
        >>> import numpy as np
        >>> data = {
        ...     "A": range(5),
        ...     "B": np.arange(5.0),
        ...     "other_geom": [1, 1, 1, 2, 2],
        ...     "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        ... }
        >>> gdf = GeoDataFrame(data, geometries=["geo1"], crs=["epsg:4326"])
        >>> gdf.dissolve(by="other_geom", col="geo1")
                                        geo1  A    B
        other_geom
        1           MULTIPOINT (0 0,1 1,2 2)  0  0.0
        2               MULTIPOINT (3 3,4 4)  3  3.0
        """
        data = self.drop(labels=col, axis=1)
        aggregated_data = data.groupby(by=by).agg(aggfunc)

        def merge_geometries(block):
            merge_geom = block.unary_union()
            return merge_geom[0]

        g = self.groupby(by=by, group_keys=False)[col].agg(
            merge_geometries
        )

        crs_str = self[col].crs
        aggregated_geometry = GeoDataFrame(g, geometries=[col], crs=[crs_str])

        aggregated = aggregated_geometry.join(aggregated_data)

        if not as_index:
            aggregated = aggregated.reset_index()

        return aggregated

    @property
    def geometries_name(self):
        return self._geometry_column_names

    @property
    def crs(self):
        """
        The Coordinate Reference System (CRS) of arctern.GeoDataFrame.

        Returns
        --------
        crs: list
            The Coordinate Reference System (CRS).

        Examples
        --------
        >>> from arctern import GeoDataFrame
        >>> import numpy as np
        >>> data = {
        ...    "A": range(5),
        ...    "B": np.arange(5.0),
        ...    "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        ...    "geo2": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        ... }
        >>> gdf = GeoDataFrame(data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
        >>> gdf.crs
        ["epsg:4326", "epsg:3857"]
        """
        return self._crs_for_cols

    # pylint: disable=too-many-arguments
    def merge(
            self,
            right,
            how="inner",
            on=None,
            left_on=None,
            right_on=None,
            left_index=False,
            right_index=False,
            sort=False,
            suffixes=("_x", "_y"),
            copy=True,
            indicator=False,
            validate=None,
    ):
        """
        Merges two GeoDataFrame objects with a database-style join.

        Returns
        -------
            GeoDataFrame or pandas.DataFrame
            Returns a GeoDataFrame if a geometry column is present; otherwise, returns a pandas DataFrame.

        Examples
        -------
        >>> from arctern import GeoDataFrame
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
        >>> gdf1.merge(gdf2, left_on="A", right_on="A")
        A    B  other_geom     geometry     location
        0  0  0.0           0  POINT (0 0)  POINT (3 0)
        1  1  1.0           1  POINT (1 1)  POINT (1 6)
        2  2  2.0           2  POINT (2 2)  POINT (2 4)
        3  3  3.0           3  POINT (3 3)  POINT (3 4)
        4  4  4.0           4  POINT (4 4)  POINT (4 2)
        """
        result = DataFrame.merge(self, right, how, on, left_on, right_on,
                                 left_index, right_index, sort, suffixes,
                                 copy, indicator, validate)
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

    # pylint: disable=protected-access
    @classmethod
    def from_file(cls, filename, **kwargs):
        """
        Constructs a GeoDataFrame from a file or url.

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
        return arctern.tools.file._read_file(filename, **kwargs)

    # pylint: disable=protected-access
    def to_file(self, filename, driver="ESRI Shapefile", geometry=None, schema=None, index=None, crs=None, **kwargs):
        """
        Writes a GeoDataFrame to a file.

        Parameters
        ----------
        df: GeoDataFrame
            GeoDataFrame to be written.
        filename: str
            File path or file handle to write to.
        driver: str
            The OGR format driver used to write the vector file, by default 'ESRI Shapefile'.
        schema: dict
            * If specified, the schema dictionary is passed to Fiona to better control how the file is written.
            * If None (default), this function determines the schema based on each column's dtype.
        index: bool
            * If None (default), writes the index into one or more columns only if the index is named, is a MultiIndex, or has a non-integer data type.
            * If True, writes index into one or more columns (for MultiIndex).
            * If False, no index is written.
        mode: str
            * 'a': Append
            * 'w' (default): Write
        crs: str
            * If specified, the CRS is passed to Fiona to better control how the file is written.
            * If None (default), this function determines the crs based on crs df attribute.
        col: str
            Specifys geometry column, by default None.

        **kwargs:
        Parameters to be passed to ``fiona.open``. Can be used to write to multi-layer data, store data within archives (zip files), etc.

        Notes
        -----
        The format drivers will attempt to detect the encoding of your data, but may fail. In this case, the proper encoding can be specified explicitly by using the encoding keyword parameter, e.g. ``encoding='utf-8'``.

        Examples
        --------
        >>> from arctern import GeoDataFrame
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
        3  3  3.0           3  POINT (4 4)  POINT (5 5)  POINT (3 3)
        4  4  4.0           4  POINT (5 5)  POINT (6 6)  POINT (4 4)
        """
        arctern.tools.file._to_file(self, filename=filename, driver=driver,
                                    schema=schema, index=index, geometry=geometry, crs=crs, **kwargs)

    @property
    def _constructor(self):
        return GeoDataFrame

    @property
    def _constructor_expanddim(self):
        pass
