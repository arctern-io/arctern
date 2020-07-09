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

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from arctern import GeoSeries
import arctern.tools


class GeoDataFrame(DataFrame):
    _metadata = ["_crs", "_geometry_column_name"]
    _geometry_column_name = []
    _crs = []

    def __init__(self, *args, **kwargs):
        crs = kwargs.pop("crs", None)
        geometries = kwargs.pop("geometries", None)
        super(GeoDataFrame, self).__init__(*args, **kwargs)

        if geometries is None and crs is not None:
            raise ValueError("No geometry column specified!")
        if geometries is None:
            self._geometry_column_name = []
        else:
            geo_length = len(geometries)
            if crs is None:
                crs = []
                crs_length = 0
            elif isinstance(crs, str):
                crs = [crs]
                crs_length = 1
            elif isinstance(crs, list):
                crs_length = len(crs)
                if geo_length < crs_length:
                    raise ValueError("The length of crs should less than geometries!")
            else:
                raise TypeError("The type of crs should be str or list!")
            for _i in range(0, geo_length - crs_length):
                crs.append(None)
            for (crs_element, geometry) in zip(crs, geometries):
                self[geometry] = GeoSeries(self[geometry])
                self[geometry].invalidate_sindex()
                self[geometry].set_crs(crs_element)

            self._geometry_column_name = geometries
            self._crs = crs

    def set_geometry(self, col, inplace=False, crs=None):
        """
        Add GeoDataFrame geometry columns using either existing column.

        Parameters
        ----------
        col : list
            The name of column to be setten as geometry column.
        inplace : bool, default false
            Modify the GeoDataFrame in place (do not create a new object)
        crs : str
            Coordinate system to use

        Returns
        -------
        GeoDataFrame
            An arctern.GeoDataFrame object.

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

        geometry_cols = frame.geometries_name
        geometry_crs = frame.crs
        if not isinstance(frame[col], GeoSeries):
            frame[col] = GeoSeries(frame[col])
            frame[col].set_crs(crs)
            geometry_cols.append(col)
            if crs is None:
                geometry_crs.append(None)
            else:
                geometry_crs.append(crs)
        if col in geometry_cols:
            index = geometry_cols.index(col)
            if crs is not None:
                geometry_crs[index] = crs
                frame[col].set_crs(crs)

        # for (crs, col) in zip(geometry_crs, geometry_cols):
        #     if crs is not None:
        #         frame[col].set_crs(crs)


        return frame

    # pylint: disable=arguments-differ
    def to_json(self, na="null", show_bbox=False, geometry=None, **kwargs):
        """
        Returns a GeoJSON representation of the ``GeoDataFrame`` as a string.

        Parameters
        ----------

        na : {'null', 'drop', 'keep'}, default 'null'
            Indicates how to output missing (NaN) values in the GeoDataFrame.
            See below.
        show_bbow : bool, optional, default: False
            Include bbox (bounds) in the geojson

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
        if geometry not in self._geometry_column_name:
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
        if self._geometry_column_name is not None:
            for col in copy_df.geometries_name:
                copy_df[col] = Series(copy_df[col].to_geopandas())
        return geopandas.GeoDataFrame(copy_df.values, columns=copy_df.columns.values.tolist())

    # pylint: disable=protected-access
    @classmethod
    def from_geopandas(cls, pdf):
        """
        Constructs an arctern.GeoSeries object from a geopandas.GeoSeries object.

        Parameters
        ----------
        pdf : geopandas.GeoDataFrame
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
                result.geometries_name.append(col)
                if isinstance(pdf[col], geopandas.GeoSeries):
                    result._crs.append(pdf[col].crs)
                else:
                    result._crs.append(None)
        return result

    def disolve(self, by=None, col="geometry", aggfunc="first", as_index=True):
        """
        Dissolve geometries within `groupby` into single observation.
        This is accomplished by applying the `unary_union` method
        to all geometries within a groupself.

        Observations associated with each `groupby` group will be aggregated
        using the `aggfunc`.

        Parameters
        ----------
        by : str, default None
            Column whose values define groups to be dissolved
        aggfunc : function or str, default "first"
            Aggregation function for manipulation of data associated
            with each group. Passed to pandas `groupby.agg` method.
        as_index : bool, default True
            If true, groupby columns become index of result.

        Returns
        -------
        arctern.GeoDataFrame
            An arctern.GeoDataFrame object.

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
        >>> gdf.disolve(by="other_geom", col="geo1")
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
        return self._geometry_column_name

    @property
    def crs(self):
        """
        The Coordinate Reference System (CRS) of arctern.GeoDataFrame.

        Returns
        --------
        crs : list
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
        return self._crs

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
        Merge two ``GeoDataFrame`` objects with a database-style join.

        Returns a ``GeoDataFrame`` if a geometry column is present; otherwise,
        returns a pandas ``DataFrame``.

        Returns
        -------
            GeoDataFrame or pandas.DataFrame

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
        if not isinstance(result, GeoDataFrame):
            return result
        left_geometries = self.geometries_name.copy()
        left_crs = self.crs
        if isinstance(right, GeoDataFrame):
            right_geometries = right.geometries_name.copy()
            right_crs = right.crs
        else:
            right_geometries = []
            right_crs = []
        result_cols = result.columns.values.tolist()
        result_geometries_name = result.geometries_name
        result_crs = result.crs
        strip_result = []
        for col in result_cols:
            col = col.replace(suffixes[0], '')
            col = col.replace(suffixes[1], '')
            strip_result.append(col)
        for i, element in enumerate(strip_result):
            if isinstance(result[result_cols[i]], GeoSeries):
                if element in left_geometries:
                    index = left_geometries.index(element)
                    result[result_cols[i]].set_crs(left_crs[index])
                    result_geometries_name.append(result_cols[i])
                    result_crs.append(left_crs[index])
                    continue
                if element in right_geometries:
                    index = right_geometries.index(element)
                    result[result_cols[i]].set_crs(right_crs[index])
                    result_geometries_name.append(result_cols[i])
                    result_crs.append(right_crs[index])
                    continue
        return result

    # pylint: disable=protected-access
    @classmethod
    def from_file(cls, filename, **kwargs):
        """
        Alternate constructor to create a ``GeoDataFrame`` from a file or url.

        Parameters
        -----------
        filename : str
            File path or file handle to read from.
        bbox : tuple or arctern.GeoSeries, default None
            Filter features by given bounding box, GeoSeries. Cannot be used
            with mask.
        mask : dict | arctern.GeoSeries | dicr, default None
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
            An arctern.GeoDataFrame object.
        """
        return arctern.tools.file._read_file(filename, **kwargs)

    # pylint: disable=protected-access
    def to_file(self, filename, driver="ESRI Shapefile", geometry=None, schema=None, index=None, crs=None, **kwargs):
        """
        Write the ``GeoDataFrame`` to a file.

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
        col : str, default None
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
