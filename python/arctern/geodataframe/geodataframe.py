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
# pylint: disable=too-many-public-methods, unused-argument, redefined-builtin

import pandas as pd
import numpy as np
from pandas import DataFrame
from arctern import GeoSeries
from warnings import warn

try:
    from rtree.core import RTreeError

    HAS_SINDEX = True
except ImportError:

    class RTreeError(Exception):
        pass


    HAS_SINDEX = False

DEFAULT_GEO_COLUMN_NAME = ["geometry"]


class GeoDataFrame(DataFrame):
    _metadata = ["_crs", "_geometry_column_name"]
    _geometry_column_name = DEFAULT_GEO_COLUMN_NAME
    _sindex = None
    _sindex_generated = False
    _crs = {}

    def __init__(self, *args, **kwargs):
        crs = kwargs.pop("crs", None)
        geometries = kwargs.pop("geometries", None)
        super(GeoDataFrame, self).__init__(*args, **kwargs)

        if geometries is None and "geometry" in self.columns:
            index = self.index
            try:
                self["geometry"] = GeoSeries(self["geometry"].values)
                geometries = ["geometries"]
                self._crs = None
            except TypeError:
                pass
            else:
                if self.index is not index:
                    self.index = index
        else:
            if geometries is not None:
                self._geometry_column_name = geometries
                for geometry in geometries:
                    try:
                        self[geometry] = GeoSeries(self[geometry].values)
                    except TypeError:
                        pass
                crs_length = len(crs)
                geo_length = len(geometries)
                if crs_length < geo_length:
                    for i in range(0, geo_length - crs_length):
                        crs.append("None")

                for (crs_element, geo_element) in zip(crs, geometries):
                    self._crs[geo_element] = crs_element
                    self[geo_element].set_crs(crs_element)
            else:
                self._geometry_column_name = None
                self._crs = None

        self._invalidate_sindex()

    @property
    def _constructor(self):
        pass

    def to_crs(self, col, epsg, inplace=False):
        if self._geometry_column_name.count(col) == 0:
            raise TypeError("Input column name {0} is not exist in dataframe".format(col))
        if inplace:
            gdf = self
        else:
            gdf = self.copy()
        geom = gdf[col].to_crs(epsg)
        gdf[col] = geom
        gdf._crs[col] = epsg
        return gdf

    def _invalidate_sindex(self):
        self._sindex = None
        self._sindex_generated = False

    def _generate_sindex(self):
        if not HAS_SINDEX:
            warn("Cannot generate spatial index: Missing package `rtree`.")
        else:
            from geopandas.sindex import SpatialIndex

            stream = (
                (i, item.bounds, idx)
                for i, (idx, item) in enumerate(self.geometry.iteritems())
                if pd.notnull(item) and not item.is_empty
            )
            try:
                self._sindex = SpatialIndex(stream)
            except RTreeError:
                pass
        self._sindex_generated = True

    @property
    def sindex(self):
        if not self._sindex_generated:
            self._generate_sindex()
        return self._sindex

    def disolve(self, by=None, col="geometry", aggfunc="first", as_index=True):
        data = self.drop(labels=col, axis=1)
        aggregated_data = data.groupby(by=by).agg(aggfunc)

        def merge_geometries(block):
            merge_geom = block.unary_union()
            return merge_geom[0]

        g = self.groupby(by=by, group_keys=False)[col].agg(
            merge_geometries
        )
        aggregated_geometry = GeoDataFrame(g)

        aggregated = aggregated_geometry.join(aggregated_data)

        if not as_index:
            aggregated = aggregated.reset_index()

        return aggregated

    def merge(self, *args, **kwargs):
        result = DataFrame.merge(self, *args, **kwargs)
        geo_col = self._geometry_column_name
        if isinstance(result, DataFrame) and geo_col in result:
            result.__class__ = GeoDataFrame
            result.crs = self.crs
            result._geometry_column_name = geo_col
            result._invalidate_sindex()
        elif isinstance(result, DataFrame) and geo_col not in result:
            result.__class__ = DataFrame
        return result
