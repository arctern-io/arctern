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

from warnings import warn

from pandas import DataFrame,Series
from arctern import GeoSeries


class GeoDataFrame(DataFrame):
    _metadata = ["_crs", "_geometry_column_name"]
    _geometry_column_name = []
    _sindex = None
    _sindex_generated = False

    def __init__(self, *args, **kwargs):
        crs = kwargs.pop("crs", None)
        geometries = kwargs.pop("geometries", None)
        super(GeoDataFrame, self).__init__(*args, **kwargs)

        if geometries is None and "geometry" in self.columns:
            index = self.index
            try:
                self["geometry"] = GeoSeries(self["geometry"])
                self._geometry_column_name = ["geometry"]
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
                        self[geometry] = GeoSeries(self[geometry])
                    except TypeError:
                        pass
                crs_length = len(crs)
                geo_length = len(geometries)
                if crs_length < geo_length:
                    for i in range(0, geo_length - crs_length):
                        crs.append("None")

                for (crs_element, geo_element) in zip(crs, geometries):
                    self[geo_element].set_crs(crs_element)
            else:
                self._geometry_column_name = None

        self._invalidate_sindex()

    def set_geometries(self, cols, inplace=False, crs=None):
        if inplace:
            frame = self
        else:
            frame = self.copy()

        geos_crs = {}
        crs_length = len(crs)
        geo_length = len(cols)
        if crs_length < geo_length:
            for i in range(0, geo_length - crs_length):
                crs.append("None")
        for col, geo_crs in zip(cols, crs):
            geos_crs[col] = geo_crs
        for col in frame._geometry_column_name:
            geos_crs[col] = frame[col].crs

        geometry_cols = frame._geometry_column_name
        for col in cols:
            if col not in geometry_cols:
                frame[col] = GeoSeries(frame[col])

        frame._geometry_column_name = frame._geometry_column_name + cols

        for col in frame._geometry_column_name:
            frame[col].set_crs(geos_crs[col])

        if not inplace:
            return frame

    def to_geopandas(self):
        import geopandas
        if self._geometry_column_name is not None:
            for col in self._geometry_column_name:
                self[col] = Series(self[col].to_geopandas())
        return geopandas.GeoDataFrame(self.values ,columns=self.columns.values.tolist())

    @classmethod
    def from_geopandas(cls, pdf):
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
                result._geometry_column_name.append(col)
        return result

    def _invalidate_sindex(self):
        self._sindex = None
        self._sindex_generated = False

    def disolve(self, by=None, col="geometry", aggfunc="first", as_index=True):
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

    @property
    def _constructor(self):
        return GeoDataFrame

    def _constructor_expanddim():
        pass
