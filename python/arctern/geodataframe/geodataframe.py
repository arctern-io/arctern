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
import json
from pandas import DataFrame, Series
from arctern import GeoSeries
import pandas as pd
import numpy as np


class GeoDataFrame(DataFrame):
    _metadata = ["_crs", "_geometry_column_name"]
    _geometry_column_name = []
    _sindex = {}
    _sindex_generated = {}

    def __init__(self, *args, **kwargs):
        crs = kwargs.pop("crs", None)
        geometries = kwargs.pop("geometries", None)
        super(GeoDataFrame, self).__init__(*args, **kwargs)

        if geometries is None and "geometry" in self.columns.values:
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
            if col not in geometry_cols and not isinstance(frame[col], GeoSeries):
                frame[col] = GeoSeries(frame[col])

        frame._geometry_column_name = frame._geometry_column_name + cols

        for col in frame._geometry_column_name:
            if frame[col].crs is None:
                frame[col].set_crs(geos_crs[col])

        if not inplace:
            return frame

    def to_json(self, na="null", show_bbox=False, col='geometry', **kwargs):
        return json.dumps(self._to_geo(na=na, show_bbox=show_bbox), **kwargs)

    def _to_geo(self, **kwargs):
        geo = {
            "type": "FeatureCollection",
            "features": list(self.iterfeatures(**kwargs))
        }

        if kwargs.get("show_bbox", False):
            geo["bbox"] = self[kwargs.get("col")].envelope_aggr()

        return geo

    def iterfeatures(self, na="null", show_bbox=False, col='geometry'):
        if na not in ["null", "drop", "keep"]:
            raise ValueError("Unknown na method {0}".format(na))
        if col not in self._geometry_column_name:
            raise ValueError("{} is not a geometry column".format(col))
        ids = np.array(self.index, copy=False)
        geometries = self[col].as_geojson()
        geometries_bbox = self[col].envelope
        print(geometries_bbox)

        propertries_cols = self.columns.difference([col])
        print(propertries_cols)

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
                    propertries_items = {k: v for k, v in zip(propertries_cols, row)}

                feature = {
                    "id": str(ids[i]),
                    "type": "Feature",
                    "properties": propertries_items,
                    "geometry": geom
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
        import geopandas
        copy_df = self.copy()
        if self._geometry_column_name is not None:
            for col in copy_df._geometry_column_name:
                copy_df[col] = Series(copy_df[col].to_geopandas())
        return geopandas.GeoDataFrame(copy_df.values , columns=copy_df.columns.values.tolist())

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
        geo_cols = self._geometry_column_name
        geos_crs = {}
        for col in geo_cols:
            geos_crs[col] = self[col].crs
        result_cols = result.columns.values.tolist()
        flag = True
        for col in geo_cols:
            if col in result_cols:
                flag = False

        if isinstance(result, DataFrame) and not flag:
            result.__class__ = GeoDataFrame
            for col in geo_cols:
                if col in result_cols:
                    result[col] = GeoSeries(result[col])
            for col in geo_cols:
                if col in result_cols:
                    result[col].set_crs(geos_crs[col])
                    result._geometry_column_name.append(col)
            result._invalidate_sindex()
        elif isinstance(result, DataFrame) and flag:
            result.__class__ = DataFrame
        return result

    @property
    def _constructor(self):
        return GeoDataFrame

    def _constructor_expanddim(self):
        pass
