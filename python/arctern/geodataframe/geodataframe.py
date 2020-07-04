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

import numpy as np
import pandas as pd
from arctern import GeoSeries
from pandas import DataFrame, Series


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
        elif geometries is None:
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
            for i in range(0, geo_length - crs_length):
                crs.append(None)
            for (crs_element, geometry) in zip(crs, geometries):
                self[geometry] = GeoSeries(self[geometry])
                self[geometry].invalidate_sindex()
                self[geometry].set_crs(crs_element)

            self._geometry_column_name = geometries
            self._crs = crs

    def set_geometry(self, col, inplace=False, crs=None):
        if inplace:
            frame = self
        else:
            frame = self.copy()

        geometry_cols = frame._geometry_column_name
        geometry_crs = frame._crs
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
        return geopandas.GeoDataFrame(copy_df.values, columns=copy_df.columns.values.tolist())

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

    @property
    def geometries_name(self):
        return self._geometry_column_name

    @property
    def crs(self):
        return self._crs

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
        result = DataFrame.merge(self, right, how, on, left_on, right_on,
                                 left_index, right_index, sort, suffixes,
                                 copy, indicator, validate)
        if not isinstance(result, GeoDataFrame):
            return result
        left_geometries = self.geometries_name
        left_crs = self.crs
        right_geometries = right.geometries_name
        right_crs = right.crs
        result_cols = result.columns.values.tolist()
        result_geometries_name = result.geometries_name
        result_crs = result.crs
        strip_result = []
        for col in result_cols:
            col = col.replace('_x', '')
            col = col.replace('_y', '')
            strip_result.append(col)
        for i in range(0, len(strip_result)):
            if isinstance(result[result_cols[i]], GeoSeries):
                if strip_result[i] in left_geometries:
                    index = left_geometries.index(strip_result[i])
                    result[result_cols[i]].set_crs(left_crs[index])
                    result_crs.append(left_crs[index])
                    continue
                if strip_result[i] in right_geometries:
                    index = right_geometries.index(strip_result[i])
                    result[result_cols[i]].set_crs(right_crs[index])
                    result_crs.append(right_crs[index])
                    continue
        return result

    @property
    def _constructor(self):
        return GeoDataFrame

    def _constructor_expanddim(self):
        pass
