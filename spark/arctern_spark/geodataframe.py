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

from itertools import zip_longest

from databricks.koalas import DataFrame, Series
from arctern_spark.geoseries import GeoSeries
from arctern_spark.scala_wrapper import GeometryUDT

_crs_dtype = str


class GeoDataFrame(DataFrame):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, geometries=None, crs=None):
        # (col_name, crs) dict to store crs data of those columns are GeoSeries
        self._crs_for_cols = {}
        if isinstance(data, GeoSeries):
            self._crs_for_cols[data.name] = data.crs
        elif isinstance(data, DataFrame):
            for col in data.columns:
                if isinstance(data[col].spark.data_type, GeometryUDT):
                    self._crs_for_cols[col] = None
                if isinstance(data[col], GeoSeries):
                    self._crs_for_cols[col] = data[col].crs
            data = data._internal_frame

        super(GeoDataFrame, self).__init__(data, index, columns, dtype, copy)

        if geometries is None:
            if "geometry" in self.columns:
                geometries = ["geometry"]
            else:
                geometries = []
        self._geometry_column_names = set()
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
                result._geometry_column_names = set()
                result._set_geometries(geometry_column_names, geometry_crs)

        return result

    def __setitem__(self, key, value):
        super(GeoDataFrame, self).__setitem__(key, value)
        if isinstance(value, GeoSeries):
            self._crs_for_cols[key] = value.crs
        elif isinstance(value, GeoDataFrame):
            for col in value._crs_for_cols.keys():
                v = value[col]
                if hasattr(v, "crs"):
                    self._crs_for_cols[col] = v.crs
        else:
            if isinstance(key, list):
                pass
            else:
                key = [key]
            for col in key:
                self._crs_for_cols.pop(col)

    def dissolve(self, by, col="geometry", aggfunc="first", as_index=True):
        if col not in self._geometry_column_names:
            raise ValueError("`col` must be a column in geometries columns which set by `set_geometry`")
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
        result = super().merge(right, how, on, left_on, right_on,
                               left_index, right_index, suffixes)
        result = GeoDataFrame(result)

        for col in result.columns:
            kser = result[col]
            if isinstance(kser, GeoSeries):
                pick = self
                if col.endswith(suffixes[0]):
                    col = col[:-len(suffixes[0])]
                elif col.endswith(suffixes[1]):
                    col = col[:-len(suffixes[1])]
                    pick = right
                elif col in right.columns:
                    pick = right

                kser.set_crs(pick._crs_for_cols.get(col, None))

        for col in self._geometry_column_names:
            result._geometry_column_names.add(col + suffixes[0])
        for col in right._geometry_column_names:
            result._geometry_column_names.add(col + suffixes[1])

        return result
