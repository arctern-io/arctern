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
        elif isinstance(data, GeoDataFrame):
            for col in data.columns:
                if isinstance(data[col], GeoSeries):
                    self._crs_for_cols[col] = data[col].crs
            data = data._internal_frame

        super(GeoDataFrame, self).__init__(data, index, columns, dtype, copy)

        # TODO: do we need it?
        self._geometry_column_names = set()
        if geometries is None:
            if "geometry" in self.columns:
                geometries = ["geometry"]
            else:
                geometries = []

        self._set_geometries(geometries, crs=crs)

    # only for internal use
    def _set_geometries(self, cols, crs=None):
        assert isinstance(cols, list)
        if len(cols) == 0:
            return

        if crs is None or isinstance(crs, _crs_dtype):
            crs = [crs] * len(cols)
        else:
            assert isinstance(crs, list)
        # align crs and cols, simply fill None to crs
        crs.extend([None] * (len(cols) - len(crs)))

        for col, _crs in zip(cols, crs):
            if col not in self._geometry_column_names:
                self[col] = GeoSeries(self[col], crs=_crs)
                self._crs_for_cols[col] = _crs
                self._geometry_column_names.add(col)

    def set_geometry(self, col, crs):
        self._set_geometries([col], crs)

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(result, Series) and isinstance(result.spark.data_type, GeometryUDT):
            result.__class__ = GeoSeries
            result.set_crs(self._crs_for_cols[result.name])
        if isinstance(result, DataFrame):
            crs = {}
            geometry_column_names = []
            geometry_crs = []

            for col in self._crs_for_cols:
                if col in result.columns:
                    crs[col] = self._crs_for_cols[col]

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
                self._crs_for_cols[col] = value[col].crs
